"""Train the short-axis CMR semantic teacher on paired UKB training subjects."""
import argparse
import json

import torch

from SaCMR_ECG_Align import engine_cmr_sem_teacher as engine
from SaCMR_ECG_Align.common import (
    add_common_args, atomic_torch_save, checkpoint_payload, configure_run,
    make_loaders, make_optimizer, make_writer, norm_tensors, phenotype_normalization,
    record_epoch, resume_run, write_json, PACKAGE_DIR, PROJECT_DIR,
)
from SaCMR_ECG_Align.models import (
    SemanticTeacher, load_sa_encoder, set_cmr_trainable, teacher_from_checkpoint,
)
from util.align_common import append_metrics_csv, append_phenotype_metrics_csv


def get_args_parser():
    parser = add_common_args(argparse.ArgumentParser('Short-axis CMR semantic teacher'), 'teacher')
    parser.add_argument('--sa_pretrained_weights', default=str(
        PROJECT_DIR / 'SaCMR_encoder/outputs/mae_vit_base_patch16_validated/checkpoint-latest.pth'))
    parser.add_argument('--sem_hidden_dim', type=int, default=768)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=50)
    parser.add_argument('--unfreeze_last_blocks', type=int, default=2)
    return parser


def main(args):
    if args.freeze_encoder_epochs < 0 or args.unfreeze_last_blocks < 0 or args.sem_hidden_dim < 1:
        raise ValueError('Invalid teacher head/freeze configuration')
    output, device = configure_run(args, 'teacher')
    datasets, loaders, fingerprints = make_loaders(args, load_ecg=False)
    norm = phenotype_normalization(datasets['train'].phenotypes)
    if args.resume:
        teacher, loaded = teacher_from_checkpoint(args.resume, use_checkpoint=args.use_checkpoint)
        del loaded
    else:
        encoder, pretrained_info = load_sa_encoder(args.sa_pretrained_weights, use_checkpoint=args.use_checkpoint)
        teacher = SemanticTeacher(encoder, sem_hidden_dim=args.sem_hidden_dim,
                                  pheno_hidden_dim=args.pheno_hidden_dim)
        write_json(output / 'pretrained_info.json', pretrained_info)
    if args.unfreeze_last_blocks > len(teacher.cmr_encoder.blocks):
        raise ValueError('unfreeze_last_blocks exceeds encoder depth')
    teacher.to(device)
    modules = {'cmr_encoder': teacher.cmr_encoder, 'sem_head': teacher.sem_head,
               'pheno_head': teacher.pheno_head}
    # A resumed teacher loader freezes all modules for inference; heads train here.
    for head in (teacher.sem_head, teacher.pheno_head):
        head.requires_grad_(True)
    optimizer = make_optimizer(teacher, args.lr, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    start_epoch, best_metric, patience = 0, float('inf'), 0
    if args.resume:
        checkpoint = resume_run(args.resume, args, modules, optimizer, scaler, fingerprints, loaders)
        if checkpoint['phenotype_norm'] != norm:
            raise ValueError('Resume train phenotype normalization changed')
        start_epoch, best_metric, patience = checkpoint['epoch'] + 1, checkpoint['best_metric'], checkpoint['patience']
        del checkpoint
    mean, std = norm_tensors(norm, device)
    write_json(output / 'run_config.json', vars(args))
    write_json(output / 'phenotype_norm.json', norm)
    write_json(output / 'data_fingerprint.json', fingerprints)
    writer = make_writer(output)
    best_path = output / 'best-cmr-pheno.pth'
    config = vars(args)
    print(json.dumps({'event': 'configuration', 'learning_rate': args.lr,
                      'effective_batch_size': args.batch_size * args.accum_iter,
                      'encoder_parameters': sum(p.numel() for p in teacher.cmr_encoder.parameters()),
                      'start_epoch': start_epoch}), flush=True)
    try:
        for epoch in range(start_epoch, args.epochs):
            unfreeze = epoch >= args.freeze_encoder_epochs and args.unfreeze_last_blocks > 0
            set_cmr_trainable(teacher.cmr_encoder, train_last_blocks=unfreeze,
                              last_blocks=args.unfreeze_last_blocks)
            if epoch == args.freeze_encoder_epochs:
                patience = 0
            print(json.dumps({'event': 'epoch_start', 'epoch': epoch, 'encoder_unfrozen': unfreeze,
                              'trainable_encoder_parameters': sum(p.numel() for p in teacher.cmr_encoder.parameters() if p.requires_grad)}), flush=True)
            train_stats = engine.train_one_epoch(teacher, loaders['train'], optimizer, device, epoch,
                                                 scaler, mean, std, config)
            val_stats, details = engine.evaluate(teacher, loaders['val'], device, epoch, mean, std, config, split='val')
            metric = val_stats['pheno_selected_mae_raw_mean']
            is_best = metric < best_metric
            if is_best:
                best_metric, patience = metric, 0
            elif args.unfreeze_last_blocks == 0 or unfreeze:
                patience += 1
            # Early stopping does not prevent the requested unfreeze phase.
            payload = checkpoint_payload(args, epoch, modules, optimizer, scaler, norm, fingerprints,
                                         loaders, best_metric, patience, val_stats,
                                         encoder_config=teacher.encoder_config)
            if is_best:
                atomic_torch_save(payload, best_path)
            atomic_torch_save(payload, output / 'checkpoint-latest.pth')
            del payload
            append_metrics_csv(str(output / 'sem_metrics.csv'), 'val', epoch, 'current', val_stats)
            append_phenotype_metrics_csv(str(output / 'phenotype_metrics.csv'), 'val', epoch, 'current', details or [])
            record_epoch(output, epoch, train_stats, val_stats, writer)
            if args.best_patience and patience >= args.best_patience:
                print(json.dumps({'event': 'early_stop', 'epoch': epoch}), flush=True)
                break
        if not best_path.is_file():
            raise FileNotFoundError('No selected checkpoint to test: %s' % best_path)
        checkpoint = torch.load(best_path, map_location='cpu')
        for key, module in modules.items():
            module.load_state_dict(checkpoint[key], strict=True)
        best_epoch = checkpoint['epoch']
        del checkpoint
        test_stats, details = engine.evaluate(teacher, loaders['test'], device, best_epoch, mean, std, config, split='test')
        write_json(output / 'test_metrics.json', {'checkpoint': str(best_path), 'epoch': best_epoch, **test_stats})
        append_metrics_csv(str(output / 'sem_metrics.csv'), 'test', best_epoch, best_path.name, test_stats)
        append_phenotype_metrics_csv(str(output / 'phenotype_metrics.csv'), 'test', best_epoch, best_path.name, details or [])
        print(json.dumps({'event': 'complete', 'best_checkpoint': str(best_path), 'best_epoch': best_epoch,
                          'test': test_stats}), flush=True)
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
