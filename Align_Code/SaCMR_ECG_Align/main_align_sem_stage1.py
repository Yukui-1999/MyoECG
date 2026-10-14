"""Align ST-MEM ECG features to a frozen short-axis CMR semantic teacher."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from SaCMR_ECG_Align import engine_align_sem as engine
from SaCMR_ECG_Align.common import (
    add_common_args, atomic_torch_save, checkpoint_payload, configure_run,
    make_loaders, make_optimizer, make_writer, norm_tensors, record_epoch,
    resume_run, validate_checkpoint_context, write_json, PACKAGE_DIR,
)
from SaCMR_ECG_Align.models import build_ecg_encoder, teacher_from_checkpoint
from util.align_common import append_metrics_csv, append_phenotype_metrics_csv, build_pheno_head
from util.sem_ablation import validate_stage1_weights


def get_args_parser():
    parser = add_common_args(argparse.ArgumentParser('Short-axis CMR–ECG semantic alignment stage1'), 'stage1')
    parser.add_argument('--cmr_teacher_path', required=True)
    parser.add_argument('--ecg_config_path', default=str(PACKAGE_DIR / 'configs/ecg_sem_stage1.yaml'))
    parser.add_argument('--sem_loss_weight', type=float, default=1.0)
    parser.add_argument('--pheno_loss_weight', type=float, default=1.0)
    parser.add_argument('--rel_ecg_weight', type=float, default=0.5)
    parser.add_argument('--tau_phi', type=float, default=1.0)
    parser.add_argument('--tau_ecg', type=float, default=0.07)
    parser.add_argument('--retrieval_chunk_size', type=int, default=256)
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    validate_stage1_weights(vars(args))
    if not all(0 < x < float('inf') for x in (args.tau_phi, args.tau_ecg)) or args.retrieval_chunk_size < 1:
        raise ValueError('Temperatures and retrieval_chunk_size must be positive')
    args.cmr_teacher_path = str(Path(args.cmr_teacher_path).expanduser().resolve())
    if not Path(args.cmr_teacher_path).is_file():
        raise FileNotFoundError('Train a short-axis semantic teacher first: %s' % args.cmr_teacher_path)
    output, device = configure_run(args, 'stage1')
    datasets, loaders, fingerprints = make_loaders(args, load_ecg=True)
    teacher, teacher_checkpoint = teacher_from_checkpoint(args.cmr_teacher_path, use_checkpoint=False)
    validate_checkpoint_context(teacher_checkpoint, fingerprints, args.phenotype_indices)
    norm = teacher_checkpoint['phenotype_norm']
    teacher_sha256 = _sha256(args.cmr_teacher_path)
    del teacher_checkpoint
    teacher.to(device).eval()
    mean, std = norm_tensors(norm, device)
    ecg_model = build_ecg_encoder(args.ecg_config_path).to(device)
    embed_dim = teacher.encoder_config['embed_dim']
    if getattr(ecg_model, 'width', getattr(ecg_model, 'embed_dim', None)) != embed_dim:
        raise ValueError('ECG and teacher embedding dimensions must agree')
    pheno_head = build_pheno_head(embed_dim, args.pheno_hidden_dim).to(device)
    trainable = torch.nn.ModuleDict({'ecg': ecg_model, 'pheno_head': pheno_head})
    optimizer = make_optimizer(trainable, args.lr, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    modules = {'model': ecg_model, 'pheno_head': pheno_head}
    start_epoch, best_metric, patience = 0, float('inf'), 0
    if args.resume:
        checkpoint = resume_run(args.resume, args, modules, optimizer, scaler, fingerprints, loaders)
        if checkpoint.get('teacher_checkpoint_sha256') != teacher_sha256:
            raise ValueError('Resume teacher checkpoint contents changed')
        if checkpoint['phenotype_norm'] != norm:
            raise ValueError('Resume teacher phenotype normalization changed')
        start_epoch, best_metric, patience = checkpoint['epoch'] + 1, checkpoint['best_metric'], checkpoint['patience']
        del checkpoint
    write_json(output / 'run_config.json', vars(args))
    write_json(output / 'phenotype_norm.json', norm)
    write_json(output / 'data_fingerprint.json', fingerprints)
    with open(args.ecg_config_path, encoding='utf-8') as handle:
        downstream_config = yaml.safe_load(handle)
    downstream_config.update(mode='align', afterAlign_path=str(output / 'best-loss.pth'))
    downstream_config['model']['num_classes'] = 1
    downstream_config['model']['pred_metric'] = False
    (output / 'st_mem_aligned_ecg_sem_stage1.yaml').write_text(yaml.safe_dump(downstream_config, sort_keys=False), encoding='utf-8')
    writer = make_writer(output)
    best_path = output / 'best-loss.pth'
    config = vars(args)
    print(json.dumps({'event': 'configuration', 'learning_rate': args.lr,
                      'effective_batch_size': args.batch_size * args.accum_iter,
                      'teacher_frozen': True, 'start_epoch': start_epoch}), flush=True)
    try:
        for epoch in range(start_epoch, args.epochs):
            print(json.dumps({'event': 'epoch_start', 'epoch': epoch}), flush=True)
            train_stats = engine.train_one_epoch(ecg_model, pheno_head, teacher, loaders['train'], optimizer,
                                                 device, epoch, scaler, mean, std, config)
            val_stats, details = engine.evaluate(ecg_model, pheno_head, teacher, loaders['val'], device,
                                                epoch, mean, std, config, split='val')
            metric = val_stats['loss']
            is_best = metric < best_metric
            best_metric, patience = (metric, 0) if is_best else (best_metric, patience + 1)
            payload = checkpoint_payload(args, epoch, modules, optimizer, scaler, norm, fingerprints,
                                         loaders, best_metric, patience, val_stats,
                                         teacher_checkpoint_sha256=teacher_sha256)
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
        test_stats, details = engine.evaluate(ecg_model, pheno_head, teacher, loaders['test'], device,
                                             best_epoch, mean, std, config, split='test')
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
