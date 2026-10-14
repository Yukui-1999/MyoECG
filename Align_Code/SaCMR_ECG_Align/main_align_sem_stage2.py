"""Henan semantic stage2: align ECG to a frozen short-axis CMR teacher."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from . import engine_align_sem_stage2 as engine
from .common import (
    PACKAGE_DIR, add_common_args, atomic_torch_save, checkpoint_payload,
    configure_run, make_optimizer, make_writer, record_epoch, write_json,
)
from .henan_common import make_henan_loaders, resume_henan_run, update_early_stopping
from .stage2_initialization import load_stage2_models
from util.align_common import append_metrics_csv


def get_args_parser():
    parser = add_common_args(argparse.ArgumentParser('Henan short-axis CMR–ECG semantic stage2'), 'stage2')
    parser.set_defaults(paired_split_dir=str(PACKAGE_DIR / 'data/paired_henan_split'),
                        exp_name='ecg_sem_stage2_henan', batch_size=64, test_batch_size=100,
                        epochs=50, warmup_epochs=5, blr=1e-4, best_patience=5)
    parser.add_argument('--ecg_stage1_path', default=str(PACKAGE_DIR / 'outputs/ecg_sem_stage1/best-loss.pth'))
    parser.add_argument('--cmr_teacher_path', default=None,
                        help='Default: the teacher recorded in the stage1 checkpoint; contents must match')
    parser.add_argument('--ecg_config_path', default=None,
                        help='ECG architecture YAML; default comes from the stage1 checkpoint')
    parser.add_argument('--sem_loss_weight', type=float, default=1.0)
    parser.add_argument('--retrieval_chunk_size', type=int, default=256)
    return parser


def main(args):
    if not 0 < args.sem_loss_weight < float('inf') or args.retrieval_chunk_size < 1:
        raise ValueError('sem_loss_weight and retrieval_chunk_size must be positive')
    if args.phenotype_groups != 'all':
        raise ValueError('Henan stage2 has no phenotype loss; --phenotype_groups is not applicable')
    args.ecg_stage1_path = str(Path(args.ecg_stage1_path).expanduser().resolve())
    if not Path(args.ecg_stage1_path).is_file():
        raise FileNotFoundError('Missing completed SA stage1 ECG checkpoint: %s' % args.ecg_stage1_path)
    output, device = configure_run(args, 'stage2')
    # No phenotype labels are used; this empty selection is just resume metadata.
    args.phenotype_indices = []
    args.phenotype_names = []
    args.phenotype_group_metadata = None
    datasets, loaders, fingerprints = make_henan_loaders(args)
    print(json.dumps({'event': 'loading_models', 'ecg_stage1_path': args.ecg_stage1_path}), flush=True)
    ecg_model, teacher, provenance = load_stage2_models(
        args.ecg_stage1_path, teacher_path=args.cmr_teacher_path,
        ecg_config_path=args.ecg_config_path)
    args.cmr_teacher_path = provenance['teacher']['path']
    args.ecg_config_path = provenance['ecg_config']['path']
    ecg_model.to(device)
    teacher.to(device).eval()
    optimizer = make_optimizer(ecg_model, args.lr, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    modules = {'model': ecg_model}
    start_epoch, best_metric, patience = 0, float('inf'), 0
    if args.resume:
        checkpoint = resume_henan_run(args, modules, optimizer, scaler, fingerprints, loaders, provenance)
        start_epoch = checkpoint['epoch'] + 1
        best_metric, patience = checkpoint['best_metric'], checkpoint['patience']
        del checkpoint
    write_json(output / 'run_config.json', vars(args))
    write_json(output / 'data_fingerprint.json', fingerprints)
    write_json(output / 'source_provenance.json', provenance)
    downstream_config = dict(provenance['ecg_config']['config'])
    downstream_config['model'] = dict(downstream_config['model'])
    downstream_config.update(mode='align', afterAlign_path=str(output / 'best-loss.pth'))
    downstream_config['model'].update(num_classes=1, pred_metric=False)
    (output / 'st_mem_aligned_ecg_sem_stage2.yaml').write_text(
        yaml.safe_dump(downstream_config, sort_keys=False), encoding='utf-8')
    writer = make_writer(output)
    config = vars(args)
    best_path = output / 'best-loss.pth'
    print(json.dumps({'event': 'configuration', 'stage': 'stage2', 'learning_rate': args.lr,
                      'effective_batch_size': args.batch_size * args.accum_iter,
                      'teacher_frozen': True, 'start_epoch': start_epoch,
                      'best_patience': args.best_patience, 'patience': patience,
                      'stage1_epoch': provenance['stage1']['epoch'],
                      'teacher_epoch': provenance['teacher']['epoch']}), flush=True)
    try:
        # A resumed run that already reached early stopping only evaluates best.
        already_stopped = args.best_patience > 0 and patience >= args.best_patience
        for epoch in range(start_epoch, args.epochs):
            if already_stopped:
                break
            print(json.dumps({'event': 'epoch_start', 'epoch': epoch,
                              'patience': patience, 'best_patience': args.best_patience}), flush=True)
            train_stats = engine.train_one_epoch(ecg_model, teacher, loaders['train'], optimizer,
                                                 device, epoch, scaler, config)
            val_stats, _ = engine.evaluate(ecg_model, teacher, loaders['val'], device,
                                           epoch, config, split='val')
            best_metric, patience, improved, stop = update_early_stopping(
                val_stats['loss'], best_metric, patience, args.best_patience)
            val_stats.update(best_loss=best_metric, patience=patience, best_patience=args.best_patience)
            payload = checkpoint_payload(args, epoch, modules, optimizer, scaler, None, fingerprints,
                                         loaders, best_metric, patience, val_stats,
                                         source_provenance=provenance)
            if improved:
                atomic_torch_save(payload, best_path)
            atomic_torch_save(payload, output / 'checkpoint-latest.pth')
            del payload
            append_metrics_csv(str(output / 'sem_metrics.csv'), 'val', epoch, 'current', val_stats)
            record_epoch(output, epoch, train_stats, val_stats, writer)
            if stop:
                print(json.dumps({'event': 'early_stop', 'epoch': epoch, 'patience': patience,
                                  'best_patience': args.best_patience, 'best_loss': best_metric}), flush=True)
                break
        if not best_path.is_file():
            raise FileNotFoundError('Selected checkpoint is missing: %s' % best_path)
        checkpoint = torch.load(best_path, map_location='cpu')
        ecg_model.load_state_dict(checkpoint['model'], strict=True)
        best_epoch = checkpoint['epoch']
        del checkpoint
        test_stats, _ = engine.evaluate(ecg_model, teacher, loaders['test'], device,
                                        best_epoch, config, split='test')
        write_json(output / 'test_metrics.json', {'checkpoint': str(best_path), 'epoch': best_epoch, **test_stats})
        append_metrics_csv(str(output / 'sem_metrics.csv'), 'test', best_epoch, best_path.name, test_stats)
        print(json.dumps({'event': 'complete', 'best_checkpoint': str(best_path), 'best_epoch': best_epoch,
                          'test': test_stats}), flush=True)
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
