"""Run configuration, bounded data loading, and resumable checkpoints."""
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.losses_pheno import PHENOTYPE_COLS
from util.sem_ablation import configure_phenotype_selection

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def add_common_args(parser, stage):
    parser.add_argument('--paired_split_dir', default=str(PACKAGE_DIR / 'data/paired_split'))
    parser.add_argument('--output_dir', default=str(PACKAGE_DIR / 'outputs'))
    parser.add_argument('--exp_name', default='cmr_sem_teacher' if stage == 'teacher' else 'ecg_sem_stage1')
    parser.add_argument('--batch_size', type=int, default=64 if stage == 'teacher' else 128)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=1)
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--no_pin_memory', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_checkpoint', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=None)
    parser.add_argument('--best_patience', type=int, default=10, help='0 disables early stopping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--phenotype_groups', default='all')
    parser.add_argument('--pheno_hidden_dim', type=int, default=256)
    parser.add_argument('--resume', default=None, help='checkpoint-latest.pth from this training stage')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit each split for a smoke run; use a separate exp_name')
    return parser


def configure_run(args, stage):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('These entry points train on one GPU; do not launch with torchrun')
    for key in ('batch_size', 'test_batch_size', 'prefetch_factor', 'cpu_threads',
                'epochs', 'accum_iter', 'log_interval', 'pheno_hidden_dim'):
        if getattr(args, key) < 1:
            raise ValueError('%s must be positive' % key)
    for key in ('num_workers', 'warmup_epochs', 'best_patience'):
        if getattr(args, key) < 0:
            raise ValueError('%s must be nonnegative' % key)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError('max_samples must be positive')
    args.lr = args.lr if args.lr is not None else args.blr * args.batch_size * args.accum_iter / 256.0
    for key in ('blr', 'lr', 'min_lr', 'weight_decay'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key in ('lr', 'blr') and value == 0):
            raise ValueError('Invalid %s' % key)
    if args.min_lr > args.lr:
        raise ValueError('min_lr must not exceed lr')
    if args.max_norm is not None and (not math.isfinite(args.max_norm) or args.max_norm <= 0):
        raise ValueError('max_norm must be positive and finite')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for CPU checks')
    args.use_amp = not args.no_amp and args.device.startswith('cuda')
    args.use_checkpoint = not args.no_checkpoint
    args.stage = stage
    configure_phenotype_selection(args)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir).expanduser().resolve() / args.exp_name
    if args.resume:
        resume = Path(args.resume).expanduser().resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        if output != resume.parent:
            raise ValueError('--output_dir/--exp_name must identify the original resume directory: %s' % resume.parent)
    elif any((output / name).exists() for name in ('checkpoint-latest.pth', 'best-cmr-pheno.pth', 'best-loss.pth')):
        raise FileExistsError('Run already has checkpoints: %s. Use --resume or a new --exp_name.' % output)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'event': 'startup', 'stage': stage, 'device': args.device,
                      'output_dir': str(output), 'use_amp': args.use_amp}), flush=True)
    return output, torch.device(args.device)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loaders(args, load_ecg):
    from SaCMR_ECG_Align.data.dataset import PairedSACMRECGDataset
    from SaCMR_ECG_Align.data.prepare_splits import validate_prepared_splits
    validate_prepared_splits(args.paired_split_dir)
    datasets, loaders, fingerprints = {}, {}, {}
    seen = set()
    # Split leakage was checked at preparation time; verify again for supplied CSVs.
    import pandas as pd
    for offset, split in enumerate(('train', 'val', 'test')):
        path = Path(args.paired_split_dir) / (split + '.csv')
        if not path.is_file():
            raise FileNotFoundError('%s missing. Run bash SaCMR_ECG_Align/scripts/prepare_data.sh first.' % path)
        frame = pd.read_csv(path, dtype={'eid': str})
        ids = set(frame['eid'].tolist())
        if len(ids) != len(frame) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping subject IDs in paired split %s' % split)
        seen.update(ids)
        fingerprints[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset = PairedSACMRECGDataset(path, training=split == 'train', load_ecg=load_ecg,
                                       max_samples=args.max_samples)
        if len(dataset) == 0:
            raise ValueError('Empty %s split' % split)
        generator = torch.Generator().manual_seed(args.seed + offset)
        kwargs = dict(batch_size=args.batch_size if split == 'train' else args.test_batch_size,
                      shuffle=split == 'train', num_workers=args.num_workers,
                      pin_memory=not args.no_pin_memory and args.device.startswith('cuda'),
                      drop_last=False, worker_init_fn=_seed_worker, generator=generator)
        if args.num_workers > 0:
            # Separate train/val/test loaders must not retain three prefetched queues.
            kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
        datasets[split] = dataset
        loaders[split] = DataLoader(dataset, **kwargs)
    fingerprints['max_samples'] = args.max_samples
    input_gib = args.batch_size * 10 * 50 * 96 * 96 * 4 / (1024 ** 3)
    print(json.dumps({'event': 'dataset_ready', 'samples': {k: len(v) for k, v in datasets.items()},
                      'batch_sa_fp32_gib': input_gib,
                      'worker_prefetch_payload_gib': input_gib * args.num_workers * args.prefetch_factor,
                      'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
                      'note': 'payload estimate excludes decode buffers, pinning and OS cache'}), flush=True)
    return datasets, loaders, fingerprints


def phenotype_normalization(train_phenotypes):
    values = np.asarray(train_phenotypes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(PHENOTYPE_COLS):
        raise ValueError('Expected 27 phenotype columns')
    if not np.isfinite(values).all():
        raise ValueError('Training phenotypes must be finite')
    mean, std = values.mean(axis=0), values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {'phenotype_names': list(PHENOTYPE_COLS), 'mean': mean.tolist(), 'std': std.tolist(),
            'source': 'paired training split only'}


def norm_tensors(norm, device):
    if norm.get('phenotype_names') != list(PHENOTYPE_COLS):
        raise ValueError('Checkpoint phenotype column order mismatch')
    mean = torch.tensor(norm['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(norm['std'], dtype=torch.float32, device=device)
    if mean.shape != (27,) or std.shape != (27,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid phenotype normalization in checkpoint')
    return mean, std


def validate_checkpoint_context(checkpoint, fingerprints, indices):
    if checkpoint.get('data_fingerprint') != fingerprints:
        raise ValueError('Checkpoint paired split fingerprint differs from this run; prepare/use the same cohort')
    if checkpoint.get('config', {}).get('phenotype_indices') != list(indices):
        raise ValueError('Checkpoint phenotype selection differs from this run')


def make_optimizer(model, lr, weight_decay):
    # Match util.optimizer in the long-axis flow, including its weight decay.
    # Include currently frozen weights: last encoder blocks unfreeze later.
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def atomic_torch_save(payload, path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_payload(args, epoch, modules, optimizer, scaler, norm, fingerprints,
                       loaders, best_metric, patience, metrics, **extra):
    payload = {key: module.state_dict() for key, module in modules.items()}
    payload.update(config=dict(vars(args)), epoch=epoch, optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(), phenotype_norm=norm, data_fingerprint=fingerprints,
                   best_metric=best_metric, patience=patience, metrics=metrics,
                   rng_state={'python': random.getstate(), 'numpy': np.random.get_state(),
                              'torch': torch.get_rng_state(),
                              'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   loader_rng={name: loader.generator.get_state() for name, loader in loaders.items()})
    payload.update(extra)
    return payload


def resume_run(path, args, modules, optimizer, scaler, fingerprints, loaders):
    checkpoint = torch.load(path, map_location='cpu')
    validate_checkpoint_context(checkpoint, fingerprints, args.phenotype_indices)
    # Keep schedule and objective identical across epoch-boundary resumes.
    stable = ('stage', 'batch_size', 'test_batch_size', 'accum_iter', 'lr', 'min_lr',
              'warmup_epochs', 'epochs', 'weight_decay', 'max_norm', 'seed', 'use_amp',
              'freeze_encoder_epochs', 'unfreeze_last_blocks', 'sem_loss_weight',
              'pheno_loss_weight', 'rel_ecg_weight', 'tau_phi', 'tau_ecg',
              'pheno_hidden_dim', 'sem_hidden_dim', 'best_patience', 'cmr_teacher_path')
    for key in stable:
        if hasattr(args, key) and checkpoint['config'].get(key) != getattr(args, key):
            raise ValueError('Resume configuration changed: %s' % key)
    for key, module in modules.items():
        module.load_state_dict(checkpoint[key], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    rng = checkpoint['rng_state']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if torch.cuda.is_available() and rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])
    for name, loader in loaders.items():
        loader.generator.set_state(checkpoint['loader_rng'][name])
    return checkpoint


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def record_epoch(output, epoch, train_stats, val_stats, writer=None):
    record = {'epoch': epoch, 'train': train_stats, 'val': val_stats}
    with (output / 'log.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    if writer is not None:
        for split, stats in (('train', train_stats), ('val', val_stats)):
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('%s/%s' % (split, name), value, epoch)
        writer.flush()
    print(json.dumps({'event': 'epoch_end', **record}), flush=True)


def make_writer(output):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print('TensorBoard unavailable; metrics are still saved to log.jsonl', flush=True)
        return None
    return SummaryWriter(str(output))
