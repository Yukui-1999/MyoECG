"""Audited Henan loaders and stage2 checkpoint/early-stop handling."""
import csv
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import _seed_worker, resume_run
from .stage2_initialization import validate_stage2_provenance


def make_henan_loaders(args):
    from .data.henan_dataset import HenanSACMRECGDataset, collate_henan_pairs
    from .data.prepare_henan_splits import validate_prepared_henan_splits

    root = Path(args.paired_split_dir)
    if not (root / 'audit.json').is_file():
        raise FileNotFoundError('Henan paired data is not prepared: %s. Run '
                                'bash SaCMR_ECG_Align/scripts/prepare_henan_data.sh first.' % root)
    validate_prepared_henan_splits(root)
    datasets, loaders, fingerprints = {}, {}, {}
    seen_patients, seen_ecgs = set(), set()
    for offset, split in enumerate(('train', 'val', 'test')):
        path = root / (split + '.csv')
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        patients = {row['patient_id'] for row in rows}
        ecgs = {row['ecg_path'] for row in rows}
        # Repeated original rows inside one split are intentionally retained.
        if seen_patients.intersection(patients) or seen_ecgs.intersection(ecgs):
            raise ValueError('Patient or ECG overlap between Henan splits')
        seen_patients.update(patients)
        seen_ecgs.update(ecgs)
        fingerprints[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset = HenanSACMRECGDataset(path, training=split == 'train', max_samples=args.max_samples)
        if not len(dataset):
            raise ValueError('No readable paired candidates in Henan %s split' % split)
        kwargs = dict(batch_size=args.batch_size if split == 'train' else args.test_batch_size,
                      shuffle=split == 'train', num_workers=args.num_workers,
                      pin_memory=not args.no_pin_memory and args.device.startswith('cuda'),
                      drop_last=False, worker_init_fn=_seed_worker,
                      generator=torch.Generator().manual_seed(args.seed + offset),
                      collate_fn=collate_henan_pairs)
        if args.num_workers:
            kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
        datasets[split] = dataset
        loaders[split] = DataLoader(dataset, **kwargs)
    fingerprints['max_samples'] = args.max_samples
    payload_gib = args.batch_size * 10 * 50 * 96 * 96 * 4 / 1024**3
    print(json.dumps({'event': 'dataset_ready', 'dataset': 'Henan',
                      'samples': {name: len(value) for name, value in datasets.items()},
                      'batch_sa_fp32_gib': payload_gib,
                      'worker_prefetch_payload_gib': payload_gib * args.num_workers * args.prefetch_factor,
                      'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor}), flush=True)
    return datasets, loaders, fingerprints


def update_early_stopping(metric, best_metric, patience, best_patience):
    """Count every nonimproving validation epoch; no frozen-stage delay."""
    if not math.isfinite(metric):
        raise ValueError('Validation loss must be finite')
    if best_patience < 0 or patience < 0:
        raise ValueError('Patience must be nonnegative')
    improved = metric < best_metric
    best_metric, patience = (metric, 0) if improved else (best_metric, patience + 1)
    stop = best_patience > 0 and patience >= best_patience
    return best_metric, patience, improved, stop


def resume_henan_run(args, modules, optimizer, scaler, fingerprints, loaders, provenance):
    # Reject a different teacher or stage1 source before applying any saved state.
    checkpoint = torch.load(args.resume, map_location='cpu')
    validate_stage2_provenance(checkpoint, provenance)
    # A byte-identical teacher may have moved. Provenance has already checked
    # its contents; preserve all other generic resume configuration checks.
    restore_args = copy.copy(args)
    restore_args.cmr_teacher_path = checkpoint['config'].get('cmr_teacher_path')
    del checkpoint
    return resume_run(args.resume, restore_args, modules, optimizer, scaler, fingerprints, loaders)
