"""Validate Henan SA/ECG pairs while preserving the original XLSX split rows."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from SaCMR_encoder.data.dataset import ShortAxisNPZDataset
from SaCMR_ECG_Align.data.henan_dataset import DATA_ERRORS, load_henan_ecg, validate_sa_numeric


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / 'data/HenanData/seed_3407'
DEFAULT_SA_ROOT = PROJECT_ROOT / 'data/Henan_SA_processed_x112_y10_cinema'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'SaCMR_ECG_Align/data/paired_henan_split'
SPLIT_FILES = {'train': 'train.xlsx', 'val': 'valid.xlsx', 'test': 'test.xlsx'}
FORMAT_VERSION = 1
PAIRED_COLUMNS = ['split', 'source_row', 'patient_id', 'ecg_path', 'sa_path']
EXCLUDED_COLUMNS = PAIRED_COLUMNS + ['reason', 'detail']
PREPROCESSING = 'henan_stmem_1000to250_crop2250_hp0.67_lp40_global_standardize_v1'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source(path):
    return {'path': str(path), 'sha256': _sha256(path)}


def _write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _file_problem(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return 'missing'
    except OSError:
        return 'unreadable'
    if not stat.S_ISREG(info.st_mode):
        return 'not_regular_file'
    if info.st_size == 0:
        return 'empty'
    return None


def _lexical_ecg_path(value, split_dir, ecg_root):
    path = Path(value).expanduser()
    if ecg_root is not None:
        path = ecg_root / path.name
    elif not path.is_absolute():
        path = split_dir / path
    return Path(os.path.abspath(str(path)))


def _ecg_path(value, split_dir, ecg_root):
    return _lexical_ecg_path(value, split_dir, ecg_root).resolve()


def _validate_sa(path):
    problem = _file_problem(path)
    if problem:
        return 'sa_' + problem, str(path)
    try:
        image, mask = ShortAxisNPZDataset.load_validated_arrays(path)
        validate_sa_numeric(image)
        # Finite float16/float32 values already fit float32. Avoid an unnecessary
        # slow float16 reduction over every large volume.
        if image.dtype.itemsize > 4 and np.max(np.abs(image[:, :, mask, :])) > np.finfo(np.float32).max:
            raise ValueError('SA image exceeds finite float32 range')
    except DATA_ERRORS as error:
        return 'invalid_sa_npz', '{}: {}'.format(type(error).__name__, error)
    return None, ''


def _validate_ecg(path):
    problem = _file_problem(path)
    if problem:
        return 'ecg_' + problem, str(path)
    try:
        load_henan_ecg(path)
    except DATA_ERRORS as error:
        return 'invalid_ecg', '{}: {}'.format(type(error).__name__, error)
    return None, ''


def _validate_inventory(paths, validator, workers, phase):
    """Keep at most workers tasks in flight and retain only validation results."""
    results, pending = {}, {}
    entries = iter(paths.items())
    started = last_report = time.monotonic()
    last_count = 0
    print(json.dumps({'event': 'henan_validation_start', 'phase': phase,
                      'total_files': len(paths), 'validation_workers': workers}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def submit_next():
            try:
                key, path = next(entries)
            except StopIteration:
                return
            pending[executor.submit(validator, path)] = key

        for _ in range(workers):
            submit_next()
        while pending:
            completed, _ = wait(pending, timeout=max(0.01, 10 - (time.monotonic() - last_report)),
                                return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                results[key] = future.result()
                submit_next()
            now = time.monotonic()
            if len(results) - last_count >= 50 or now - last_report >= 10 or not pending:
                print(json.dumps({'event': 'henan_validation_progress', 'phase': phase,
                                  'checked_files': len(results), 'total_files': len(paths),
                                  'invalid_files': sum(reason is not None for reason, _ in results.values()),
                                  'elapsed_seconds': round(now - started, 1)}), flush=True)
                last_report, last_count = now, len(results)
    return results


def validate_prepared_henan_splits(output_dir, verify_sources=True):
    """Check small source and prepared artifacts without rescanning NPZ/ECG data."""
    output_dir = Path(output_dir).expanduser().resolve()
    path = output_dir / 'audit.json'
    try:
        audit = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Missing or unreadable Henan preparation audit {}'.format(path)) from error

    def require(condition):
        if not condition:
            raise ValueError('Invalid Henan preparation audit schema in {}'.format(path))

    require(isinstance(audit, dict))
    require(audit.get('format_version') == FORMAT_VERSION)
    require(audit.get('dataset') == 'henan_sa_ecg')
    require(audit.get('subject_overlap') == 0 and audit.get('ecg_overlap') == 0)
    require(audit.get('npz_full_validation') is True and audit.get('ecg_payloads_validated') is True)
    require(audit.get('checked_exists_nonempty') is True)
    require(audit.get('preprocessing') == PREPROCESSING)
    config = audit.get('config')
    require(isinstance(config, dict) and config.get('format_version') == FORMAT_VERSION)
    require(config.get('npz_full_validation') is True and config.get('ecg_payloads_validated') is True)
    require(isinstance(config.get('sa_root'), str))
    require(config.get('ecg_root') is None or isinstance(config.get('ecg_root'), str))
    sources, hashes, summaries = audit.get('sources'), audit.get('outputs_sha256'), audit.get('splits')
    require(isinstance(sources, dict) and set(sources) == set(SPLIT_FILES))
    require(isinstance(hashes, dict) and set(hashes) == {s + '.csv' for s in SPLIT_FILES} | {'excluded.csv'})
    require(isinstance(summaries, dict) and set(summaries) == set(SPLIT_FILES))
    for summary in summaries.values():
        require(isinstance(summary, dict))
        require(all(type(summary.get(key)) is int and summary[key] >= 0
                    for key in ('source_rows', 'kept', 'excluded', 'source_unique_patients',
                                'kept_unique_patients', 'duplicate_patient_ecg_rows')))
        require(summary['source_rows'] == summary['kept'] + summary['excluded'])
        reasons = summary.get('excluded_reasons')
        require(isinstance(reasons, dict) and all(type(n) is int and n >= 0 for n in reasons.values()))
        require(sum(reasons.values()) == summary['excluded'])
    for name, source in sources.items():
        require(isinstance(source, dict) and isinstance(source.get('path'), str)
                and isinstance(source.get('sha256'), str))
        require(Path(source['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}', source['sha256']) is not None)
        if verify_sources:
            try:
                current = _sha256(source['path'])
            except OSError as error:
                raise ValueError('Cannot read recorded Henan source {}'.format(name)) from error
            if current != source['sha256']:
                raise ValueError('Recorded Henan source {} changed; prepare a new output directory'.format(name))
    for name, expected in hashes.items():
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None)
        try:
            current = _sha256(output_dir / name)
        except OSError as error:
            raise ValueError('Prepared Henan integrity check cannot read {}'.format(name)) from error
        if current != expected:
            raise ValueError('Prepared Henan hash changed for {}'.format(name))
    return audit


def prepare_henan_splits(split_dir=None, sa_root=None, output_dir=None, ecg_root=None,
                         validation_workers=4):
    """Decode each unique matched SA and ECG once, then atomically publish rows.

    Only canonical PatientID/PatientID_sa_processed.npz paths are considered.
    Missing, corrupt and unreadable payloads are excluded with an audit reason.
    Original row order, split membership and within-split repeats are retained.
    """
    if type(validation_workers) is not int or not 1 <= validation_workers <= 4:
        raise ValueError('validation_workers must be an integer from 1 to 4')
    split_dir = Path(split_dir or DEFAULT_SPLIT_DIR).expanduser().resolve()
    sa_root = Path(sa_root or DEFAULT_SA_ROOT).expanduser().resolve()
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    ecg_root = Path(ecg_root).expanduser().resolve() if ecg_root is not None else None
    source_paths = {split: split_dir / filename for split, filename in SPLIT_FILES.items()}
    sources = {split: _source(path) for split, path in source_paths.items()}
    config = {'format_version': FORMAT_VERSION, 'sa_root': str(sa_root),
              'ecg_root': str(ecg_root) if ecg_root else None,
              'npz_full_validation': True, 'ecg_payloads_validated': True}
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        try:
            audit = validate_prepared_henan_splits(output_dir, verify_sources=False)
            valid = audit['sources'] == sources and audit['config'] == config
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError('Prepared Henan output is stale, incomplete, or from different inputs; use a new --output_dir')
        print(json.dumps({'event': 'henan_preparation_reused', 'splits': audit['splits']}), flush=True)
        return audit

    rows_by_split, summaries = {}, {}
    observed_ids, observed_ecgs = set(), set()
    for split, path in source_paths.items():
        try:
            frame = pd.read_excel(path, usecols=['PatientID', 'ECG_data_path'], dtype=str,
                                  keep_default_na=False)
        except (OSError, ValueError) as error:
            raise ValueError('Cannot read required PatientID/ECG_data_path columns from {}'.format(path)) from error
        rows = frame.to_dict('records')
        for row in rows:
            row['PatientID'] = row['PatientID'].strip()
            row['ECG_data_path'] = row['ECG_data_path'].strip()
            if re.fullmatch(r'P\d{8}', row['PatientID']) is None:
                raise ValueError('Invalid PatientID in source split {}; expected P and eight digits'.format(split))
            row['_ecg_path_error'] = None
            row['_ecg_path'] = None
            if row['ECG_data_path']:
                try:
                    row['_ecg_path'] = _ecg_path(row['ECG_data_path'], split_dir, ecg_root)
                except (OSError, RuntimeError) as error:
                    # Retain the lexical path for the exclusion audit and overlap
                    # check when following symlinks or inaccessible parents fails.
                    row['_ecg_path'] = _lexical_ecg_path(row['ECG_data_path'], split_dir, ecg_root)
                    row['_ecg_path_error'] = ('ecg_path_unreadable', '{}: {}'.format(type(error).__name__, error))
        ids = {row['PatientID'] for row in rows}
        ecgs = {str(row['_ecg_path']) for row in rows if row['_ecg_path'] is not None}
        if ids & observed_ids:
            raise ValueError('PatientID overlap across source splits: {} patients in {}'.format(len(ids & observed_ids), split))
        if ecgs & observed_ecgs:
            raise ValueError('ECG path overlap across source splits: {} paths in {}'.format(len(ecgs & observed_ecgs), split))
        observed_ids.update(ids)
        observed_ecgs.update(ecgs)
        pair_count = len({(row['PatientID'], str(row['_ecg_path'])) for row in rows})
        duplicates = len(rows) - pair_count
        if len(rows) != len(ids):
            warnings.warn('Henan {} has {} repeated PatientID rows and {} duplicate PatientID/ECG rows; preserving all source rows'.format(
                split, len(rows) - len(ids), duplicates), RuntimeWarning, stacklevel=2)
        summaries[split] = {'source_rows': len(rows), 'source_unique_patients': len(ids),
                            'duplicate_patient_ecg_rows': duplicates}
        rows_by_split[split] = rows
    if not any(rows_by_split.values()):
        raise ValueError('All Henan source splits are empty')

    prepared, excluded = {}, []
    started = time.monotonic()
    total_rows = sum(len(rows) for rows in rows_by_split.values())
    print(json.dumps({'event': 'henan_preparation_start', 'source_rows': total_rows,
                      'source_unique_patients': len(observed_ids), 'npz_full_validation': True}), flush=True)
    sa_paths = {row['PatientID']: sa_root / row['PatientID'] / (row['PatientID'] + '_sa_processed.npz')
                for rows in rows_by_split.values() for row in rows}
    sa_cache = _validate_inventory(sa_paths, _validate_sa, validation_workers, 'sa')
    ecg_paths = {str(row['_ecg_path']): row['_ecg_path']
                 for rows in rows_by_split.values() for row in rows
                 if row['_ecg_path'] is not None and row['_ecg_path_error'] is None
                 and sa_cache[row['PatientID']][0] is None}
    ecg_cache = _validate_inventory(ecg_paths, _validate_ecg, validation_workers, 'ecg')
    for split, rows in rows_by_split.items():
        kept, reasons = [], Counter()
        for source_row, row in enumerate(rows):
            patient_id = row['PatientID']
            sa_path = sa_root / patient_id / (patient_id + '_sa_processed.npz')
            ecg_path = row['_ecg_path']
            record = {'split': split, 'source_row': source_row, 'patient_id': patient_id,
                      'ecg_path': str(ecg_path) if ecg_path else '', 'sa_path': str(sa_path)}
            reason, detail = row['_ecg_path_error'] or sa_cache[patient_id]
            if reason is None:
                if ecg_path is None:
                    reason, detail = 'ecg_missing_path', 'Empty ECG_data_path in source row'
                else:
                    reason, detail = ecg_cache[str(ecg_path)]
            if reason is None:
                kept.append(record)
            else:
                reasons[reason] += 1
                excluded.append(dict(record, reason=reason, detail=detail))
        prepared[split] = kept
        summaries[split].update({'kept': len(kept), 'kept_unique_patients': len({r['patient_id'] for r in kept}),
                                  'excluded': sum(reasons.values()), 'excluded_reasons': dict(reasons)})
    # Prevent a source file change during lengthy payload validation from being
    # certified under the hash of an earlier source version.
    if sources != {split: _source(path) for split, path in source_paths.items()}:
        raise ValueError('Henan source splits changed during preparation; rerun with stable input files')
    audit = {'format_version': FORMAT_VERSION, 'dataset': 'henan_sa_ecg', 'sources': sources,
             'config': config, 'splits': summaries, 'subject_overlap': 0, 'ecg_overlap': 0,
             'matching_policy': 'exact PatientID canonical SA filename; original split, row order and duplicate rows',
             'npz_full_validation': True, 'ecg_payloads_validated': True, 'checked_exists_nonempty': True,
             'validated_unique_sa': sum(reason is None for reason, _ in sa_cache.values()),
             'checked_unique_sa': len(sa_cache), 'checked_unique_ecg': len(ecg_cache),
             'preprocessing': PREPROCESSING}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output_dir.name + '-', dir=str(output_dir.parent)))
    try:
        for split, rows in prepared.items():
            _write_csv(staging / (split + '.csv'), PAIRED_COLUMNS, rows)
        _write_csv(staging / 'excluded.csv', EXCLUDED_COLUMNS, excluded)
        audit['outputs_sha256'] = {p.name: _sha256(p) for p in sorted(staging.iterdir())}
        with (staging / 'audit.json').open('w', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(staging), str(output_dir))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({'event': 'henan_preparation_complete', 'splits': summaries,
                      'validated_unique_sa': audit['validated_unique_sa'],
                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split_dir', default=None)
    parser.add_argument('--sa_root', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--ecg_root', default=None, help='Override ECG directory using original source basenames')
    parser.add_argument('--validation_workers', default=4, type=int,
                        help='Simultaneous file validations, 1 to 4; does not change cohort identity')
    prepare_henan_splits(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
