"""Intersect original UKB splits with the validated SA acquisition inventory."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from SaCMR_encoder.data.dataset import ShortAxisNPZDataset
from util.losses_pheno import PHENOTYPE_COLS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')
FORMAT_VERSION = 1
PAIRED_COLUMNS = ['split', 'source_row', 'eid', 'instance', 'array_index', 'ecg_path', 'sa_path'] + PHENOTYPE_COLS
EXCLUDED_COLUMNS = ['split', 'source_row', 'eid', 'reason', 'detail']
SA_PATTERN = re.compile(r'(\d+)_20209_(\d+)_(\d+)_sa_processed\.npz')
ECG_PATTERN = re.compile(r'(\d+)__20205_(\d+)_(\d+)\.pkl')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source(path):
    return {'path': str(path), 'sha256': _sha256(path)}


def _read_csv(path, required):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('{} is empty or missing columns: {}'.format(path, sorted(missing)))
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed CSV row in {}'.format(path))
    return rows


def _file_problem(path):
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return 'missing'
    # Permission and I/O errors must abort rather than silently change a cohort.
    if not stat.S_ISREG(metadata.st_mode):
        return 'not_regular_file'
    if metadata.st_size == 0:
        return 'empty'
    return None


def _ecg_path(value, split_dir, ecg_root):
    if ecg_root is not None:
        return ecg_root / Path(value).name
    # CSV entries are relative to the split directory unless --ecg_root is set.
    path = Path(value).expanduser()
    return path if path.is_absolute() else split_dir / path


def _write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def validate_prepared_splits(output_dir, verify_sources=True):
    """Verify the preparation audit and cohort identity before training.

    Hashes cover all three paired CSVs and the exclusion report. By default the
    original split CSVs and SA inventory must still match their recorded hashes.
    This reads only small manifests and tables, never the image/ECG inventory.
    ``verify_sources=False`` is an explicit offline check of prepared artifacts.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    audit_path = output_dir / 'audit.json'
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Missing or unreadable preparation audit {}; run prepare_splits first.'.format(audit_path)) from error

    def require(condition):
        if not condition:
            raise ValueError('Invalid preparation audit schema in {}; prepare a new output directory.'.format(audit_path))

    require(isinstance(audit, dict))
    require(audit.get('format_version') == FORMAT_VERSION)
    require(audit.get('phenotype_names') == list(PHENOTYPE_COLS))
    require(audit.get('checked_exists_nonempty') is True)
    require(type(audit.get('npz_full_validation')) is bool)
    require(audit.get('subject_overlap') == 0)
    config = audit.get('config')
    require(isinstance(config, dict))
    require(config.get('format_version') == FORMAT_VERSION)
    require(config.get('npz_full_validation') == audit['npz_full_validation'])
    require(isinstance(config.get('sa_root'), str))
    sources, hashes, summaries = audit.get('sources'), audit.get('outputs_sha256'), audit.get('splits')
    require(isinstance(sources, dict) and set(sources) == set(SPLITS) | {'sa_manifest'})
    require(isinstance(hashes, dict) and set(hashes) == {name + '.csv' for name in SPLITS} | {'excluded.csv'})
    require(isinstance(summaries, dict) and set(summaries) == set(SPLITS))
    for name, summary in summaries.items():
        require(isinstance(summary, dict))
        require(all(type(summary.get(key)) is int and summary[key] >= 0
                    for key in ('source_rows', 'kept', 'excluded')))
        require(summary['source_rows'] == summary['kept'] + summary['excluded'])
        reasons = summary.get('excluded_reasons')
        require(isinstance(reasons, dict) and all(type(count) is int and count >= 0 for count in reasons.values()))
        require(sum(reasons.values()) == summary['excluded'])
    for name, entry in sources.items():
        require(isinstance(entry, dict) and isinstance(entry.get('path'), str)
                and isinstance(entry.get('sha256'), str))
        require(Path(entry['path']).is_absolute() and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is not None)
        if verify_sources:
            try:
                current = _sha256(entry['path'])
            except OSError as error:
                raise ValueError('Cannot read recorded source {} from {}; restore it or prepare a new cohort.'.format(name, audit_path)) from error
            if current != entry['sha256']:
                raise ValueError('Recorded source {} changed since preparation; use the original source or prepare a new cohort.'.format(name))
    for name, expected in hashes.items():
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected) is not None)
        try:
            current = _sha256(output_dir / name)
        except OSError as error:
            raise ValueError('Prepared split integrity check failed: cannot read {}'.format(output_dir / name)) from error
        if current != expected:
            raise ValueError('Prepared split hash changed for {}; restore audited output or prepare a new cohort.'.format(output_dir / name))
    return audit


def prepare_splits(split_dir=None, sa_root=None, sa_manifest=None, output_dir=None,
                   validate_npz=False, ecg_root=None):
    """Publish fixed train/val/test intersections and return their audit.

    Existing output is reused only if sources, configuration and generated file
    hashes match. Use a new output directory when any inputs change. Publication
    uses a sibling staging directory so an interrupted run exposes no partial
    prepared cohort. Subject overlap/duplicates are rejected before filtering.
    """
    split_dir = Path(split_dir or PROJECT_ROOT / 'data/UKB/split').expanduser().resolve()
    sa_root = Path(sa_root or PROJECT_ROOT / 'data/processed_sa_x112_y10').expanduser().resolve()
    sa_manifest = Path(sa_manifest or PROJECT_ROOT / 'SaCMR_encoder/manifests/train_manifest_nvme.txt').expanduser().resolve()
    output_dir = Path(output_dir or PROJECT_ROOT / 'SaCMR_ECG_Align/data/paired_split').expanduser().resolve()
    ecg_root = Path(ecg_root).expanduser().resolve() if ecg_root is not None else None
    source_paths = {name: split_dir / (name + '_data.csv') for name in SPLITS}
    sources = {name: _source(path) for name, path in source_paths.items()}
    sources['sa_manifest'] = _source(sa_manifest)
    config = {'sa_root': str(sa_root), 'ecg_root': str(ecg_root) if ecg_root else None,
              'npz_full_validation': bool(validate_npz), 'format_version': FORMAT_VERSION}
    if output_dir.exists() and any(output_dir.iterdir()):
        try:
            # Source hashes were just read above, so avoid hashing them twice.
            audit = validate_prepared_splits(output_dir, verify_sources=False)
            valid = audit.get('sources') == sources and audit.get('config') == config
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError('Prepared output {} is stale, incomplete, or from different inputs; use a new --output_dir.'.format(output_dir))
        return audit

    rows_by_split = {name: _read_csv(path, ['Eid', 'ecg_list'] + PHENOTYPE_COLS)
                     for name, path in source_paths.items()}
    if not any(rows_by_split.values()):
        raise ValueError('All source splits are empty: {}'.format(split_dir))
    observed = set()
    for name, rows in rows_by_split.items():
        ids = [row['Eid'].strip() for row in rows]
        if any(not re.fullmatch(r'\d+', eid) for eid in ids):
            raise ValueError('Source split {} contains invalid Eid values'.format(name))
        if len(ids) != len(set(ids)):
            raise ValueError('Source split {} contains duplicate subjects'.format(name))
        overlap = observed.intersection(ids)
        if overlap:
            raise ValueError('Subject overlap between source splits: {} subjects in {}'.format(len(overlap), name))
        observed.update(ids)

    inventory = {}
    with sa_manifest.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            path = Path(line.strip())
            match = SA_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError('Invalid SA manifest filename on line {}'.format(number))
            key = match.groups()
            if key in inventory:
                raise ValueError('Duplicate SA acquisition in manifest on line {}'.format(number))
            inventory[key] = sa_root / key[0] / path.name
    if not inventory:
        raise ValueError('SA manifest is empty: {}'.format(sa_manifest))

    prepared, excluded, summaries = {}, [], {}
    for split, rows in rows_by_split.items():
        kept, reasons = [], Counter()
        for source_row, row in enumerate(rows):
            eid = row['Eid'].strip()
            match = ECG_PATTERN.fullmatch(Path(row['ecg_list']).name)
            reason, detail = None, ''
            if match is None:
                reason = 'invalid_ecg_filename'
            elif match[1] != eid:
                reason = 'ecg_eid_mismatch'
            elif match.groups() not in inventory:
                reason = 'no_matching_sa_acquisition'
            else:
                sa_path = inventory[match.groups()]
                ecg_path = _ecg_path(row['ecg_list'], split_dir, ecg_root)
                for kind, path in [('ecg', ecg_path), ('sa', sa_path)]:
                    problem = _file_problem(path)
                    if problem:
                        reason, detail = kind + '_' + problem, str(path)
                        break
                if reason is None:
                    try:
                        targets = [float(row[column]) for column in PHENOTYPE_COLS]
                    except ValueError:
                        reason = 'invalid_phenotypes'
                    else:
                        if not all(math.isfinite(value) for value in targets):
                            reason = 'invalid_phenotypes'
                if reason is None and validate_npz:
                    try:
                        ShortAxisNPZDataset.load_validated_arrays(sa_path)
                    except ValueError as error:
                        if isinstance(error.__cause__, OSError):
                            raise
                        reason, detail = 'invalid_sa_npz', str(error)
                if reason is None:
                    record = {'split': split, 'source_row': source_row, 'eid': eid,
                              'instance': match[2], 'array_index': match[3],
                              'ecg_path': str(ecg_path), 'sa_path': str(sa_path)}
                    record.update({column: row[column] for column in PHENOTYPE_COLS})
                    kept.append(record)
            if reason is not None:
                reasons[reason] += 1
                excluded.append({'split': split, 'source_row': source_row, 'eid': eid,
                                 'reason': reason, 'detail': detail})
        prepared[split] = kept
        summaries[split] = {'source_rows': len(rows), 'kept': len(kept),
                            'excluded': sum(reasons.values()), 'excluded_reasons': dict(reasons)}
    if not any(prepared.values()):
        raise ValueError('No valid paired acquisitions remain; check data roots and source manifest')
    audit = {'format_version': FORMAT_VERSION, 'sources': sources, 'config': config,
             'splits': summaries, 'sa_manifest_acquisitions': len(inventory),
             'matching_policy': 'exact (Eid, instance, array_index); original split and row order',
             'subject_overlap': 0, 'phenotype_names': list(PHENOTYPE_COLS),
             'checked_exists_nonempty': True, 'npz_full_validation': bool(validate_npz),
             'ecg_payloads_validated': False,
             'preprocessing': 'ukb_stmem_500to250_crop2250_hp0.67_lp40_global_standardize_v1'}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output_dir.name + '-', dir=str(output_dir.parent)))
    try:
        for split, rows in prepared.items():
            _write_csv(staging / (split + '.csv'), PAIRED_COLUMNS, rows)
        _write_csv(staging / 'excluded.csv', EXCLUDED_COLUMNS, excluded)
        audit['outputs_sha256'] = {path.name: _sha256(path) for path in sorted(staging.iterdir())}
        with (staging / 'audit.json').open('w', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(staging), str(output_dir))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split_dir', default=None)
    parser.add_argument('--sa_root', default=None)
    parser.add_argument('--sa_manifest', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--ecg_root', default=None, help='Override ECG directory using each source basename')
    parser.add_argument('--validate_npz', action='store_true', help='Fully decode/validate matched SA arrays')
    args = parser.parse_args()
    audit = prepare_splits(**vars(args))
    print(json.dumps({'splits': audit['splits'], 'npz_full_validation': audit['npz_full_validation']}, indent=2))


if __name__ == '__main__':
    main()
