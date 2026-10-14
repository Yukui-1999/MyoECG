"""Henan SA/ECG pairs with the original 1000 Hz ST-MEM preprocessing."""

import csv
import lzma
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample, sosfiltfilt
from torch.utils.data import Dataset, default_collate

from SaCMR_encoder.data.dataset import ShortAxisNPZDataset


DATA_ERRORS = (OSError, ValueError, TypeError, EOFError, KeyError, RuntimeError,
               pickle.UnpicklingError, AttributeError, ImportError, IndexError,
               OverflowError, lzma.LZMAError)
_HIGHPASS = butter(5, 0.67, btype='highpass', fs=250, output='sos')
_LOWPASS = butter(5, 40, btype='lowpass', fs=250, output='sos')


def validate_henan_ecg(array):
    """Validate the original ten-second, twelve-lead 1000 Hz ECG array."""
    array = np.asarray(array)
    if array.shape != (12, 10000):
        raise ValueError('Henan ECG must have shape [12,10000] at 1000 Hz; got {}'.format(array.shape))
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError('Henan ECG must contain real numeric values')
    if not np.isfinite(array).all():
        raise ValueError('Henan ECG contains nonfinite values')
    return array


def load_henan_ecg(path):
    # These are trusted project-generated pickles, never external uploads.
    with Path(path).open('rb') as handle:
        return validate_henan_ecg(pickle.load(handle))


def preprocess_henan_ecg(array, training=False):
    """Match HeNan_ECGCMRBase: 1000->250 Hz, crops, filters, standardization."""
    array = validate_henan_ecg(array)
    signal = resample(array, 2500, axis=1)
    if training:
        start = np.random.randint(0, 251)
        signal = signal[:, start:start + 2250]
    else:
        signal = np.stack([signal[:, start:start + 2250] for start in (0, 125, 250)])
    signal = sosfiltfilt(_HIGHPASS, signal)
    signal = sosfiltfilt(_LOWPASS, signal)
    mean = signal.mean(axis=(-1, -2), keepdims=True)
    std = signal.std(axis=(-1, -2), keepdims=True)
    signal = np.divide(signal.astype(np.float32) - mean, std,
                       out=np.zeros(signal.shape, dtype=np.float32), where=std != 0)
    if not np.isfinite(signal).all():
        raise ValueError('Henan ECG preprocessing produced nonfinite values')
    return torch.from_numpy(signal)


def validate_sa_numeric(image):
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError('SA image must contain real numeric values')


class HenanSACMRECGDataset(Dataset):
    """Load audited pairs; report unreadable samples for the collate function.

    Preparation fully validates the cohort. Files that become unreadable later
    produce explicit invalid markers and warnings so the engine can count skips.
    SA image values retain their stored percentile normalization.
    """

    def __init__(self, csv_path, training=False, size=96, max_samples=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if type(size) is not int or size <= 0:
            raise ValueError('size must be a positive integer')
        if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
            raise ValueError('max_samples must be a positive integer or None')
        with self.csv_path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            required = {'patient_id', 'source_row', 'sa_path', 'ecg_path'}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError('{} is missing columns: {}'.format(self.csv_path, sorted(missing)))
            self.rows = list(reader)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError('No paired samples in {}'.format(self.csv_path))
        for row in self.rows:
            if any(not isinstance(row.get(key), str) or not row[key] for key in required):
                raise ValueError('Malformed prepared row in {}'.format(self.csv_path))
            row['source_row'] = int(row['source_row'])
            if row['source_row'] < 0:
                raise ValueError('source_row must be nonnegative')
        self.training = bool(training)
        self.size = size

    def __len__(self):
        return len(self.rows)

    def _path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.csv_path.parent / path

    def _invalid(self, path, error):
        reason = '{}: {}'.format(type(error).__name__, error)
        warnings.warn('Skipping unreadable Henan pair {}: {}'.format(path, reason),
                      RuntimeWarning, stacklevel=2)
        return {'_invalid_sample': True, 'path': str(path), 'reason': reason}

    def __getitem__(self, index):
        row = self.rows[index]
        sa_path = self._path(row['sa_path'])
        try:
            image, mask = ShortAxisNPZDataset.load_validated_arrays(sa_path)
            validate_sa_numeric(image)
            image = image.astype(np.float32, copy=True)
            image[:, :, ~mask, :] = 0.0
            if not np.isfinite(image).all():
                raise ValueError('SA image exceeds finite float32 range')
        except DATA_ERRORS as error:
            return self._invalid(sa_path, error)
        ecg_path = self._path(row['ecg_path'])
        try:
            ecg = preprocess_henan_ecg(load_henan_ecg(ecg_path), self.training)
        except DATA_ERRORS as error:
            return self._invalid(ecg_path, error)
        image = torch.from_numpy(image).permute(2, 3, 0, 1).contiguous()
        if self.size != 112:
            image = F.interpolate(image, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return {'sa_cmr': image, 'slice_valid_mask': torch.from_numpy(mask.copy()),
                'ecg': ecg, 'patient_id': row['patient_id'], 'source_row': row['source_row']}


def collate_henan_pairs(samples):
    """Drop explicitly invalid records while retaining a visible skip count."""
    valid = [sample for sample in samples if not sample.get('_invalid_sample', False)]
    batch = default_collate(valid) if valid else {}
    batch['_skipped_samples'] = len(samples) - len(valid)
    return batch
