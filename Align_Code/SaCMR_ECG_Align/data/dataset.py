"""Lazy paired loading with the existing UKB ST-MEM preprocessing."""

import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample, sosfiltfilt
from torch.utils.data import Dataset

from SaCMR_encoder.data.dataset import ShortAxisNPZDataset
from util.losses_pheno import PHENOTYPE_COLS


_HIGHPASS = butter(5, 0.67, btype='highpass', fs=250, output='sos')
_LOWPASS = butter(5, 40, btype='lowpass', fs=250, output='sos')


def preprocess_ecg(array, training=False):
    """Convert a trusted 500 Hz, 12-lead ECG to train/evaluation ST-MEM crops.

    Output is float32 [12,2250] for training or [3,12,2250] for evaluation.
    Filtering and standardization follow data.dataset's UKB pipeline, without
    importing that module's unrelated imaging and external-dataset dependencies.
    """
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != 12 or array.shape[1] < 4500:
        raise ValueError('ECG must have shape [12, time>=4500] at 500 Hz; got {}'.format(array.shape))
    if not np.isfinite(array).all():
        raise ValueError('ECG contains nonfinite values')
    signal = resample(array, int(array.shape[1] * 250 / 500), axis=1)
    available = signal.shape[1] - 2250
    if training:
        start = np.random.randint(0, available + 1)
        signal = signal[:, start:start + 2250]
    else:
        # For the UKB 5000-point input this is exactly the existing 0/125/250
        # NCrop schedule; linspace also handles an input with no spare samples.
        starts = np.linspace(0, available, 3, dtype=int)
        signal = np.stack([signal[:, start:start + 2250] for start in starts])
    signal = sosfiltfilt(_HIGHPASS, signal)
    signal = sosfiltfilt(_LOWPASS, signal)
    mean = signal.mean(axis=(-1, -2), keepdims=True)
    std = signal.std(axis=(-1, -2), keepdims=True)
    standardized = np.divide(signal.astype(np.float32) - mean, std,
                             out=np.zeros(signal.shape, dtype=np.float32), where=std != 0)
    return torch.from_numpy(standardized)


class PairedSACMRECGDataset(Dataset):
    """Consume prepared CSVs without discovering or substituting samples.

    ECG files are trusted project-generated pickle arrays. SA arrays always pass
    the shared NPZ validation when loaded, including when preparation used only
    file metadata. Missing/corrupt samples raise an error containing their path.
    """

    def __init__(self, csv_path, training=False, load_ecg=True, size=96, max_samples=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError('size must be a positive integer')
        if max_samples is not None and (not isinstance(max_samples, int)
                                        or isinstance(max_samples, bool) or max_samples <= 0):
            raise ValueError('max_samples must be a positive integer or None')
        with self.csv_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            missing = set(['eid', 'sa_path', 'ecg_path'] + PHENOTYPE_COLS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError('{} is missing columns: {}'.format(self.csv_path, sorted(missing)))
            self.rows = list(reader)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError('No paired samples in {}'.format(self.csv_path))
        self.phenotypes = np.asarray([[float(row[name]) for name in PHENOTYPE_COLS]
                                      for row in self.rows], dtype=np.float32)
        if not np.isfinite(self.phenotypes).all():
            raise ValueError('Nonfinite phenotype targets in {}'.format(self.csv_path))
        self.phenotype_names = list(PHENOTYPE_COLS)
        self.training = bool(training)
        self.load_ecg = bool(load_ecg)
        self.size = size

    def __len__(self):
        return len(self.rows)

    def _path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.csv_path.parent / path

    def __getitem__(self, index):
        row = self.rows[index]
        image, mask = ShortAxisNPZDataset.load_validated_arrays(self._path(row['sa_path']))
        image = image.astype(np.float32, copy=True)
        image[:, :, ~mask, :] = 0.0
        image = torch.from_numpy(image).permute(2, 3, 0, 1).contiguous()
        if self.size != 112:
            image = F.interpolate(image, size=(self.size, self.size), mode='bilinear', align_corners=False)
        result = {'sa_cmr': image, 'slice_valid_mask': torch.from_numpy(mask.copy()),
                  'phenotype': torch.from_numpy(self.phenotypes[index].copy()), 'eid': row['eid']}
        if self.load_ecg:
            path = self._path(row['ecg_path'])
            try:
                with path.open('rb') as handle:
                    array = pickle.load(handle)
                result['ecg'] = preprocess_ecg(array, self.training)
            except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
                raise ValueError('Failed to load ECG {}: {}. Check this file before training.'.format(
                    path, error)) from error
        return result
