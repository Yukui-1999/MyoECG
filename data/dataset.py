from __future__ import annotations
import copy
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, resample, sosfiltfilt, square, iirnotch, filtfilt, medfilt
from typing import Any, Dict, List, Optional, Tuple, Union
import wfdb 
try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_12_leads(ecg_data):
    """
    Plots a 12-lead ECG in a 2x6 grid format.
    
    Parameters:
    - ecg_data: A 2D numpy array of shape (12, length) representing the 12 leads of ECG signals.
    
    Returns:
    - None
    """
    # Check if the input data has the correct shape
    if ecg_data.shape[0] != 12:
        raise ValueError("Input ECG data must have 12 leads (shape should be (12, length))")
    
    # Set up the 2x6 grid for the 12 leads
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
    
    # Flatten the 2D array of axes for easy iteration
    axs = axs.flatten()
    
    # Define the lead names (if desired)
    lead_names = [
        'Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ]
    
    # Plot each lead on the 2x6 grid
    for i in range(12):
        axs[i].plot(ecg_data[i, :])  # Plot each lead
        axs[i].set_title(lead_names[i])  # Set the title for each subplot
        axs[i].grid(True)  # Add grid for better visibility

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f'ecg_12_leads.png')

# Example usage:
# Assuming `ecg_data` is a numpy array of shape (12, length)
# ecg_data = np.random.randn(12, 1000)  # Replace with actual ECG data
# plot_ecg_12_leads(ecg_data)


def image_normalization(image, scale=1, mode="2D"):
    if isinstance(image, np.ndarray) and np.iscomplexobj(image):
        image = np.abs(image)
    low = image.min()
    high = image.max()
    im_ = (image - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_

class Resample:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = 500) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class Resample250:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = 250) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class Resample1000:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = 1000) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class SOSFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(start=0,
                              stop=x.shape[1] - self.crop_length + 1,
                              step=(x.shape[1] - self.crop_length) // (self.num_segments - 1))
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)
    
class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        x = x.astype(np.float32)
        # Set rst = 0 if std = 0
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)
class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)

class ReplaceNaN:
    """Replace NaN values with 0."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0)

class ReorderLeads:
    """Reorder ECG leads based on specified input and target orders."""
    def __init__(self, 
                 input_leads: List[str] = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                 new_leads: List[str] = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']) -> None:
        self.lead_indices = [input_leads.index(lead) for lead in new_leads]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Assuming shape is [leads, length]
        return x[self.lead_indices, :]


class BaselineFilterTransform:
    """Apply the baseline's specific notch, bandpass, and median filters."""
    def __init__(self, fs: int = 500) -> None:
        self.fs = fs
        # 预计算陷波滤波器系数 (50Hz 去工频干扰)
        self.notch_b, self.notch_a = iirnotch(50, 30, fs)
        # 预计算带通滤波器系数 (0.67 - 40Hz)
        self.bp_b, self.bp_a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
        
        # 预计算中值滤波的 kernel_size
        kernel_size = int(0.4 * fs) + 1
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x shape: (leads, length)
        filtered_signal = np.zeros_like(x)
        
        # 1. 陷波滤波 & 带通滤波
        for c in range(x.shape[0]):
            notch_out = filtfilt(self.notch_b, self.notch_a, x[c])
            filtered_signal[c] = filtfilt(self.bp_b, self.bp_a, notch_out)
            
        # 2. 中值滤波去除基线漂移
        baseline = np.zeros_like(filtered_signal)
        for c in range(filtered_signal.shape[0]):
            baseline[c] = medfilt(filtered_signal[c], kernel_size=self.kernel_size)
            
        # 3. 减去基线
        return filtered_signal - baseline


class MinMaxNormalize:
    """
    Min-Max normalization to scale input sequence to [-1, 1] per channel.
    Replaces the second baseline's ecg_transform.
    """
    def __init__(self, axis: int = -1) -> None:
        # 默认 axis=-1，适配我们 (leads, length) 的数据格式
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_min = np.min(x, axis=self.axis, keepdims=True)
        x_max = np.max(x, axis=self.axis, keepdims=True)
        
        # 忽略除以 0 产生的警告，因为后面的 nan_to_num 会把 NaN 变成 0
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
            
        # 完美复刻原版逻辑：处理 NaN（包含原本的 NaN 和 刚产生的除零 NaN）
        return np.nan_to_num(x, nan=0.0)

class GlobalMinMaxNormalize:
    """
    Global Min-Max normalization to scale the entire ECG to [0, 1].
    Replaces the third baseline's transform.
    """
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # np.min() 和 np.max() 不加 axis，代表计算全局极值
        # x_min = np.min(x, axis=1, keepdims=True)
        # x_max = np.max(x, axis=1, keepdims=True)
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min + self.eps)    



# ==========================================================
# HENAN 数据集 (原始 1000Hz -> 目标 500Hz, 10秒整)
# ==========================================================
# ==========================================================
# HARVARD 数据集 (原始 250Hz -> 目标 500Hz, 10秒整)
# ==========================================================
# ==========================================================
# UKB / MIMIC / ZJU 数据集 (原始 500Hz -> 目标 500Hz, 10秒整)
# ==========================================================

STMEM_ECG_TRANSFORMS_TRAIN_HENAN = Compose([
    Resample1000(target_fs=250),
    RandomCrop(2250),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

STMEM_ECG_TRANSFORMS_TEST_HENAN = Compose([
    Resample1000(target_fs=250),
    NCrop(2250, 3),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

STMEM_ECG_TRANSFORMS_TRAIN_HARVARD = Compose([
    RandomCrop(2250),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

STMEM_ECG_TRANSFORMS_TEST_HARVARD = Compose([
    NCrop(2250, 3),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])


STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU = Compose([
    Resample(target_fs=250),
    RandomCrop(2250),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU = Compose([
    Resample(target_fs=250),
    NCrop(2250, 3),
    HighpassFilter(250, 0.67),
    LowpassFilter(250, 40),
    Standardize(axis=(-1, -2)),
    ToTensor()
])





BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HENAN = Compose([
    ReplaceNaN(),
    ReorderLeads(),
    Resample1000(target_fs=500), 
    BaselineFilterTransform(fs=500), # 包含陷波、带通、去基线
    Standardize(axis=(-1, -2)), 
    ToTensor()
])
BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HARVARD = Compose([
    ReplaceNaN(),
    ReorderLeads(),
    Resample(target_fs=500),    
    BaselineFilterTransform(fs=500),
    Standardize(axis=(-1, -2)),
    ToTensor()
])
BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU = Compose([
    ReplaceNaN(),
    ReorderLeads(),
    BaselineFilterTransform(fs=500),
    Standardize(axis=(-1, -2)),
    ToTensor()
])



BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HENAN = Compose([
    Resample1000(target_fs=100),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HARVARD = Compose([
    Resample250(target_fs=100),
    Standardize(axis=(-1, -2)),
    ToTensor()
])

BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU = Compose([
    Resample(target_fs=100),
    Standardize(axis=(-1, -2)),
    ToTensor()
])





BASELINE_CLEP_TRANSFORMS_TRAINTEST_HENAN = Compose([
    Resample1000(target_fs=500),
    MinMaxNormalize(axis=-1),
    ToTensor()
])
BASELINE_CLEP_TRANSFORMS_TRAINTEST_HARVARD = Compose([
    Resample250(target_fs=500),
    MinMaxNormalize(axis=-1),
    ToTensor()
])

BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU = Compose([
    Resample(target_fs=500),
    MinMaxNormalize(axis=-1),
    ToTensor()
])



BASELINE_MERL_TRANSFORMS_TRAINTEST_HENAN = Compose([
    Resample1000(target_fs=500),
    GlobalMinMaxNormalize(eps=1e-8),
    ToTensor()
])
BASELINE_MERL_TRANSFORMS_TRAINTEST_HARVARD = Compose([
    Resample250(target_fs=500),
    GlobalMinMaxNormalize(eps=1e-8),
    ToTensor()
])
BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU = Compose([
    Resample(target_fs=500),
    GlobalMinMaxNormalize(eps=1e-8),
    ToTensor()
])



import pickle as pkl
class ECGCMRBase(Dataset):
    def __init__(self,txt_file,isTrain=True):
        print(f'Loading ECGCMR dataset from {txt_file}')
        self.json_file = txt_file
        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/ukb/processed_ecg'
        self.data = json.load(open(txt_file, "r"))
        self.eid = copy.deepcopy(self.data)
        for i in range(len(self.data)):
            file_name = self.data[i] + '_20208_2_0'
            self.data[i] = os.path.join(f'/home/liziyu/CMRGEN/cmrmar/cmr_data/ukb/{file_name}', file_name+'.pt')

        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item)
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        # print(image.shape)  # (bs,) c, f, h, w

        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, image



class ECGCMRTrain(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/UKBData/train_data_v1.json", isTrain=True )
class ECGCMRValid(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/UKBData/valid_data_v1.json", isTrain=False )
class ECGCMRTest(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/UKBData/test_data_v1.json", isTrain=False )

class ECGCMRAll(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/UKBData/all_data_v1.json", isTrain=False )

    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item)
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        # print(image.shape)  # (bs,) c, f, h, w

        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, image, item.split('/')[-1].split('_')[0]



class HeNan_ECGCMRBase(Dataset):
    def __init__(self,data_excel,isTrain=True):

        self.data = pd.read_excel(data_excel)
        
        self.ecg_data = self.data['ECG_data_path'].tolist()
        self.cmr_data = self.data['selected_data_HW_crop'].tolist()
        self.cmr_data = [self.process_cmr_path(path) for path in self.cmr_data]

        if isTrain:
            self.ecg_transforms = Compose([
                Resample1000(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample1000(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
    def process_cmr_path(self, path):
        cmr_path = path.replace('/mnt/', '')
        cmr_path = cmr_path.replace('/raw_data.nii', '.pt')
        cmr_path = cmr_path.replace('/','_')
        cmr_path = os.path.join('/mnt/sda1/dingzhengyao/data/HANAN_CMR_segout', cmr_path)
        return cmr_path
    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, i):

        ecg = pkl.load(open(self.ecg_data[i], 'rb'))
        ecg = self.ecg_transforms(ecg)

        
        cmr_path = self.cmr_data[i]
        cmr = torch.load(cmr_path)['img']
        # cmr to -1 1
        cmr = 2 * (cmr - np.min(cmr)) / (np.max(cmr) - np.min(cmr)) - 1
        cmr = cmr.transpose(2, 0, 1)
        cmr = torch.tensor(cmr, dtype=torch.float32)
        return ecg, cmr


class HeNan_ECGCMRTrain(HeNan_ECGCMRBase):
    def __init__(self):
        super().__init__(data_excel='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407/train.xlsx', isTrain=True)

class HeNan_ECGCMRValid(HeNan_ECGCMRBase):
    def __init__(self):
        super().__init__(data_excel='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407/valid.xlsx', isTrain=False)

class HeNan_ECGCMRTest(HeNan_ECGCMRBase):
    def __init__(self):
        super().__init__(data_excel='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407/test.xlsx', isTrain=False)




class ECGBaseDis(Dataset):
    def __init__(self,data,isTrain=True,args=None):


        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/ukb/processed_ecg'
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.eid = [str(eid) for eid in self.eid]
        self.labels = list(labels)
        if -1 in self.labels:
            label_idx = [i for i, label in enumerate(self.labels) if label != -1]
            self.eid = [self.eid[i] for i in label_idx]
            self.labels = [self.labels[i] for i in label_idx]
            
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        total_count = len(self.labels)
        print(f'Total samples: {total_count}')
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')

        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, i):
        item = self.eid[i]
        label = self.labels[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg,ecg, label

class CMRBaseDis(Dataset):
    def __init__(self,data,isTrain=True,args=None):
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.eid = [str(eid) for eid in self.eid]
        self.labels = list(labels)
        
        for i in range(len(self.data)):
            file_name = self.data[i] + '_20208_2_0'
            self.cmr[i] = os.path.join(f'/home/liziyu/CMRGEN/cmrmar/cmr_data/ukb/{file_name}', file_name+'.pt')


    def __len__(self):
        return len(self.cmr)

    def __getitem__(self, i):
        item = self.cmr[i]
        image = torch.load(item)
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        # print(image.shape)  # (bs,) c, f, h, w
        label = self.labels[i]
        return image, label

class HeNan_ECGBaseDis(Dataset):
    def __init__(self,data_excel,isTrain=True,args=None):

        if type(data_excel) == str:
            if data_excel.endswith('.xlsx'):
                self.data_excel = pd.read_excel(data_excel)
            elif data_excel.endswith('.csv'):
                self.data_excel = pd.read_csv(data_excel)
        else:
            self.data_excel = data_excel
        self.ecg_paths = self.data_excel['ECG_data_path'].tolist()
        self.cmr_paths = self.data_excel['selected_data_HW_crop'].tolist()
        self.cmr_paths = [self.process_cmr_path(path) for path in self.cmr_paths]
        self.identifiers = self.data_excel['Unnamed: 0'].tolist()
        self.args = args
        labels = []
        if args.dis == 'cm':
            print(f'dis: {args.dis}')
            diag_list = self.data_excel['出院诊断'].tolist()
            for diag in diag_list:
                if 'I42' in diag or 'I43' in diag in diag:
                    labels.append(1)
                
                else:
                    labels.append(0)
            print(f'Number of positive samples: {labels.count(1)}, Number of negative samples: {labels.count(0)}')
        elif args.dis == 'cm_three':
            print(f'dis: {args.dis}')
            if 'label' in self.data_excel.columns:
                labels = self.data_excel['label'].tolist()
            else:
                diag_list = self.data_excel['出院诊断'].tolist()
                for diag in diag_list:  
                    if '限制' in diag or '淀粉样' in diag:
                        labels.append(0)
                    elif 'I42.1' in diag or 'I42.2' in diag:
                        labels.append(2)
                    elif 'I42.0' in diag:
                        labels.append(1)
                    else:
                        labels.append(-1)
            print(f'Number of RCM samples: {labels.count(0)}, Number of DCM samples: {labels.count(1)}, Number of HCM samples: {labels.count(2)}')
        else:
            pass
        
        self.labels = labels
        self.data_excel['label'] = labels
        if -1 in self.labels:
            label_None_idx = [i for i, label in enumerate(self.labels) if label == -1]
            label_idx = [i for i, label in enumerate(self.labels) if label != -1]
            self.identifiers = [self.identifiers[i] for i in range(len(self.identifiers)) if i not in label_None_idx]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i not in label_None_idx]
            self.ecg_paths = [self.ecg_paths[i] for i in range(len(self.ecg_paths)) if i not in label_None_idx]
            self.cmr_paths = [self.cmr_paths[i] for i in range(len(self.cmr_paths)) if i not in label_None_idx]
            if args.cal_popular_index:
                
                self.valid_data = self.data_excel.iloc[label_idx]
                if args.dis == 'cm':
                    self.valid_data.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/popular', 'valid_data_cm_addLabel.xlsx'), index=False)
                elif args.dis == 'cm_three':
                    self.valid_data.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/popular', 'valid_data_cm_three_addLabel.xlsx'), index=False)      
                else:
                    raise ValueError('Please specify the disease type!')  
                exit()
        if args.cal_popular_index:
            if args.dis == 'cm':
                self.data_excel.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/popular', 'valid_data_cm_addLabel_Nofilter.xlsx'), index=False)
            elif args.dis == 'cm_three':
                self.data_excel.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/popular', 'valid_data_cm_three_addLabel.xlsx'), index=False)      
            else:
                raise ValueError('Please specify the disease type!')  
            exit()
            
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_HENAN
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_HENAN
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_HENAN
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_HENAN
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_HENAN
    def process_cmr_path(self, path):
        cmr_path = path.replace('/mnt/', '')
        cmr_path = cmr_path.replace('/raw_data.nii', '.pt')
        cmr_path = cmr_path.replace('/','_')
        cmr_path = os.path.join('/mnt/sda1/dingzhengyao/data/HANAN_CMR_segout', cmr_path)
        return cmr_path
    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, i):
        eid = self.identifiers[i]
        label = self.labels[i]
        
        ecg = pkl.load(open(self.ecg_paths[i], 'rb'))
        ecg = self.ecg_transforms(ecg)

        
        cmr_path = self.cmr_paths[i]
        cmr = torch.load(cmr_path)['img']
        # cmr to -1 1
        cmr = 2 * (cmr - np.min(cmr)) / (np.max(cmr) - np.min(cmr)) - 1
        cmr = cmr.transpose(2, 0, 1)
        cmr = torch.tensor(cmr, dtype=torch.float32)
        if self.args.record_eid:
            return ecg, eid, label
        else:
            return ecg, cmr, label


class Harvard_ECGBaseDis(Dataset):
    def __init__(self,data_df,isTrain=True,args=None):

        self.data_df = data_df
        self.ecg_data_list = self.data_df['save_path'].tolist()
        
        self.identifiers = self.data_df['FileName'].tolist()
        self.args = args
        
        if args.dis == 'cm':
            self.labels = self.data_df['Label'].tolist()
            print(f'Number of positive samples: {self.labels.count(1)}, Number of negative samples: {self.labels.count(0)}')
        elif args.dis == 'cm_three':
            self.labels = self.data_df['three_label'].tolist()
            print(f'Number of RCM samples: {self.labels.count(0)}, Number of DCM samples: {self.labels.count(1)}, Number of HCM samples: {self.labels.count(2)}')
        else:
            raise ValueError(f"Invalid dis type: {args.dis}")
        
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_HARVARD
            elif args.ecg_model == 'ecg_found':
                print(f'use ecg_found ecg transforms')
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_HARVARD
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_HARVARD
            elif args.ecg_model == 'ecg_found':
                print(f'use ecg_found ecg transforms')
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_HARVARD
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_HARVARD
    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, i):
        identifier = self.identifiers[i]
        label = self.labels[i]
        ecg = pkl.load(open(self.ecg_data_list[i], 'rb'))

        ecg = self.ecg_transforms(ecg)
        if self.args.record_eid:
            return ecg, identifier, label
        else:
            return ecg, ecg, label

class ECGBaseMIMICDis(Dataset):
    def __init__(self,data,isTrain=True,args=None):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = list(labels)
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/MIMIC/low_quality_mimic_data_path_CM.json', 'r'))
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.good_quality_labels)
        neg_count = sum(label == 0 for label in self.good_quality_labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
       
        return ecg,ecg, label
        






class ECGBaseMIMIC_CMthree(Dataset):
    def __init__(self,data,isTrain=True,args=None):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = np.array(list(labels))
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/MIMIC/low_quality_mimic_data_path_CM.json', 'r'))
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        
        count_ones = np.sum(self.labels == 1)
        count_twos = np.sum(self.labels == 2)
        count_threes = np.sum(self.labels == 0)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
       
        return ecg,ecg, label

        

class ECGzheyi_three_Base(Dataset):
    def __init__(self, data=None,isTrain=True,args=None):
        
        # self.ecg = [tup[0] for tup in data]
        # self.labels = [tup[1] for tup in data]
        # print(f'length of labels: {len(self.labels)}')
        
        # self.valid_index = [i for i, label in enumerate(self.labels) if label != -1]
        # self.ecg = [self.ecg[i] for i in self.valid_index]
        # self.labels = [self.labels[i] for i in self.valid_index]
        # print(f'length of labels: {len(self.labels)}')
        
        
        ecg, labels, path = zip(*data)
        self.ecg = list(ecg)
        self.labels = np.array(list(labels))
        self.path = list(path)
        print(self.labels)
        
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        
        count_ones = sum(label == 1 for label in self.labels)
        count_twos = sum(label == 2 for label in self.labels)
        count_threes = sum(label == 0 for label in self.labels)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')

    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg,ecg, label



class ECGzheyi_two_Base(Dataset):
    def __init__(self, data=None,isTrain=True,args=None):
        
        self.ecg = [tup[0] for tup in data]
        self.labels = [tup[1] for tup in data]
        self.labels = [ 0 if i == -1 else 1 for i in self.labels]
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg,ecg, label


class ECGzheyi_two_BaseCMR(Dataset):
    def __init__(self, data_path="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/nejm_revision_zheyi_cmr/zheyi_data_final3.pkl",isTrain=True,args=None):
        if type(data_path) == str:
            data = pkl.load(open(data_path, "rb"))
        else:
            data = data_path
        self.args = args
        self.ecg = data['ecg']
        self.labels = data['label']
        self.labels = [ 0 if i == 0 else 1 for i in self.labels]
        self.cmr = data['img']
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        img = self.cmr[i]
        img = np.transpose(img, (2, 0, 1))
        img = (image_normalization(img)-0.5)*2
        img = torch.from_numpy(img)
        img = img.float()
        if self.args.input_modality == 'ECG':
            return ecg, label
        elif self.args.input_modality == 'CMR':
            return img, label
        

import ast
import ntpath
import re
import pandas as pd


def normalize_one_path(p):
    """
    统一单个 Windows path
    """
    p = str(p).strip()
    p = p.strip('"').strip("'")

    # 把多个连续反斜杠统一成一个反斜杠
    # 比如 D:\\\\documents -> D:\documents
    p = re.sub(r'\\+', r'\\', p)

    # 统一 Windows 路径格式
    p = ntpath.normpath(p)
    p = ntpath.normcase(p)

    return p


def normalize_tuple_like_path(x, sort_paths=True):
    """
    把 tuple-like 字符串统一成可比较的 key

    输入可以是：
    1. "('path1', 'path2')"
    2. ('path1', 'path2')
    3. 'path1'
    """

    if pd.isna(x):
        return None

    x = str(x).strip()

    try:
        parsed = ast.literal_eval(x)
    except Exception:
        parsed = x

    if isinstance(parsed, (tuple, list)):
        paths = [normalize_one_path(p) for p in parsed]
    else:
        paths = [normalize_one_path(parsed)]

    # 如果 tuple 里两个 path 顺序可能不同，建议排序
    if sort_paths:
        paths = sorted(paths)

    return tuple(paths)

class ECGzheer_two_Base(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/zheer_data/finetuned_data/zheer_new_data.pkl",isTrain=True,args=None):
        if type(data_path) == str:
            data = pkl.load(open(data_path, "rb"))
        else:
            data = data_path
        if args.group_analysis_csv:
            zheer_data_grouped = {
                'ecg': [],
                'label': [],
                'path': []
            }

            group_csv = pd.read_csv(args.group_analysis_csv)
            print(f"Group analysis CSV loaded: {args.group_analysis_csv}")

            group_eid = group_csv['eid'].tolist()

            group_key_set = set(
                normalize_tuple_like_path(x)
                for x in group_eid
                if normalize_tuple_like_path(x) is not None
            )

            print(f'Number of group keys: {len(group_key_set)}')

            for ecg, label, path in zip(data['ecg'], data['label'], data['path']):
                path_key = normalize_tuple_like_path(path)

                if path_key in group_key_set:
                    zheer_data_grouped['ecg'].append(ecg)
                    zheer_data_grouped['label'].append(label)
                    zheer_data_grouped['path'].append(path)

            print('Grouped ECG number:', len(zheer_data_grouped['ecg']))
            data = zheer_data_grouped
            
        self.args = args
        self.ecg = data['ecg']
        self.labels = data['label']
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg,ecg, label





class ECGquzhou_two_Base(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/quzhou_data/quzhouECG_v3.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = np.array(data['label'])
        self.labels[self.labels == 2] = 1
        print(self.labels)
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg, label




class renji_dataset(Dataset):
    def __init__(self, args,isTrain=False):
        data = pkl.load(open(args.data_path, "rb"))
        self.data_ecg = data['ecg']
        self.data_eid = data['binglihao']
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
    
    def __len__(self):
        return len(self.data_ecg)
    def __getitem__(self, index):
        ecg = self.data_ecg[index]
        ecg = self.ecg_transforms(ecg)
        eid = self.data_eid[index]
        return ecg, eid


from pathlib import Path
from typing import Union, Optional, Tuple

import pickle
import sys
import types
TARGET_FS_HZ = 500
TARGET_DURATION_S = 10.0
TARGET_LENGTH = int(TARGET_FS_HZ * TARGET_DURATION_S)
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MDC_LEAD_MAP = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_aVR": "aVR",
    "MDC_ECG_LEAD_aVL": "aVL",
    "MDC_ECG_LEAD_aVF": "aVF",
    "MDC_ECG_LEAD_V1": "V1",
    "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3",
    "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5",
    "MDC_ECG_LEAD_V6": "V6",
}


def install_pandas_pickle_shim() -> None:
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    shim = types.ModuleType("pandas.core.indexes.numeric")
    shim.Int64Index = pd.Index
    shim.UInt64Index = pd.Index
    shim.Float64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = shim


def load_pickle_compat(path: Union[str, Path]) -> Any:
    install_pandas_pickle_shim()
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def parse_fs_hz(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        fs_hz = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None
    return fs_hz


def resample_signal(signal: np.ndarray, original_fs_hz: float, target_fs_hz: float) -> np.ndarray:
    if int(round(original_fs_hz)) == int(round(target_fs_hz)):
        return signal.astype(np.float32, copy=False)

    old_length = signal.shape[1]
    new_length = max(1, int(round(old_length * target_fs_hz / original_fs_hz)))

    old_time = np.arange(old_length, dtype=np.float64) / float(original_fs_hz)
    new_time = np.arange(new_length, dtype=np.float64) / float(target_fs_hz)

    resampled = np.empty((signal.shape[0], new_length), dtype=np.float32)
    for lead_idx in range(signal.shape[0]):
        resampled[lead_idx] = np.interp(new_time, old_time, signal[lead_idx]).astype(
            np.float32
        )
    return resampled


def pad_or_crop(signal: np.ndarray, target_length: Optional[int]) -> np.ndarray:
    if target_length is None:
        return signal
    current_length = signal.shape[1]
    if current_length == target_length:
        return signal
    if current_length > target_length:
        start = (current_length - target_length) // 2
        end = start + target_length
        return signal[:, start:end]

    padded = np.zeros((signal.shape[0], target_length), dtype=np.float32)
    padded[:, :current_length] = signal
    return padded


def load_signal_and_fs_from_pickle(path: Union[str, Path]) -> tuple[np.ndarray, Optional[float]]:
    record = load_pickle_compat(path)

    if isinstance(record, dict) and "signal" in record:
        signal = np.asarray(record["signal"], dtype=np.float32)
        lead_order = [MDC_LEAD_MAP.get(name, name) for name in record.get("lead_order", [])]
        if signal.ndim != 2:
            raise ValueError(f"Unsupported signal ndim for {path}: {signal.ndim}")
        if signal.shape[0] != len(lead_order) and signal.shape[1] == len(lead_order):
            signal = signal.T
        if lead_order:
            lead_to_index = {lead_name: idx for idx, lead_name in enumerate(lead_order)}
            signal = np.stack([signal[lead_to_index[lead]] for lead in STANDARD_LEADS], axis=0)
        return signal, parse_fs_hz(record.get("fs"))

    if isinstance(record, dict) and "waveform" in record:
        waveform = record["waveform"]
        if isinstance(waveform, pd.DataFrame):
            missing_leads = [lead for lead in STANDARD_LEADS if lead not in waveform.columns]
            if missing_leads:
                raise ValueError(f"Missing leads {missing_leads} in waveform for {path}")
            return waveform[STANDARD_LEADS].to_numpy(dtype=np.float32).T, parse_fs_hz(
                record.get("fs_hz")
            )

    raise ValueError(f"Unsupported pickle format for {path}")


class ShaoyifuCardiomyopathyDataset(Dataset):
    def __init__(
        self,
        table_path: str | Path,
        task: str = "binary",
        isTrain: bool = True,
        args=None,
        transform: Any = None,
        return_metadata: bool = False,
    ) -> None:
        self.table_path = Path(table_path)
        self.task = task
        self.target_fs_hz = TARGET_FS_HZ
        self.target_duration_s = TARGET_DURATION_S
        self.target_length = TARGET_LENGTH
        self.transform = transform
        self.return_metadata = return_metadata

        self.df = pd.read_csv(self.table_path)
        # df有一列叫'quality',里面0代表没问题要保留，1代表质量有问题要丢弃
        if 'quality' in self.df.columns:
            original_len = len(self.df)
            self.df = self.df[self.df['quality'] == 0]
            filtered_len = len(self.df)
            print(f"Filtered out {original_len - filtered_len} low quality samples, remaining {filtered_len} samples.")
        if args.group_analysis_csv:
            group_csv = pd.read_csv(args.group_analysis_csv)
            print(f"Group analysis CSV loaded: {args.group_analysis_csv}")
            group_eid = set(group_csv['merged_record_id'].astype(str).tolist())
            self.df = self.df[self.df['merged_record_id'].astype(str).isin(group_eid)]
            print(f'After filtering with group analysis CSV, dataset size: {len(self.df)}')
        if task == "binary":
            self.label_column = "binary_label"
            print(f'label 0 number: {sum(self.df[self.label_column] == 0)}')
            print(f'label 1 number: {sum(self.df[self.label_column] == 1)}')
        elif task == "three_class":
            self.label_column = "three_class_label"
            self.df[self.label_column] = self.df[self.label_column].replace({0: 2, 2: 0})
            print(f'label 0 number RCM: {sum(self.df[self.label_column] == 0)}')
            print(f'label 1 number DCM: {sum(self.df[self.label_column] == 1)}')
            print(f'label 2 number HCM: {sum(self.df[self.label_column] == 2)}')
        else:
            raise ValueError(f"Unsupported task: {task}")

        if args.cal_popular_index:
            if task == "binary":
                self.df.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/ShaoyifuData/popular', 'valid_data_cm.xlsx'), index=False)
            elif task == "three_class":
                self.df.to_excel(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/ShaoyifuData/popular', 'valid_data_cm_three.xlsx'), index=False)
            
        
        if self.label_column not in self.df.columns:
            raise ValueError(f"{self.label_column} not found in {self.table_path}")
        
        if isTrain:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
        else:
            if args.ecg_model == 'stmem':
                self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecg_found':
                self.ecg_transforms = BASELINE_ECGFOUNDER_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'ecgfm_ked':
                self.ecg_transforms = BASELINE_ECGFMKED_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'fg_clep':
                self.ecg_transforms = BASELINE_CLEP_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
            elif args.ecg_model == 'merl':
                self.ecg_transforms = BASELINE_MERL_TRANSFORMS_TRAINTEST_UKB_MIMIC_ZJU
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        signal, pickle_fs_hz = load_signal_and_fs_from_pickle(row["rhythm_pkl_path"])
        original_fs_hz = parse_fs_hz(row.get("fs_hz"))
        if original_fs_hz is None:
            original_fs_hz = pickle_fs_hz
        if original_fs_hz is None:
            raise ValueError(f"Unable to determine fs_hz for {row['rhythm_pkl_path']}")

        signal = resample_signal(signal, original_fs_hz=original_fs_hz, target_fs_hz=self.target_fs_hz)
        signal = pad_or_crop(signal, self.target_length)
        if self.transform is not None:
            signal = self.transform(signal)
            
        signal = self.ecg_transforms(signal)
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32)
        label = int(row[self.label_column])
        

        if self.return_metadata:
            metadata = {
                "merged_record_id": row.get("merged_record_id"),
                "patient_id": row.get("patient_id"),
                "rhythm_pkl_path": row.get("rhythm_pkl_path"),
                "original_fs_hz": original_fs_hz,
                "target_fs_hz": self.target_fs_hz,
                "target_duration_s": self.target_duration_s,
                "binary_label_name": row.get("binary_label_name"),
                "three_class_label_name": row.get("three_class_label_name"),
                "diagnosis_items": row.get("诊断_items_all"),
            }
            return signal_tensor, label, metadata

        return signal_tensor,signal_tensor, label
