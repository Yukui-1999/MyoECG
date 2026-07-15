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
import nibabel as nib

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





import torch.nn.functional as F
class UKB_SingleSaxCMR_Lax4chCMR_ECG_Base(Dataset):
    def __init__(self,txt_file,isTrain=True):
        print(f'Loading ECGCMR dataset from {txt_file}')
        self.data = pd.read_csv(txt_file)
        self.sax_cmr_path = self.data['sa_cmr'].tolist()
        self.lax_cmr_path = self.data['la_cmr'].tolist()
        self.ecg_path = self.data['ecg_list'].tolist()
        self.eid = self.data['Eid'].tolist()
        self.sex = self.data['sex'].tolist()
        self.age = self.data['age'].tolist()
        self.bmi = self.data['BMI'].tolist()
        phenotype_cols = [
            'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVCO', 'LVM',
            'RVEDV', 'RVESV', 'RVSV', 'RVEF',
            'WT_AHA_1', 'WT_AHA_2', 'WT_AHA_3', 'WT_AHA_4',
            'WT_AHA_5', 'WT_AHA_6', 'WT_AHA_7', 'WT_AHA_8',
            'WT_AHA_9', 'WT_AHA_10', 'WT_AHA_11', 'WT_AHA_12',
            'WT_AHA_13', 'WT_AHA_14', 'WT_AHA_15', 'WT_AHA_16',
            'WT_Global'
        ]

        self.phenotype_names = phenotype_cols
        self.phenotypes = self.data[phenotype_cols].to_numpy(dtype='float32')
        self.size = 96
        assert self.phenotypes.shape[1] == 27

        self._length = len(self.ecg_path)

        if isTrain:
            self.ecg_transforms = STMEM_ECG_TRANSFORMS_TRAIN_UKB_MIMIC_ZJU
        else:
            self.ecg_transforms = STMEM_ECG_TRANSFORMS_TEST_UKB_MIMIC_ZJU

    def __len__(self):
        return self._length
    def _read_CMR(self,dir, eid):
        dir = dir.replace('/mnt/sda1/dingzhengyao/Work/Unified_ECGCMR/mutimage/Real_data', '/home/dingzhengyao/ECG_CMR_data')
        img = torch.load(os.path.join(dir, eid + '_img_int8.pt'))
        img = np.array(img).astype(np.uint8)
        # h, w, = img.shape[0], img.shape[1]
        img = (img / 127.5 - 1.0).astype(np.float32)
        if img.shape[-1] == 50:
            img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)

        if img.shape[-1] != self.size:
            # 把图像（c，h，w）插值到（c，size，size）
            assert  img.shape[-1] == img.shape[-2]
            img = F.interpolate(
                img.unsqueeze(0),  # (C, H, W) -> (1, C, H, W)
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img
    def _processSeg(self,img):

        if img.shape[-1] == 50:
            img = img.permute(2, 0, 1).contiguous()
        if img.shape[-1] != self.size:
            img = F.interpolate(
                img[None].float(),  # (1, 50, 80, 80)
                size=(96, 96),
                mode="nearest",
            )[0].long()
        return img
    def __getitem__(self, idx):
        with open(self.ecg_path[idx].replace('/mnt/sda1/liziyu/CMR_data/ukb/','/home/dingzhengyao/ECG_CMR_data/'), 'rb') as f:
            ecg = pkl.load(f)
        ecg = self.ecg_transforms(ecg)
        eid = str(self.eid[idx])
        la_cmr = self._read_CMR(self.lax_cmr_path[idx], eid)
        sa_cmr = self._read_CMR(self.sax_cmr_path[idx], eid)

        la_cmr_seg = self._processSeg(torch.load(os.path.join(self.lax_cmr_path[idx].replace('/mnt/sda1/dingzhengyao/Work/Unified_ECGCMR/mutimage/Real_data', '/home/dingzhengyao/ECG_CMR_data'), eid + '_seg.pt')))
        sa_cmr_seg = self._processSeg(torch.load(os.path.join(self.sax_cmr_path[idx].replace('/mnt/sda1/dingzhengyao/Work/Unified_ECGCMR/mutimage/Real_data', '/home/dingzhengyao/ECG_CMR_data'), eid + '_seg.pt')))
        phenotypes = self.phenotypes[idx]
        # print(f'eid: {eid}, ecg: {ecg.shape}, la_cmr: {la_cmr.shape}, sa_cmr: {sa_cmr.shape}, la_cmr_seg: {la_cmr_seg.shape}, sa_cmr_seg: {sa_cmr_seg.shape}, phenotypes: {phenotypes.shape}')
        # raise RuntimeError("debug stop")
        data_dict = {
            'eid': str(eid),
            'ecg': ecg,
            'la_cmr': la_cmr,
            'sa_cmr': sa_cmr,
            'la_cmr_seg': la_cmr_seg,
            'sa_cmr_seg': sa_cmr_seg,
            'phenotypes': phenotypes,
            'age': np.float32(self.age[idx]),
            'sex': np.int64(self.sex[idx]),
        }
        return data_dict


class UKB_SingleSaxCMR_Lax4chCMR_ECG_train(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/train_data.csv", isTrain=True )
class UKB_SingleSaxCMR_Lax4chCMR_ECG_valid(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/val_data.csv", isTrain=False )
class UKB_SingleSaxCMR_Lax4chCMR_ECG_test(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/test_data.csv", isTrain=False )



class UKB_SingleSaxCMR_Lax4chCMR_ECG_train_onlyLaxCMRECG(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/train_data.csv", isTrain=True )

    def __getitem__(self, idx):
        with open(self.ecg_path[idx].replace('/mnt/sda1/liziyu/CMR_data/ukb/', '/home/dingzhengyao/ECG_CMR_data/'), 'rb') as f:
            ecg = pkl.load(f)
        ecg = self.ecg_transforms(ecg)
        eid = str(self.eid[idx])
        image = self._read_CMR(self.lax_cmr_path[idx], eid)

        return ecg, image
class UKB_SingleSaxCMR_Lax4chCMR_ECG_valid_onlyLaxCMRECG(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/val_data.csv", isTrain=False )

    def __getitem__(self, idx):
        with open(self.ecg_path[idx].replace('/mnt/sda1/liziyu/CMR_data/ukb/', '/home/dingzhengyao/ECG_CMR_data/'), 'rb') as f:
            ecg = pkl.load(f)
        ecg = self.ecg_transforms(ecg)
        eid = str(self.eid[idx])
        image = self._read_CMR(self.lax_cmr_path[idx], eid)

        return ecg, image
class UKB_SingleSaxCMR_Lax4chCMR_ECG_test_onlyLaxCMRECG(UKB_SingleSaxCMR_Lax4chCMR_ECG_Base):
    def __init__(self):
        super().__init__(txt_file="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/split/test_data.csv", isTrain=False )

    def __getitem__(self, idx):
        with open(self.ecg_path[idx].replace('/mnt/sda1/liziyu/CMR_data/ukb/', '/home/dingzhengyao/ECG_CMR_data/'), 'rb') as f:
            ecg = pkl.load(f)
        ecg = self.ecg_transforms(ecg)
        eid = str(self.eid[idx])
        image = self._read_CMR(self.lax_cmr_path[idx], eid)

        return ecg, image




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

