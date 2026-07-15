# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import util.transforms as T
from util.transforms import get_transforms_from_config, get_rand_augment_from_config
from util.misc import get_rank, get_world_size
from tqdm import tqdm
def extact_UKBinfo(eid_list,csv_data,work_edu_csv,save_name):
    select_index = [
        '4079-2.0',
        '4080-2.0',
        '21003-2.0',
        '31-0.0',
        '21000-0.0',
        '21001-2.0',
        '20160-2.0',
        '1160-2.0',
        '1558-2.0',
        '6142-2.0',
        '6138-2.0',
        '24100-2.0',
        '24101-2.0',
        '24102-2.0',
        '24103-2.0',
        '24104-2.0',
        '24105-2.0',
        '24106-2.0',
        '24107-2.0',    
        '24108-2.0',
        '24109-2.0',
    ]
    
    select_index_name_number = {
        'DBP':[],
        'SBP':[],
        'Age':[],
        'Sex_Female':0,
        'Sex_Male':0,
        'Ethnic_White':0,
        'Ethnic_Mixed':0,
        'Ethnic_Asian or Asian British':0,
        'Ethnic_Black or Black British':0,
        'Ethnic_Chinese':0,
        'Ethnic_Other ethnic group':0,
        'Ethnic_Unknown':0,
        'BMI':[],
        'Ever_smoked_yes':0,
        'Ever_smoked_no':0,
        'Ever_smoked_Unknown':0,
        'Sleep_duration':[],
        'Alcohol_Daily or almost daily':0,
        'Alcohol_Three or four times a week':0,
        'Alcohol_Once or twice a week':0,
        'Alcohol_One to three times a month':0,
        'Alcohol_Special occasions only':0,
        'Alcohol_Never':0,
        'Alcohol_Unknown':0,
        'Work_In paid employment or self-employed':0,
        'Work_Retired':0,
        'Work_Looking after home and/or family':0,
        'Work_Unable to work because of sickness or disability':0,
        'Work_Unemployed':0,
        'Work_Doing unpaid or voluntary work':0,
        'Work_Full or part-time student':0,
        'Work_Unknown':0,
        'Education_College or University degree':0,
        'Education_A levels/AS levels or equivalent':0,
        'Education_O levels/GCSEs or equivalent':0,
        'Education_CSEs or equivalent':0,
        'Education_NVQ or HND or HNC or equivalent':0,
        'Education_Other professional qualifications eg: nursing, teaching':0,
        'Education_Unknown':0,
        'LV end diastolic volume':[],
        'LV end systolic volume':[],
        'LV stroke volume':[],
        'LV ejection fraction':[],
        'LV cardiac output':[],
        'LV myocardial mass':[],
        'RV end diastolic volume':[],
        'RV end systolic volume':[],
        'RV stroke volume':[],
        'RV ejection fraction':[],
    }
    
    
    for eid in tqdm(eid_list):
        if int(eid) not in csv_data['eid'].values:
            print(f'{eid} not in csv_data')
            continue
        if int(eid) not in work_edu_csv['eid'].values:
            print(f'{eid} not in work_edu_csv')
            continue
        index = np.where(csv_data['eid'].values == int(eid))[0][0]
        index_work_edu = np.where(work_edu_csv['eid'].values == int(eid))[0][0]
        for i, index_name in enumerate(select_index):
            if index_name == '6142-2.0' or index_name == '6138-2.0':
                work_edu_row = work_edu_csv.iloc[index_work_edu]
                
                if index_name == '6142-2.0':
                    work = work_edu_row['6142-2.0']
                    if pd.isna(work):
                        select_index_name_number['Work_Unknown'] += 1
                    else:
                        work = int(work)
                        if work == 1:
                            select_index_name_number['Work_In paid employment or self-employed'] += 1
                        elif work == 2:
                            select_index_name_number['Work_Retired'] += 1
                        elif work == 3:
                            select_index_name_number['Work_Looking after home and/or family'] += 1
                        elif work == 4:
                            select_index_name_number['Work_Unable to work because of sickness or disability'] += 1
                        elif work == 5:
                            select_index_name_number['Work_Unemployed'] += 1
                        elif work == 6:
                            select_index_name_number['Work_Doing unpaid or voluntary work'] += 1
                        elif work == 7:
                            select_index_name_number['Work_Full or part-time student'] += 1
                        else:
                            select_index_name_number['Work_Unknown'] += 1
                
                elif index_name == '6138-2.0':
                    education = work_edu_row['6138-2.0']
                    if pd.isna(education):
                        select_index_name_number['Education_Unknown'] += 1
                    else:
                        education = int(education)
                        if education == 1:
                            select_index_name_number['Education_College or University degree'] += 1
                        elif education == 2:
                            select_index_name_number['Education_A levels/AS levels or equivalent'] += 1
                        elif education == 3:
                            select_index_name_number['Education_O levels/GCSEs or equivalent'] += 1
                        elif education == 4:
                            select_index_name_number['Education_CSEs or equivalent'] += 1
                        elif education == 5:
                            select_index_name_number['Education_NVQ or HND or HNC or equivalent'] += 1
                        elif education == 6:
                            select_index_name_number['Education_Other professional qualifications eg: nursing, teaching'] += 1
                        else:
                            select_index_name_number['Education_Unknown'] += 1
            
            else:
                data_row = csv_data.iloc[index]
                if index_name == '4079-2.0':
                    select_index_name_number['DBP'].append(float(data_row[index_name]))
                elif index_name == '4080-2.0':
                    select_index_name_number['SBP'].append(float(data_row[index_name]))
                elif index_name == '21003-2.0':
                    select_index_name_number['Age'].append(float(data_row[index_name]))
                elif index_name == '31-0.0':
                    sex = data_row[index_name]
                    sex = int(sex)
                    if sex == 0:
                        select_index_name_number['Sex_Female'] += 1
                    else:
                        select_index_name_number['Sex_Male'] += 1
                elif index_name == '21000-0.0':
                    ethnic = data_row[index_name]
                    if pd.isna(ethnic):
                        select_index_name_number['Ethnic_Unknown'] += 1
                    else:
                        ethnic = str(ethnic)
                        if ethnic.startswith('1'):
                            select_index_name_number['Ethnic_White'] += 1
                        elif ethnic.startswith('2'):
                            select_index_name_number['Ethnic_Mixed'] += 1
                        elif ethnic.startswith('3'):
                            select_index_name_number['Ethnic_Asian or Asian British'] += 1
                        elif ethnic.startswith('4'):
                            select_index_name_number['Ethnic_Black or Black British'] += 1
                        elif ethnic.startswith('5'):
                            select_index_name_number['Ethnic_Chinese'] += 1
                        elif ethnic.startswith('6'):
                            select_index_name_number['Ethnic_Other ethnic group'] += 1
                        else:
                            select_index_name_number['Ethnic_Unknown'] += 1
                elif index_name == '21001-2.0':
                    select_index_name_number['BMI'].append(float(data_row[index_name]))
                elif index_name == '20160-2.0':
                    smoke = data_row[index_name]
                    if pd.isna(smoke):
                        select_index_name_number['Ever_smoked_Unknown'] += 1
                    else:
                        smoke = int(smoke)
                        if smoke == 1:
                            select_index_name_number['Ever_smoked_yes'] += 1
                        else:
                            select_index_name_number['Ever_smoked_no'] += 1
                elif index_name == '1160-2.0':
                    select_index_name_number['Sleep_duration'].append(float(data_row[index_name]))
                elif index_name == '1558-2.0':
                    drike = data_row[index_name]
                    if pd.isna(drike):
                        select_index_name_number['Alcohol_Unknown'] += 1
                    else:
                        drike = int(drike)
                        if drike == 1:
                            select_index_name_number['Alcohol_Daily or almost daily'] += 1
                        elif drike == 2:
                            select_index_name_number['Alcohol_Three or four times a week'] += 1
                        elif drike == 3:
                            select_index_name_number['Alcohol_Once or twice a week'] += 1
                        elif drike == 4:
                            select_index_name_number['Alcohol_One to three times a month'] += 1
                        elif drike == 5:
                            select_index_name_number['Alcohol_Special occasions only'] += 1
                        elif drike == 6:
                            select_index_name_number['Alcohol_Never'] += 1
                        else:
                            select_index_name_number['Alcohol_Unknown'] += 1
                elif index_name == '24100-2.0':
                    select_index_name_number['LV end diastolic volume'].append(float(data_row[index_name]))
                elif index_name == '24101-2.0':
                    select_index_name_number['LV end systolic volume'].append(float(data_row[index_name]))
                elif index_name == '24102-2.0':
                    select_index_name_number['LV stroke volume'].append(float(data_row[index_name]))
                elif index_name == '24103-2.0':
                    select_index_name_number['LV ejection fraction'].append(float(data_row[index_name]))
                elif index_name == '24104-2.0':
                    select_index_name_number['LV cardiac output'].append(float(data_row[index_name]))
                elif index_name == '24105-2.0': 
                    select_index_name_number['LV myocardial mass'].append(float(data_row[index_name]))
                elif index_name == '24106-2.0':
                    select_index_name_number['RV end diastolic volume'].append(float(data_row[index_name]))
                elif index_name == '24107-2.0':
                    select_index_name_number['RV end systolic volume'].append(float(data_row[index_name]))
                elif index_name == '24108-2.0':
                    select_index_name_number['RV stroke volume'].append(float(data_row[index_name]))
                elif index_name == '24109-2.0':
                    select_index_name_number['RV ejection fraction'].append(float(data_row[index_name]))
                else:
                    raise ValueError(f'Unknown index name: {index_name}')
    
    processed_dict = {}
    for key, value in select_index_name_number.items():
        if isinstance(value, list) and len(value) > 0:
            mean = np.nanmean(value)
            std = np.nanstd(value)
            processed_dict[key] = f"{mean:.3f}±{std:.3f}"
        else:
            processed_dict[key] = value

    # 转换为DataFrame并导出CSV
    df = pd.DataFrame([processed_dict])
    df.to_csv(save_name, index=False)

    print(f"处理完成，结果已保存到 {save_name}")


def extact_MIMICinfo(eid_list,base_df,admission_df,save_name):
    eid_list = [i.replace('/mnt/sda1/lihaitao/datasets/ECG/','') for i in eid_list]
    eid_list = [i.replace('.dat','') for i in eid_list]

    type_id = [
        'gender',
        'race',
        'marital_status',
        'age',
        ]
    
    select_index_name_number = {
        'Age':[],
        'Sex_Female':0,
        'Sex_Male':0,
        'Sex_Unknown':0,
        'marital_status_WIDOWED':0,
        'marital_status_SINGLE':0,
        'marital_status_MARRIED':0,
        'marital_status_DIVORCED':0,
        'marital_status_Unknown':0,
        'Ethnic_WHITE':0,
        'Ethnic_BLACK':0,
        'Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':0,
        'Ethnic_PORTUGUESE':0,
        'Ethnic_HISPANIC':0,
        'Ethnic_ASIAN':0,
        'Ethnic_SOUTH AMERICAN':0,
        'Ethnic_AMERICAN INDIAN':0,
        'Ethnic_MULTIPLE RACE':0,
        'Ethnic_Unknown':0,
    }
    
    
    for eid in tqdm(eid_list):
        if int(eid.split('/')[-3][1:]) not in admission_df['subject_id'].values:
            # print(f"subject_id {eid} not in admission_df")
            select_index_name_number['Ethnic_Unknown'] += 1
            select_index_name_number['marital_status_Unknown'] += 1
            index_base = np.where(base_df['file_name'].values == eid)[0][0]
            base_row = base_df.iloc[index_base]

            gender = base_row['gender']
            if pd.isna(gender):
                select_index_name_number['Sex_Unknown'] += 1
            else:
                if gender == 'M':
                    select_index_name_number['Sex_Male'] += 1
                elif gender == 'F':
                    select_index_name_number['Sex_Female'] += 1
                else:
                    select_index_name_number['Sex_Unknown'] += 1

            age = base_row['age']
            select_index_name_number['Age'].append(float(age))
            continue
        
        index_base = np.where(base_df['file_name'].values == eid)[0][0]
        index_admission = np.where(admission_df['subject_id'].values == int(eid.split('/')[-3][1:]))[0][0]
        
        
        for i, index_name in enumerate(type_id):
            if index_name == 'gender' or index_name == 'age':
                base_row = base_df.iloc[index_base]
                if index_name == 'gender':
                    gender = base_row['gender']
                    if pd.isna(gender):
                        select_index_name_number['Sex_Unknown'] += 1
                    else:
                        if gender == 'M':
                            select_index_name_number['Sex_Male'] += 1
                        elif gender == 'F':
                            select_index_name_number['Sex_Female'] += 1
                        else:
                            select_index_name_number['Sex_Unknown'] += 1
                elif index_name == 'age':
                    age = base_row['age']
                    select_index_name_number['Age'].append(float(age))
            
            elif index_name == 'marital_status' or index_name == 'race':
                admission_df_row = admission_df.iloc[index_admission]
                if index_name == 'marital_status':
                    marital_status = admission_df_row['marital_status']
                    if pd.isna(marital_status):
                        select_index_name_number['marital_status_Unknown'] += 1
                    else:
                        if marital_status == 'WIDOWED':
                            select_index_name_number['marital_status_WIDOWED'] += 1
                        elif marital_status == 'SINGLE':
                            select_index_name_number['marital_status_SINGLE'] += 1
                        elif marital_status == 'MARRIED':
                            select_index_name_number['marital_status_MARRIED'] += 1
                        elif marital_status == 'DIVORCED':
                            select_index_name_number['marital_status_DIVORCED'] += 1
                        else:
                            select_index_name_number['marital_status_Unknown'] += 1
                elif index_name == 'race':
                    race = admission_df_row['race']
                    if pd.isna(race):
                        select_index_name_number['Ethnic_Unknown'] += 1
                    else:
                        if race.startswith('WHITE'):
                            select_index_name_number['Ethnic_WHITE'] += 1
                        elif race.startswith('BLACK'):
                            select_index_name_number['Ethnic_BLACK'] += 1
                        elif race.startswith('NATIVE HAWAIIAN'):
                            select_index_name_number['Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'] += 1
                        elif race.startswith('PORTUGUESE'):
                            select_index_name_number['Ethnic_PORTUGUESE'] += 1
                        elif race.startswith('HISPANIC'):
                            select_index_name_number['Ethnic_HISPANIC'] += 1
                        elif race.startswith('SOUTH AMERICAN'):
                            select_index_name_number['Ethnic_SOUTH AMERICAN'] += 1
                        elif race.startswith('AMERICAN INDIAN'):
                            select_index_name_number['Ethnic_AMERICAN INDIAN'] += 1
                        elif race.startswith('MULTIPLE RACE'):
                            select_index_name_number['Ethnic_MULTIPLE RACE'] += 1 
                        else:
                            select_index_name_number['Ethnic_Unknown'] += 1
                
            
    
    processed_dict = {}
    for key, value in select_index_name_number.items():
        if isinstance(value, list) and len(value) > 0:
            mean = np.nanmean(value)
            std = np.nanstd(value)
            processed_dict[key] = f"{mean:.3f}±{std:.3f}"
        else:
            processed_dict[key] = value

    # 转换为DataFrame并导出CSV
    df = pd.DataFrame([processed_dict])
    df.to_csv(save_name, index=False)

    print(f"处理完成，结果已保存到 {save_name}")


class ECGDataset(Dataset):
    _LEAD_SLICE = {"12lead": slice(0, 12),
                   "limb_lead": slice(0, 6),
                   "lead1": slice(0, 1),
                   "lead2": slice(1, 2)}

    def __init__(self,
                 root_dir: str,
                 filenames: Iterable = None,
                 labels: Optional[Iterable] = None,
                 fs_list: Optional[Iterable] = None,
                 target_lead: str = "12lead",
                 target_fs: int = 250,
                 transform: Optional[object] = None,
                 label_transform: Optional[object] = None):
        """
        Args:
            root_dir: Directory with all the data.
            filenames: List of filenames. (.pkl)
            labels: List of labels.
            fs_list: List of sampling rates.
            target_lead: lead to use. {'limb_lead', 'lead1', 'lead2'}
            target_fs: Target sampling rate.
            transform: Optional transform to be applied on a sample.
            label_transform: Optional transform to be applied on a label.
        """
        self.root_dir = root_dir
        self.filenames = filenames
        self.labels = labels
        self.target_lead = target_lead
        self.fs_list = fs_list
        self.check_dataset()
        self.resample = T.Resample(target_fs=target_fs) if fs_list is not None else None

        self.transform = transform
        self.label_transform = label_transform

    def check_dataset(self):
        fname_not_pkl = [f for f in self.filenames if not f.endswith('.pkl')]
        assert len(fname_not_pkl) == 0, \
            f"Some files do not have .pkl extension. (e.g. {fname_not_pkl[0]}...)"
        fpaths = [os.path.join(self.root_dir, fname) for fname in self.filenames]
        assert all([os.path.exists(fpath) for fpath in fpaths]), \
            f"Some files do not exist. (e.g. {fpaths[0]}...)"
        if self.labels is not None:
            assert len(self.filenames) == len(self.labels), \
                "The number of filenames and labels are different."
        if self.fs_list is not None:
            assert len(self.filenames) == len(self.fs_list), \
                "The number of filenames and fs_list are different."
        assert self.target_lead in self._LEAD_SLICE.keys(), \
            f"target_lead should be one of {list(self._LEAD_SLICE.keys())}"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        fpath = os.path.join(self.root_dir, fname)
        with open(fpath, 'rb') as f:
            x = pkl.load(f)
        assert isinstance(x, np.ndarray), f"Data should be numpy array. ({fpath})"
        x = x[self._LEAD_SLICE[self.target_lead]]
        if self.resample is not None:
            x = self.resample(x, self.fs_list[idx])
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            if self.label_transform:
                y = self.label_transform(y)
            return x, y
        else:
            return x


def build_dataset(cfg: dict, split: str) -> ECGDataset:
    """
    Load train, validation, and test dataloaders.
    """
    fname_col = cfg.get("filename_col", "FILE_NAME")
    fs_col = cfg.get("fs_col", None)
    label_col = cfg.get("label_col", None)
    target_lead = cfg.get("lead", "12lead")
    target_fs = cfg.get("fs", 250)

    index_dir = os.path.realpath(cfg["index_dir"])
    ecg_dir = os.path.realpath(cfg["ecg_dir"])

    df_name = cfg.get(f"{split}_csv", None)
    assert df_name is not None, f"{split}_csv is not defined in the config."
    df = pd.read_csv(os.path.join(index_dir, df_name))
    filenames = df[fname_col].tolist()
    fs_list = df[fs_col].astype(int).tolist() if fs_col is not None else None
    labels = df[label_col].astype(int).values if label_col is not None else None

    if split == "train":
        transforms = get_transforms_from_config(cfg["train_transforms"])
        randaug_config = cfg.get("rand_augment", {})
        use_randaug = randaug_config.get("use", False)
        if use_randaug:
            randaug_kwargs = randaug_config.get("kwargs", {})
            transforms.append(get_rand_augment_from_config(randaug_kwargs))
    else:
        transforms = get_transforms_from_config(cfg["eval_transforms"])
    transforms = T.Compose(transforms + [T.ToTensor()])
    label_transform = T.ToTensor(dtype=cfg["label_dtype"]) if labels is not None else None

    dataset = ECGDataset(ecg_dir,
                         filenames=filenames,
                         labels=labels,
                         fs_list=fs_list,
                         target_lead=target_lead,
                         target_fs=target_fs,
                         transform=transforms,
                         label_transform=label_transform)

    return dataset


def get_dataloader(dataset: Dataset,
                   is_distributed: bool = False,
                   dist_eval: bool = False,
                   mode: Literal["train", "eval"] = "train",
                   **kwargs) -> DataLoader:
    is_train = mode == "train"
    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        # shuffle=True to reduce monitor bias even if it is for validation.
        # https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L189
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=num_tasks,
                                                                  rank=global_rank,
                                                                  shuffle=True)
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return DataLoader(dataset,
                      sampler=sampler,
                      drop_last=is_train,
                      **kwargs)
