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

import re

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








def extact_UKBinfo_persubject(eid_list, csv_data, work_edu_csv, save_name):
    """
    每个 eid 输出一行信息。

    连续变量：保留原始数值
    分类变量：转成 one-hot
    """

    columns = [
        'Eid',
        'DBP',
        'SBP',
        'Age',
        'Sex_Female',
        'Sex_Male',
        'Ethnic_White',
        'Ethnic_Mixed',
        'Ethnic_Asian or Asian British',
        'Ethnic_Black or Black British',
        'Ethnic_Chinese',
        'Ethnic_Other ethnic group',
        'Ethnic_Unknown',
        'BMI',
        'Ever_smoked_yes',
        'Ever_smoked_no',
        'Ever_smoked_Unknown',
        'Sleep_duration',
        'Alcohol_Daily or almost daily',
        'Alcohol_Three or four times a week',
        'Alcohol_Once or twice a week',
        'Alcohol_One to three times a month',
        'Alcohol_Special occasions only',
        'Alcohol_Never',
        'Alcohol_Unknown',
        'Work_In paid employment or self-employed',
        'Work_Retired',
        'Work_Looking after home and/or family',
        'Work_Unable to work because of sickness or disability',
        'Work_Unemployed',
        'Work_Doing unpaid or voluntary work',
        'Work_Full or part-time student',
        'Work_Unknown',
        'Education_College or University degree',
        'Education_A levels/AS levels or equivalent',
        'Education_O levels/GCSEs or equivalent',
        'Education_CSEs or equivalent',
        'Education_NVQ or HND or HNC or equivalent',
        'Education_Other professional qualifications eg: nursing, teaching',
        'Education_Unknown',
        'LV end diastolic volume',
        'LV end systolic volume',
        'LV stroke volume',
        'LV ejection fraction',
        'LV cardiac output',
        'LV myocardial mass',
        'RV end diastolic volume',
        'RV end systolic volume',
        'RV stroke volume',
        'RV ejection fraction',
    ]

    continuous_map = {
        '4079-2.0': 'DBP',
        '4080-2.0': 'SBP',
        '21003-2.0': 'Age',
        '21001-2.0': 'BMI',
        '1160-2.0': 'Sleep_duration',
        '24100-2.0': 'LV end diastolic volume',
        '24101-2.0': 'LV end systolic volume',
        '24102-2.0': 'LV stroke volume',
        '24103-2.0': 'LV ejection fraction',
        '24104-2.0': 'LV cardiac output',
        '24105-2.0': 'LV myocardial mass',
        '24106-2.0': 'RV end diastolic volume',
        '24107-2.0': 'RV end systolic volume',
        '24108-2.0': 'RV stroke volume',
        '24109-2.0': 'RV ejection fraction',
    }

    sex_cols = ['Sex_Female', 'Sex_Male']

    ethnic_cols = [
        'Ethnic_White',
        'Ethnic_Mixed',
        'Ethnic_Asian or Asian British',
        'Ethnic_Black or Black British',
        'Ethnic_Chinese',
        'Ethnic_Other ethnic group',
        'Ethnic_Unknown',
    ]

    smoke_cols = [
        'Ever_smoked_yes',
        'Ever_smoked_no',
        'Ever_smoked_Unknown',
    ]

    alcohol_cols = [
        'Alcohol_Daily or almost daily',
        'Alcohol_Three or four times a week',
        'Alcohol_Once or twice a week',
        'Alcohol_One to three times a month',
        'Alcohol_Special occasions only',
        'Alcohol_Never',
        'Alcohol_Unknown',
    ]

    work_cols = [
        'Work_In paid employment or self-employed',
        'Work_Retired',
        'Work_Looking after home and/or family',
        'Work_Unable to work because of sickness or disability',
        'Work_Unemployed',
        'Work_Doing unpaid or voluntary work',
        'Work_Full or part-time student',
        'Work_Unknown',
    ]

    education_cols = [
        'Education_College or University degree',
        'Education_A levels/AS levels or equivalent',
        'Education_O levels/GCSEs or equivalent',
        'Education_CSEs or equivalent',
        'Education_NVQ or HND or HNC or equivalent',
        'Education_Other professional qualifications eg: nursing, teaching',
        'Education_Unknown',
    ]

    sex_map = {
        0: 'Sex_Female',
        1: 'Sex_Male',
    }

    alcohol_map = {
        1: 'Alcohol_Daily or almost daily',
        2: 'Alcohol_Three or four times a week',
        3: 'Alcohol_Once or twice a week',
        4: 'Alcohol_One to three times a month',
        5: 'Alcohol_Special occasions only',
        6: 'Alcohol_Never',
    }

    work_map = {
        1: 'Work_In paid employment or self-employed',
        2: 'Work_Retired',
        3: 'Work_Looking after home and/or family',
        4: 'Work_Unable to work because of sickness or disability',
        5: 'Work_Unemployed',
        6: 'Work_Doing unpaid or voluntary work',
        7: 'Work_Full or part-time student',
    }

    education_map = {
        1: 'Education_College or University degree',
        2: 'Education_A levels/AS levels or equivalent',
        3: 'Education_O levels/GCSEs or equivalent',
        4: 'Education_CSEs or equivalent',
        5: 'Education_NVQ or HND or HNC or equivalent',
        6: 'Education_Other professional qualifications eg: nursing, teaching',
    }

    def _build_eid_index(df, df_name):
        if 'eid' not in df.columns:
            raise KeyError(f"{df_name} must contain a column named 'eid'.")

        out = df.copy()
        out['eid'] = pd.to_numeric(out['eid'], errors='coerce')
        out = out.dropna(subset=['eid'])
        out['eid'] = out['eid'].astype(int)
        out = out.drop_duplicates(subset=['eid'], keep='first')
        return out.set_index('eid', drop=False)

    def _eid_to_int(eid):
        try:
            return int(float(eid))
        except (TypeError, ValueError):
            return None

    def _get(row, col):
        if row is None or col not in row.index:
            return np.nan
        return row[col]

    def _safe_float(x):
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
            x = list(x)[0] if len(x) > 0 else np.nan

        if pd.isna(x):
            return np.nan

        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    def _parse_codes(x):
        """
        把 UKB 分类变量解析成 int code。
        兼容 1、1.0、'1.0'、'1,2'、'[1, 2]' 这种情况。
        """
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
            codes = []
            for item in x:
                codes.extend(_parse_codes(item))
            return codes

        if pd.isna(x):
            return []

        s = str(x).strip()
        if s == '' or s.lower() in {'nan', 'none', 'na'}:
            return []

        nums = re.findall(r'-?\d+(?:\.\d+)?', s)
        codes = []
        for num in nums:
            try:
                codes.append(int(float(num)))
            except ValueError:
                pass

        return codes

    def _first_code(x):
        codes = _parse_codes(x)
        return codes[0] if len(codes) > 0 else None

    def _init_onehot(row, cols):
        for col in cols:
            row[col] = 0

    def _set_single_onehot(row, cols, selected_col=None, unknown_col=None):
        _init_onehot(row, cols)

        if selected_col is not None:
            row[selected_col] = 1
        elif unknown_col is not None:
            row[unknown_col] = 1

    def _set_multi_onehot(row, cols, selected_cols, unknown_col=None):
        _init_onehot(row, cols)

        matched = False
        for selected_col in selected_cols:
            if selected_col in cols:
                row[selected_col] = 1
                matched = True

        if not matched and unknown_col is not None:
            row[unknown_col] = 1

    csv_idx = _build_eid_index(csv_data, 'csv_data')
    work_edu_idx = _build_eid_index(work_edu_csv, 'work_edu_csv')

    rows = []

    for eid in tqdm(eid_list):
        eid_int = _eid_to_int(eid)

        row = {col: np.nan for col in columns}
        row['Eid'] = eid

        data_row = None
        work_edu_row = None

        if eid_int is not None and eid_int in csv_idx.index:
            data_row = csv_idx.loc[eid_int]
        else:
            print(f'{eid} not in csv_data')

        if eid_int is not None and eid_int in work_edu_idx.index:
            work_edu_row = work_edu_idx.loc[eid_int]
        else:
            print(f'{eid} not in work_edu_csv')

        if data_row is not None:
            for source_col, target_col in continuous_map.items():
                row[target_col] = _safe_float(_get(data_row, source_col))

            sex = _first_code(_get(data_row, '31-0.0'))
            _set_single_onehot(
                row,
                sex_cols,
                selected_col=sex_map.get(sex),
                unknown_col=None
            )

            ethnic_raw = _get(data_row, '21000-0.0')
            if pd.isna(ethnic_raw):
                ethnic_col = 'Ethnic_Unknown'
            else:
                ethnic = str(ethnic_raw).strip()
                if ethnic.startswith('1'):
                    ethnic_col = 'Ethnic_White'
                elif ethnic.startswith('2'):
                    ethnic_col = 'Ethnic_Mixed'
                elif ethnic.startswith('3'):
                    ethnic_col = 'Ethnic_Asian or Asian British'
                elif ethnic.startswith('4'):
                    ethnic_col = 'Ethnic_Black or Black British'
                elif ethnic.startswith('5'):
                    ethnic_col = 'Ethnic_Chinese'
                elif ethnic.startswith('6'):
                    ethnic_col = 'Ethnic_Other ethnic group'
                else:
                    ethnic_col = 'Ethnic_Unknown'

            _set_single_onehot(
                row,
                ethnic_cols,
                selected_col=ethnic_col,
                unknown_col='Ethnic_Unknown'
            )

            smoke = _first_code(_get(data_row, '20160-2.0'))
            if smoke == 1:
                smoke_col = 'Ever_smoked_yes'
            elif smoke == 0:
                smoke_col = 'Ever_smoked_no'
            else:
                smoke_col = 'Ever_smoked_Unknown'

            _set_single_onehot(
                row,
                smoke_cols,
                selected_col=smoke_col,
                unknown_col='Ever_smoked_Unknown'
            )

            alcohol = _first_code(_get(data_row, '1558-2.0'))
            alcohol_col = alcohol_map.get(alcohol, 'Alcohol_Unknown')

            _set_single_onehot(
                row,
                alcohol_cols,
                selected_col=alcohol_col,
                unknown_col='Alcohol_Unknown'
            )

        if work_edu_row is not None:
            work_codes = _parse_codes(_get(work_edu_row, '6142-2.0'))
            work_selected_cols = [
                work_map[c] for c in work_codes if c in work_map
            ]

            _set_multi_onehot(
                row,
                work_cols,
                selected_cols=work_selected_cols,
                unknown_col='Work_Unknown'
            )

            education_codes = _parse_codes(_get(work_edu_row, '6138-2.0'))
            education_selected_cols = [
                education_map[c] for c in education_codes if c in education_map
            ]

            _set_multi_onehot(
                row,
                education_cols,
                selected_cols=education_selected_cols,
                unknown_col='Education_Unknown'
            )

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(save_name, index=False)

    print(f"处理完成：共输出 {len(df)} 个 eid 的信息，结果已保存到 {save_name}")

    return df








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





import ast
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def extact_MIMICinfo_presubject(eid_list, base_df, admission_df, save_name):
    """
    把 MIMIC ECG 对应的每个 eid 的人口学信息和疾病标签整理成一个表。

    输出：每个 eid 一行。
    - Age 保留原始数值
    - Sex / marital_status / race 转为 one-hot
    - 疾病标签根据 records_w_diag_icd10['all_diag_all'] 解析 ICD-10，输出 0/1
    """
    records_w_diag_icd10 = base_df
    disease_cols = [
        'cardiomyopathy_I42_I43_I255',
        'heart_failure_I50',
        'ischemic_heart_disease_I20_I25',
        'pulmonary_heart_vascular_I27',
        'pericardial_disease_I30_I32',
        'arrhythmia_I44_I49',
        'hypertension_I10_I15',
        'myocarditis_I40_I41',
    ]

    columns = [
        'Eid',
        'subject_id',
        'Age',
        'Sex_Female',
        'Sex_Male',
        'Sex_Unknown',
        'marital_status_WIDOWED',
        'marital_status_SINGLE',
        'marital_status_MARRIED',
        'marital_status_DIVORCED',
        'marital_status_Unknown',
        'Ethnic_WHITE',
        'Ethnic_BLACK',
        'Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
        'Ethnic_PORTUGUESE',
        'Ethnic_HISPANIC',
        'Ethnic_ASIAN',
        'Ethnic_SOUTH AMERICAN',
        'Ethnic_AMERICAN INDIAN',
        'Ethnic_MULTIPLE RACE',
        'Ethnic_Unknown',
    ] + disease_cols

    sex_cols = [
        'Sex_Female',
        'Sex_Male',
        'Sex_Unknown',
    ]

    marital_cols = [
        'marital_status_WIDOWED',
        'marital_status_SINGLE',
        'marital_status_MARRIED',
        'marital_status_DIVORCED',
        'marital_status_Unknown',
    ]

    ethnic_cols = [
        'Ethnic_WHITE',
        'Ethnic_BLACK',
        'Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
        'Ethnic_PORTUGUESE',
        'Ethnic_HISPANIC',
        'Ethnic_ASIAN',
        'Ethnic_SOUTH AMERICAN',
        'Ethnic_AMERICAN INDIAN',
        'Ethnic_MULTIPLE RACE',
        'Ethnic_Unknown',
    ]

    def _normalize_eid(x):
        x = str(x)
        x = x.replace('/mnt/sda1/lihaitao/datasets/ECG/', '')
        if x.endswith('.dat'):
            x = x[:-4]
        return x

    def _extract_subject_id(eid):
        eid = str(eid)
        parts = eid.split('/')

        try:
            candidate = parts[-3]
            if candidate.startswith('p'):
                candidate = candidate[1:]
            return int(candidate)
        except Exception:
            pass

        candidates = []
        for part in parts:
            if re.fullmatch(r'p\d+', part):
                candidates.append(int(part[1:]))

        if len(candidates) > 0:
            return max(candidates)

        return np.nan

    def _safe_float(x):
        if pd.isna(x):
            return np.nan
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    def _set_onehot(row, cols, selected_col):
        for col in cols:
            row[col] = 0

        if selected_col in cols:
            row[selected_col] = 1

    def _map_gender(gender):
        if pd.isna(gender):
            return 'Sex_Unknown'

        gender = str(gender).strip().upper()

        if gender == 'M':
            return 'Sex_Male'
        elif gender == 'F':
            return 'Sex_Female'
        else:
            return 'Sex_Unknown'

    def _map_marital_status(marital_status):
        if pd.isna(marital_status):
            return 'marital_status_Unknown'

        marital_status = str(marital_status).strip().upper()

        if marital_status == 'WIDOWED':
            return 'marital_status_WIDOWED'
        elif marital_status == 'SINGLE':
            return 'marital_status_SINGLE'
        elif marital_status == 'MARRIED':
            return 'marital_status_MARRIED'
        elif marital_status == 'DIVORCED':
            return 'marital_status_DIVORCED'
        else:
            return 'marital_status_Unknown'

    def _map_race(race):
        if pd.isna(race):
            return 'Ethnic_Unknown'

        race = str(race).strip().upper()

        if race == '':
            return 'Ethnic_Unknown'

        unknown_values = {
            'UNKNOWN',
            'UNABLE TO OBTAIN',
            'PATIENT DECLINED TO ANSWER',
            'OTHER',
        }

        if race in unknown_values:
            return 'Ethnic_Unknown'

        if race.startswith('WHITE'):
            return 'Ethnic_WHITE'
        elif race.startswith('BLACK'):
            return 'Ethnic_BLACK'
        elif race.startswith('NATIVE HAWAIIAN') or 'PACIFIC ISLANDER' in race:
            return 'Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'
        elif race.startswith('PORTUGUESE'):
            return 'Ethnic_PORTUGUESE'
        elif race.startswith('HISPANIC') or race.startswith('LATINO'):
            return 'Ethnic_HISPANIC'
        elif race.startswith('ASIAN'):
            return 'Ethnic_ASIAN'
        elif race.startswith('SOUTH AMERICAN'):
            return 'Ethnic_SOUTH AMERICAN'
        elif race.startswith('AMERICAN INDIAN') or 'ALASKA NATIVE' in race:
            return 'Ethnic_AMERICAN INDIAN'
        elif race.startswith('MULTIPLE RACE'):
            return 'Ethnic_MULTIPLE RACE'
        else:
            return 'Ethnic_Unknown'

    def _parse_diag_list(x):
        """
        兼容：
        1. 已经是 list
        2. 字符串形式的 list，例如 "['I42.0', 'I50.9']"
        3. 普通分隔符字符串，例如 "I42.0;I50.9"
        """
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
            return [str(i) for i in x if not pd.isna(i)]

        if pd.isna(x):
            return []

        s = str(x).strip()
        if s == '' or s.lower() in {'nan', 'none', 'null'}:
            return []

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
                return [str(i) for i in parsed if not pd.isna(i)]
            else:
                return [str(parsed)]
        except Exception:
            return [i.strip() for i in re.split(r'[;,|]', s) if i.strip() != '']

    def _normalize_icd10(code):
        code = str(code).strip().upper()
        code = code.replace('.', '')
        code = code.replace(' ', '')
        return code

    def _has_any_prefix(codes, prefixes):
        return any(
            any(code.startswith(prefix) for prefix in prefixes)
            for code in codes
        )

    def _map_disease_flags(all_diag):
        raw_codes = _parse_diag_list(all_diag)
        codes = [_normalize_icd10(code) for code in raw_codes]
        codes = [code for code in codes if code != '']

        flags = {
            'cardiomyopathy_I42_I43_I255': int(
                _has_any_prefix(codes, ['I42', 'I43']) or
                _has_any_prefix(codes, ['I255'])
            ),
            'heart_failure_I50': int(
                _has_any_prefix(codes, ['I50'])
            ),
            'ischemic_heart_disease_I20_I25': int(
                _has_any_prefix(codes, ['I20', 'I21', 'I22', 'I23', 'I24', 'I25'])
            ),
            'pulmonary_heart_vascular_I27': int(
                _has_any_prefix(codes, ['I27'])
            ),
            'pericardial_disease_I30_I32': int(
                _has_any_prefix(codes, ['I30', 'I31', 'I32'])
            ),
            'arrhythmia_I44_I49': int(
                _has_any_prefix(codes, ['I44', 'I45', 'I46', 'I47', 'I48', 'I49'])
            ),
            'hypertension_I10_I15': int(
                _has_any_prefix(codes, ['I10', 'I11', 'I12', 'I13', 'I14', 'I15'])
            ),
            'myocarditis_I40_I41': int(
                _has_any_prefix(codes, ['I40', 'I41'])
            ),
        }

        return flags

    eid_list = [_normalize_eid(i) for i in eid_list]

    if 'file_name' not in base_df.columns:
        raise KeyError("base_df 里必须有 'file_name' 这一列。")

    if 'subject_id' not in admission_df.columns:
        raise KeyError("admission_df 里必须有 'subject_id' 这一列。")

    if 'file_name' not in records_w_diag_icd10.columns:
        raise KeyError("records_w_diag_icd10 里必须有 'file_name' 这一列。")

    if 'all_diag_all' not in records_w_diag_icd10.columns:
        raise KeyError("records_w_diag_icd10 里必须有 'all_diag_all' 这一列。")

    base_df_indexed = base_df.copy()
    base_df_indexed['file_name_norm'] = base_df_indexed['file_name'].apply(_normalize_eid)
    base_df_indexed = base_df_indexed.drop_duplicates(
        subset='file_name_norm',
        keep='first'
    )
    base_df_indexed = base_df_indexed.set_index('file_name_norm', drop=False)

    admission_df_indexed = admission_df.copy()
    admission_df_indexed['subject_id'] = pd.to_numeric(
        admission_df_indexed['subject_id'],
        errors='coerce'
    )
    admission_df_indexed = admission_df_indexed.dropna(subset=['subject_id'])
    admission_df_indexed['subject_id'] = admission_df_indexed['subject_id'].astype(int)
    admission_df_indexed = admission_df_indexed.drop_duplicates(
        subset='subject_id',
        keep='first'
    )
    admission_df_indexed = admission_df_indexed.set_index('subject_id', drop=False)

    diag_df_indexed = records_w_diag_icd10.copy()
    diag_df_indexed['file_name_norm'] = diag_df_indexed['file_name'].apply(_normalize_eid)
    diag_df_indexed = diag_df_indexed.drop_duplicates(
        subset='file_name_norm',
        keep='first'
    )
    diag_df_indexed = diag_df_indexed.set_index('file_name_norm', drop=False)

    rows = []

    for eid in tqdm(eid_list):
        subject_id = _extract_subject_id(eid)

        row = {col: 0 for col in columns}
        row['Eid'] = eid
        row['subject_id'] = subject_id
        row['Age'] = np.nan

        _set_onehot(row, sex_cols, 'Sex_Unknown')
        _set_onehot(row, marital_cols, 'marital_status_Unknown')
        _set_onehot(row, ethnic_cols, 'Ethnic_Unknown')

        for col in disease_cols:
            row[col] = np.nan

        if eid in base_df_indexed.index:
            base_row = base_df_indexed.loc[eid]

            gender_col = _map_gender(base_row.get('gender', np.nan))
            _set_onehot(row, sex_cols, gender_col)

            row['Age'] = _safe_float(base_row.get('age', np.nan))
        else:
            print(f'{eid} not in base_df')

        if not pd.isna(subject_id) and int(subject_id) in admission_df_indexed.index:
            admission_row = admission_df_indexed.loc[int(subject_id)]

            marital_col = _map_marital_status(
                admission_row.get('marital_status', np.nan)
            )
            _set_onehot(row, marital_cols, marital_col)

            ethnic_col = _map_race(
                admission_row.get('race', np.nan)
            )
            _set_onehot(row, ethnic_cols, ethnic_col)
        else:
            pass
            # print(f'subject_id {subject_id} of {eid} not in admission_df')

        if eid in diag_df_indexed.index:
            diag_row = diag_df_indexed.loc[eid]
            disease_flags = _map_disease_flags(diag_row.get('all_diag_all', np.nan))

            for col, value in disease_flags.items():
                row[col] = value
        else:
            print(f'{eid} not in records_w_diag_icd10')

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(save_name, index=False)

    print(f"处理完成：共输出 {len(df)} 个 eid 的信息，结果已保存到 {save_name}")

    return df


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
