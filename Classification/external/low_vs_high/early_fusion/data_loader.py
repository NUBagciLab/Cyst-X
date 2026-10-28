"""Paired T1/T2 cases for leave-one-center-out external evaluation."""
import os
import numpy as np
import pandas as pd

DEFAULT_ROOT = "/dataset/Cyst-X-resampled/classification"
CENTER_NAMES = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]


def case_id(name):
    return str(name).strip().removesuffix('.gz').removesuffix('.nii')


def get_pair_table(root=DEFAULT_ROOT):
    table = pd.read_excel(os.path.join(root, 'IPMN_labels_total.xlsx'), usecols=[0, 1, 6])
    table.columns = ['ID', 'ID_T2', 'Risk Assessment']
    table = table.dropna(subset=['ID', 'ID_T2', 'Risk Assessment']).copy()
    table['ID'] = table['ID'].map(case_id)
    table['ID_T2'] = table['ID_T2'].map(case_id)
    if table['ID'].duplicated().any() or table['ID_T2'].duplicated().any():
        raise ValueError('Each T1 and T2 scan must occur in only one labeled pair.')
    if not table['Risk Assessment'].isin([0, 1, 2]).all():
        raise ValueError('Expected risk labels 0 (no), 1 (low), or 2 (high).')
    table = table[table['Risk Assessment'].isin([1, 2])].copy()
    return table


def get_data_list(root=DEFAULT_ROOT, center=None):
    table = get_pair_table(root)
    centers = range(len(CENTER_NAMES)) if center is None else ([center] if isinstance(center, int) else center)
    aliases = [alias for index in centers for alias in CENTER_NAMES[index]]
    image1, image2, labels = [], [], []
    for name1, name2, risk in table.itertuples(index=False, name=None):
        if not any(alias in name1 for alias in aliases):
            continue
        path1 = os.path.join(root, 't1', 'images', name1+'.nii.gz')
        path2 = os.path.join(root, 't2', 'images', name2+'.nii.gz')
        if not os.path.isfile(path1) or not os.path.isfile(path2):
            raise FileNotFoundError(f'Missing paired T1/T2 image: {path1} | {path2}')
        image1.append(path1)
        image2.append(path2)
        labels.append(np.array([risk - 1], dtype=np.float32))
    return image1, image2, labels

