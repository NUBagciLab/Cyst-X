"""Paired T1/T2 cases using the T1 five-fold assignments for both inputs."""
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
    return keep_histology(root, table)


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


def get_fold(image1, image2, label, root=DEFAULT_ROOT, fold=0, split_path=None):
    # Each pair follows its T1 ID: keep both inputs in the same partition, ignoring T2 folds.
    path = split_path or os.path.join(root, 't1_split.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Provide the T1 split CSV at {path}, or use --split-path. '
                                'Required columns: ID (T1 ID), Fold (0 through 4).')
    table = pd.read_csv(path, dtype={'ID': str})
    if not {'ID', 'Fold'}.issubset(table.columns):
        raise ValueError('Paired split CSV must contain ID (T1 ID) and Fold.')
    if table[['ID', 'Fold']].isna().any().any():
        raise ValueError('Split CSV has missing IDs or assignments.')
    table['ID'] = table['ID'].map(case_id)
    if table['ID'].duplicated().any():
        raise ValueError('Paired split CSV has duplicate IDs or missing assignments.')
    if not table['Fold'].isin(range(5)).all() or set(table['Fold']) != set(range(5)):
        raise ValueError('Expected five paired fold assignments numbered 0 through 4.')
    assignment = dict(zip(table['ID'], table['Fold']))
    train1, train2, train_label, test1, test2, test_label = [], [], [], [], [], []
    excluded = 0
    for path1, path2, target in zip(image1, image2, label):
        name = case_id(os.path.basename(path1))
        if name not in assignment:
            excluded += 1
            continue
        destination = (test1, test2, test_label) if assignment[name] == fold else (train1, train2, train_label)
        for values, value in zip(destination, (path1, path2, target)):
            values.append(value)
    if excluded:
        print(f'Paired split excludes {excluded} labeled pairs from this center.')
    return train1, train2, train_label, test1, test2, test_label


def keep_histology(root, table):
    """Match the existing cohort rule: exclude IDs with 'no change' follow-up."""
    reference = pd.read_excel(os.path.join(root, 'IPMN_labels_total.xlsx'))
    followup = reference[reference['follow-up information'].str.contains('no change', na=False)]
    identifiers = pd.concat([followup['Patient ID T1'], followup['Patient ID T2']]).dropna()
    excluded = {case_id(os.path.basename(str(name))) for name in identifiers}
    return table[~table['ID'].isin(excluded) & ~table['ID_T2'].isin(excluded)].copy()
