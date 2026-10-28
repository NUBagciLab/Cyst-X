"""Cohort alignment adapted from Hongyi's compare_results.py snapshot (2026-10-24).
Preserves ID normalization, reference labels, surveillance exclusion and reader alignment.
No training or threshold selection is performed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
def confusion_matrix(y_true, predictions, labels_order=None, **kwargs):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(predictions, dtype=int)
    return np.bincount(2*y+p, minlength=4).reshape(2, 2)


TASKS = ['2-class', 'low_vs_high', 'histology']
METHODS = [
    ('3D Radiomics', 'T1W Only', 'radiomics', 'random_forest/t1'),
    ('3D Radiomics', 'T2W Only', 'radiomics', 'random_forest/t2'),
    ('DenseNet-121', 'T1W Only', 'centralized', 'densenet121/t1'),
    ('DenseNet-121', 'T2W Only', 'centralized', 'densenet121/t2'),
    ('Early Fusion', 'T1W+T2W', 'early_fusion', 'densenet121'),
    ('Late Fusion', 'T1W+T2W', 'late_fusion', 'densenet121'),
    ('Siamese Fusion', 'T1W+T2W', 'siamese_fusion', 'densenet121'),
    ('Logit Fusion', 'T1W+T2W', 'logit_fusion', 'densenet121'),
]


def case_id(value):
    return str(value).strip().removesuffix('.gz').removesuffix('.nii')


def cohorts(labels_path, readers_path):
    reference = pd.read_excel(labels_path)
    aliases, metadata = {}, {}
    for _, row in reference.iterrows():
        if row[['Patient ID T1', 'Patient ID T2', 'risk assessment']].isna().any():
            continue
        canonical = case_id(row['Patient ID T1'])
        if canonical in metadata:
            raise ValueError(f'Duplicate reference patient: {canonical}')
        risk = int(row['risk assessment'])
        if risk not in (0, 1, 2):
            raise ValueError(f'Invalid reference risk: {canonical}')
        metadata[canonical] = (risk, 'no change' in str(row['follow-up information']).lower())
        for column in ['Patient ID T1', 'Patient ID T2']:
            alias = case_id(row[column])
            if alias in aliases and aliases[alias] != canonical:
                raise ValueError(f'Ambiguous patient alias: {alias}')
            aliases[alias] = canonical
    readers = pd.read_csv(readers_path)
    readers['canonical'] = readers['Patient ID'].map(case_id).map(aliases)
    if readers['canonical'].isna().any() or readers['canonical'].duplicated().any():
        raise ValueError('Unmapped or duplicate reader cases')
    readers = readers.set_index('canonical').sort_index()
    for patient, row in readers.iterrows():
        if metadata[patient][0] != row['Ground_truth_Risk_Assessment']:
            raise ValueError(f'Reader/reference ground truth mismatch: {patient}')
        if not row[['Ground_truth_Risk_Assessment', 'reader1', 'reader2', 'reader3']].isin([0, 1, 2]).all():
            raise ValueError(f'Invalid reader assessment: {patient}')
    groups = {}
    for task in TASKS:
        keep = [patient for patient in readers.index if task == '2-class' or
                metadata[patient][0] > 0 and (task != 'histology' or not metadata[patient][1])]
        groups[task] = readers.loc[keep]
    return aliases, groups


def metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError('Both risk classes are required')
    return dict(n=int(tn + fp + fn + tp), positive_n=int(tp + fn),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sens=float(tp / (tp + fn) * 100), spec=float(tn / (tn + fp) * 100))


def predictions(path, aliases, cohort):
    frame = pd.read_excel(path)
    if 'Held-out Center' in frame and 'Fold' not in frame:
        frame = frame.rename(columns={'Held-out Center': 'Fold'})
    required = ['ID', 'Label', 'Prediction', 'Probability', 'Fold']
    if not set(required).issubset(frame):
        raise ValueError(f'Missing prediction columns: {path}')
    frame['canonical'] = frame['ID'].map(case_id).map(aliases)
    frame = frame[frame['canonical'].isin(cohort.index)].copy()
    if frame['canonical'].duplicated().any():
        raise ValueError(f'Duplicate cohort predictions: {path}')
    missing = set(cohort.index) - set(frame['canonical'])
    if missing:
        raise ValueError(f'{len(missing)} cohort cases missing in {path}: {sorted(missing)[:5]}')
    frame = frame.set_index('canonical').loc[cohort.index]
    if frame[required].isna().any().any():
        raise ValueError(f'Missing prediction values: {path}')
    if not frame['Label'].isin([0, 1]).all() or not frame['Prediction'].isin([0, 1]).all():
        raise ValueError(f'Invalid binary values: {path}')
    truth = (cohort['Ground_truth_Risk_Assessment'] // 2).to_numpy()
    if not np.array_equal(frame['Label'].to_numpy(), truth):
        raise ValueError(f'Model/reader labels disagree: {path}')
    if not np.isfinite(frame['Probability']).all() or not frame['Probability'].between(0, 1).all():
        raise ValueError(f'Invalid probabilities: {path}')
    if path.name == 'result.xlsx' and not np.array_equal(
            frame['Prediction'].to_numpy(), (frame['Probability'] >= 0.5).astype(int).to_numpy()):
        raise ValueError(f'Uncalibrated predictions do not use threshold 0.50: {path}')
    return frame

