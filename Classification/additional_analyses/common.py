"""Shared helpers for the second-round revision analyses.

All functions read saved held-out predictions (``result.xlsx``) and validation
predictions (``val_result.xlsx``) written by the experiment ``fold_test.py`` and
``fold_test_val.py`` scripts. Nothing is retrained.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Same center keys and order as the experiment data loaders.
CENTER_KEYS = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
CENTER_NAMES = ['NYU', 'MCF', 'NU', 'AHN', 'MCA', 'IU', 'EMC']

# (label, method directory, model directory) for each evaluated configuration.
DENSENET = ('DenseNet-121', 'centralized', 'densenet121')
RADIOMICS = ('Radiomics', 'radiomics', 'random_forest')
FEDAVG = ('FedAvg', 'fedavg', 'densenet121')
FEDPROX01 = ('FedProx (0.1)', 'fedprox', 'densenet121_0.1')
FEDPROX03 = ('FedProx (0.3)', 'fedprox', 'densenet121_0.3')
MODALITIES = {'t1': 'T1W', 't2': 'T2W'}


def center_index(case_id):
    """Map a case ID to its center index, using the data-loader rule."""
    for k, keys in enumerate(CENTER_KEYS):
        if any(key in str(case_id) for key in keys):
            return k
    raise ValueError(f'Unknown center for case {case_id}')


def experiment_dir(root, scope, task, method, model, modality):
    return Path(root) / scope / task / method / 'saved' / model / modality


def load_test(root, scope, task, method, model, modality):
    """Held-out test predictions with a 'center' column and a 'group' column.

    'group' identifies the model that scored each case: the fold (internal) or
    the held-out center (external).
    """
    path = experiment_dir(root, scope, task, method, model, modality) / 'result.xlsx'
    frame = pd.read_excel(path)
    key = 'Fold' if scope == 'internal' else 'Held-out Center'
    frame['group'] = frame[key]
    frame['center'] = frame['ID'].map(center_index)
    frame['Label'] = frame['Label'].astype(int)
    return frame


def load_validation(root, scope, task, method, model, modality):
    """Validation predictions of every fold or held-out-center model, stacked."""
    path = experiment_dir(root, scope, task, method, model, modality) / 'val_result.xlsx'
    sheets = pd.read_excel(path, sheet_name=None)
    frame = pd.concat(sheets.values(), ignore_index=True)
    key = 'Fold' if scope == 'internal' else 'Held-out Center'
    frame['group'] = frame[key]
    frame['Label'] = frame['Label'].astype(int)
    return frame


def auc(y, p):
    """Mann-Whitney AUC with half credit for ties."""
    y = np.asarray(y); p = np.asarray(p)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if not n0 or not n1:
        raise ValueError('AUC requires both classes')
    ranks = pd.Series(p).rank(method='average').to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def center_mean_auc(y, p, c):
    """Unweighted mean of the center-specific AUCs (centers with both classes)."""
    return np.mean([auc(y[c == k], p[c == k]) for k in np.unique(c) if len(np.unique(y[c == k])) == 2])


def stratified_resample(rng, strata):
    """Indices of one bootstrap replicate, resampling within each stratum."""
    return np.concatenate([rng.choice(idx, len(idx)) for idx in strata])

