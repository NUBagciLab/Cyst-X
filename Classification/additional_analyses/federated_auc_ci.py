"""Section 2.4: pooled and center-averaged external AUC of federated versus centralized
DenseNet-121 (broad binary task), with paired bootstrap 95% CIs for the differences.

Patients are resampled within each center and true class; the same resampled
patients are used for every model, so the differences are paired.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DENSENET, FEDAVG, FEDPROX01, FEDPROX03, MODALITIES, auc, center_mean_auc,
                    load_test, stratified_resample)


def run(root, bootstrap, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for modality in ['t1', 't2']:
        configs = [DENSENET, FEDAVG, FEDPROX01, FEDPROX03]
        frames = {name: load_test(root, 'external', '2-class', method, model, modality).set_index('ID')
                  for name, method, model in configs}
        ids = sorted(set.intersection(*(set(f.index) for f in frames.values())))
        ref = frames[DENSENET[0]].loc[ids]
        y = ref['Label'].to_numpy(); c = ref['center'].to_numpy()
        probs = {name: f.loc[ids, 'Probability'].to_numpy() for name, f in frames.items()}
        # center-by-class strata in a fixed order (centers 0-6, then class 0/1)
        strata = [np.where((c == k) & (y == v))[0] for k in range(7) for v in (0, 1)]
        strata = [s for s in strata if len(s)]
        point = {n: (auc(y, p), center_mean_auc(y, p, c)) for n, p in probs.items()}
        boots = {n: [] for n in probs}
        for _ in range(bootstrap):
            i = stratified_resample(rng, strata)
            for n, p in probs.items():
                boots[n].append((auc(y[i], p[i]), center_mean_auc(y[i], p[i], c[i])))
        boots = {n: np.array(v) for n, v in boots.items()}
        for name in probs:
            row = dict(mri=MODALITIES[modality], model=name, n=len(ids),
                       pooled_auc=point[name][0] * 100, center_mean_auc=point[name][1] * 100)
            if name != DENSENET[0]:
                for j, label in [(0, 'pooled'), (1, 'center_mean')]:
                    diff = (boots[name][:, j] - boots[DENSENET[0]][:, j]) * 100
                    row[f'{label}_diff'] = (point[name][j] - point[DENSENET[0]][j]) * 100
                    row[f'{label}_diff_lower'], row[f'{label}_diff_upper'] = np.percentile(diff, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = run(args.classification_root, args.bootstrap, args.seed)
    table.to_csv(args.output_dir / 'federated_auc_ci.csv', index=False, float_format='%.4f')
    print(table.round(2).to_string(index=False))
