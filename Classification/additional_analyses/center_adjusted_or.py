"""Section 2.2.1 and Limitations: association of the DenseNet-121 score with high-risk
IPMN after adjustment for center.

Logistic regression of the outcome on center (fixed effect) and the standardized
model score. Scores are rank-normalized within each cross-validation fold
(internal) or across all held-out predictions (external) before standardization.
Odds ratios are per standard deviation of the score, with Wald 95% CIs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DENSENET, MODALITIES, load_test


def logistic_fit(X, y, iterations=100):
    """Maximum-likelihood logistic regression (Newton-Raphson); returns beta and covariance."""
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    covariance = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return beta, covariance


def adjusted_or(frame, scope):
    if scope == 'internal':
        rank = frame.groupby('group')['Probability'].rank(pct=True)
    else:
        rank = frame['Probability'].rank(pct=True)
    z = ((rank - rank.mean()) / rank.std()).to_numpy()  # sample standard deviation
    centers = pd.get_dummies(frame['center'], prefix='c', drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(frame)), centers.to_numpy(), z])
    beta, cov = logistic_fit(X, frame['Label'].to_numpy().astype(float))
    se = np.sqrt(cov[-1, -1])
    return np.exp(beta[-1]), np.exp(beta[-1] - 1.959964 * se), np.exp(beta[-1] + 1.959964 * se)


def run(root, tasks):
    rows = []
    for task in tasks:
        for scope in ['internal', 'external']:
            for modality in ['t1', 't2']:
                frame = load_test(root, scope, task, DENSENET[1], DENSENET[2], modality)
                odds, lower, upper = adjusted_or(frame, scope)
                rows.append(dict(task=task, scope=scope, mri=MODALITIES[modality], n=len(frame),
                                 odds_ratio_per_sd=odds, lower=lower, upper=upper))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['low_vs_high'],
                        help="tasks to analyse; the manuscript reports 'low_vs_high' (add '2-class' for the broad task)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = run(args.classification_root, args.tasks)
    table.to_csv(args.output_dir / 'center_adjusted_or.csv', index=False, float_format='%.4f')
    print(table.round(2).to_string(index=False))
