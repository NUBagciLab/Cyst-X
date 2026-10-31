"""Supplementary Tables A19 and A20: calibration and decision-curve analysis.

For every evaluated model, pooled held-out predictions are summarized by the
calibration slope (logistic regression of the outcome on the logit of the
predicted probability; ideal 1), the calibration-in-the-large intercept (logit
used as an offset; ideal 0), the Brier score and the pooled AUC.

Each fold-specific (internal) or held-out-center-specific (external) model is
then recalibrated by logistic regression on the logit of its OWN validation
predictions (``val_result.xlsx``), and the fitted transformation is applied to
that model's test predictions. Test labels and held-out-center data are never
used for fitting. Net benefit of the recalibrated models is compared with
treating all or no patients at threshold probabilities of 0.10-0.40.

95% CIs: 1,000 patient-level bootstrap replicates stratified by true class.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from common import (DENSENET, FEDAVG, FEDPROX01, FEDPROX03, MODALITIES, RADIOMICS, auc,
                    load_test, load_validation)

EPS = 1e-6
THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
# Order matters: the bootstrap random stream is shared across configurations.
CONFIGS = [('low_vs_high', DENSENET), ('low_vs_high', RADIOMICS), ('2-class', DENSENET),
           ('2-class', FEDAVG), ('2-class', FEDPROX01), ('2-class', FEDPROX03)]
TASK_NAMES = {'low_vs_high': 'IPMN-only', '2-class': 'Broad'}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope(y, p):
    model = LogisticRegression(C=1e8, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(model.coef_[0, 0])


def calibration_intercept(y, p):
    """Calibration-in-the-large: intercept with the logit as an offset (slope fixed at 1)."""
    lg, a = logit(p), 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(a + lg)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return float(a)


def net_benefit(y, p, t):
    n = len(y); positive = p >= t
    tp = (positive & (y == 1)).sum(); fp = (positive & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def recalibrate(test, val):
    """Logistic recalibration of each model on its own validation predictions."""
    out = test.copy(); out['P_recal'] = np.nan
    for group, v in val.groupby('group'):
        model = LogisticRegression(C=1e8, max_iter=1000).fit(logit(v['Probability'].to_numpy()).reshape(-1, 1),
                                                           v['Label'].to_numpy())
        idx = out['group'] == group
        if not idx.any():
            raise ValueError(f'No test predictions for validation group {group}')
        out.loc[idx, 'P_recal'] = model.predict_proba(logit(out.loc[idx, 'Probability'].to_numpy()).reshape(-1, 1))[:, 1]
    if out['P_recal'].isna().any():
        raise ValueError('Some test predictions have no matching validation model')
    return out


def summarize(y, p):
    row = dict(auc=auc(y, p) * 100, brier=brier_score_loss(y, p),
               intercept=calibration_intercept(y, p), slope=calibration_slope(y, p))
    for t in THRESHOLDS:
        row[f'nb_{int(t * 100)}'] = net_benefit(y, p, t)
    return row


def bootstrap(rng, y, raw, rec, replicates):
    strata = [np.where(y == v)[0] for v in (0, 1)]
    keys = ['intercept_raw', 'slope_raw', 'intercept_rec', 'slope_rec'] + [f'dnb_{int(t * 100)}' for t in THRESHOLDS]
    res = {k: [] for k in keys}
    for _ in range(replicates):
        i = np.concatenate([rng.choice(s, len(s)) for s in strata])
        yy = y[i]
        res['intercept_raw'].append(calibration_intercept(yy, raw[i])); res['slope_raw'].append(calibration_slope(yy, raw[i]))
        res['intercept_rec'].append(calibration_intercept(yy, rec[i])); res['slope_rec'].append(calibration_slope(yy, rec[i]))
        prevalence = yy.mean()
        for t in THRESHOLDS:  # recalibrated model minus the better of treat-all / treat-none
            reference = max(prevalence - (1 - prevalence) * t / (1 - t), 0.0)
            res[f'dnb_{int(t * 100)}'].append(net_benefit(yy, rec[i], t) - reference)
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in res.items()}


def run(root, replicates, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for scope in ['internal', 'external']:
        for task, (name, method, model) in CONFIGS:
            for modality in ['t1', 't2']:
                test = load_test(root, scope, task, method, model, modality)
                val = load_validation(root, scope, task, method, model, modality)
                d = recalibrate(test, val)
                y = d['Label'].to_numpy(); raw = d['Probability'].to_numpy(); rec = d['P_recal'].to_numpy()
                a, b = summarize(y, raw), summarize(y, rec)
                ci = bootstrap(rng, y, raw, rec, replicates)
                prevalence = y.mean()
                row = dict(setting='Int.' if scope == 'internal' else 'Ext.', task=TASK_NAMES[task], model=name,
                           mri=MODALITIES[modality], n=len(y), prevalence=prevalence,
                           slope_raw=a['slope'], slope_raw_lo=ci['slope_raw'][0], slope_raw_hi=ci['slope_raw'][1],
                           intercept_raw=a['intercept'], intercept_raw_lo=ci['intercept_raw'][0], intercept_raw_hi=ci['intercept_raw'][1],
                           slope_rec=b['slope'], slope_rec_lo=ci['slope_rec'][0], slope_rec_hi=ci['slope_rec'][1],
                           intercept_rec=b['intercept'], intercept_rec_lo=ci['intercept_rec'][0], intercept_rec_hi=ci['intercept_rec'][1],
                           brier_raw=a['brier'], brier_rec=b['brier'], brier_reference=prevalence * (1 - prevalence),
                           auc_raw=a['auc'], auc_rec=b['auc'])
                for t in THRESHOLDS:
                    k = int(t * 100)
                    row[f'nb_{k}'] = b[f'nb_{k}']
                    row[f'nb_{k}_treat_all'] = prevalence - (1 - prevalence) * t / (1 - t)
                    row[f'dnb_{k}_lo'], row[f'dnb_{k}_hi'] = ci[f'dnb_{k}']
                rows.append(row)
                print(f"{row['setting']} {row['task']:9s} {name:13s} {row['mri']}  slope {a['slope']:.2f} -> {b['slope']:.2f}"
                      f"  AUC {a['auc']:.2f} -> {b['auc']:.2f}", flush=True)
    return pd.DataFrame(rows)


def latex_tables(table):
    """Rows of Supplementary Tables A19 (calibration) and A20 (net benefit)."""
    m = lambda x: f'{x:.2f}'.replace('-', '$-$')
    ci = lambda r, k: f"{m(r[k])} [{m(r[k + '_lo'])}, {m(r[k + '_hi'])}]"
    a19, a20 = [], []
    for _, r in table.iterrows():
        base = f"{r.setting} & {r.task} & {r.model} & {r.mri}"
        a19.append(base + f" & {ci(r, 'slope_raw')} & {ci(r, 'intercept_raw')} & {ci(r, 'slope_rec')} & "
                          f"{ci(r, 'intercept_rec')} & {r.brier_raw:.3f} / {r.brier_rec:.3f} ({r.brier_reference:.3f}) & "
                          f"{r.auc_raw:.2f} / {r.auc_rec:.2f}\\\\")
        cells = []
        for t in THRESHOLDS:
            k = int(t * 100)
            star = '$^{*}$' if r[f'dnb_{k}_lo'] > 0 else ''
            cells.append(f"{r[f'nb_{k}']:.3f}{star} ({r[f'nb_{k}_treat_all']:.3f})".replace('-', '$-$'))
        a20.append(base + ' & ' + ' & '.join(cells) + '\\\\')
    return '\n'.join(a19), '\n'.join(a20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = run(args.classification_root, args.bootstrap, args.seed)
    table.to_csv(args.output_dir / 'calibration_dca.csv', index=False, float_format='%.6f')
    a19, a20 = latex_tables(table)
    (args.output_dir / 'calibration_rows.tex').write_text(a19 + '\n')
    (args.output_dir / 'net_benefit_rows.tex').write_text(a20 + '\n')
