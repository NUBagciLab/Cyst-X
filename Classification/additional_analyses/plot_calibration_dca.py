"""Supplementary Fig. A2: decision curves and calibration curves for the IPMN-only
DenseNet-121 models (internal cross-validation and external leave-one-center-out).

(a) Net benefit of the recalibrated models versus treating all or no patients.
(b) Calibration curves before and after validation-based recalibration: mean
predicted probability and observed proportion of high-risk IPMN in eight
quantile bins; the dotted line indicates perfect calibration.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibration_dca import net_benefit, recalibrate
from common import DENSENET, load_test, load_validation

THRESHOLDS = np.linspace(0.05, 0.50, 46)
C_MODEL, C_RAW, C_ALL = '#1f5fa8', '#d9822b', '#8c8c8c'


def calibration_curve(y, p, bins=8):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return np.array([p[b == i].mean() for i in range(bins)]), np.array([y[b == i].mean() for i in range(bins)])


def plot(root, output):
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'font.size': 7.5, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.7))
    for j, (scope, modality) in enumerate([('internal', 't1'), ('internal', 't2'), ('external', 't1'), ('external', 't2')]):
        test = load_test(root, scope, 'low_vs_high', DENSENET[1], DENSENET[2], modality)
        val = load_validation(root, scope, 'low_vs_high', DENSENET[1], DENSENET[2], modality)
        d = recalibrate(test, val)
        y = d['Label'].to_numpy(); raw = d['Probability'].to_numpy(); rec = d['P_recal'].to_numpy()
        prevalence = y.mean()
        a = ax[0, j]
        a.plot(THRESHOLDS, [net_benefit(y, rec, t) for t in THRESHOLDS], color=C_MODEL, lw=1.2, label='DenseNet-121 (recalibrated)')
        a.plot(THRESHOLDS, prevalence - (1 - prevalence) * THRESHOLDS / (1 - THRESHOLDS), color=C_ALL, lw=1.0, label='Treat all')
        a.axhline(0, color='black', lw=0.8, ls='--', label='Treat none')
        a.set_ylim(-0.05, 0.32); a.set_xlim(0.05, 0.50); a.set_xlabel('Threshold probability')
        a.set_title(f"{scope.capitalize()} · {modality.upper()}W", fontsize=8)
        if j == 0:
            a.set_ylabel('Net benefit')
        b = ax[1, j]
        b.plot([0, 1], [0, 1], color='black', lw=0.7, ls=':')
        for p, color, label in [(raw, C_RAW, 'Before recalibration'), (rec, C_MODEL, 'After recalibration')]:
            xm, ym = calibration_curve(y, p)
            b.plot(xm, ym, 'o-', color=color, ms=2.8, lw=1.0, label=label)
        b.set_xlim(0, 1); b.set_ylim(0, 1); b.set_aspect('equal'); b.set_xlabel('Predicted probability')
        if j == 0:
            b.set_ylabel('Observed proportion')
    for a in ax.flat:
        a.spines[['top', 'right']].set_visible(False)
    for i, letter in enumerate('ab'):
        ax[i, 0].text(-0.32, 1.08, letter, transform=ax[i, 0].transAxes, fontsize=10, fontweight='bold')
    h0, l0 = ax[0, 0].get_legend_handles_labels(); h1, l1 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2, h_pad=1.0)
    fig.savefig(output.with_suffix('.pdf')); fig.savefig(output.with_suffix('.png'), dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(args.classification_root, args.output_dir / 'figA2_calibration_dca')
