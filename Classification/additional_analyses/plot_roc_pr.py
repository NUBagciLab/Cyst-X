"""Figure 4: mean ROC and PR curves across the five internal cross-validation folds.

Panels (a)-(d): IPMN-only task, DenseNet-121 versus 3D radiomics.
Panels (e)-(h): broad binary task, centralized versus federated DenseNet-121.

Legends give fold-mean AUC and average precision (AP). Plotted PR curves average
interpolated precision across folds, whereas AP is computed per fold from the
step-wise curve and then averaged. Each panel is saved as its own PDF, using the
file names referenced by the manuscript (Figure/rocprc/).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common import DENSENET, FEDAVG, FEDPROX01, FEDPROX03, RADIOMICS, load_test

GRID = np.linspace(0, 1, 201)
PANELS = {
    'lvh': ('low_vs_high', [(DENSENET, 'DenseNet-121', '#0072B2'), (RADIOMICS, '3D Radiomics', '#E69F00')]),
    '': ('2-class', [(DENSENET, 'Centralized', '#0072B2'), (FEDAVG, 'FedAvg', '#E69F00'),
                     (FEDPROX01, 'FedProx (0.1)', '#009E73'), (FEDPROX03, 'FedProx (0.3)', '#CC79A7')]),
}


def mean_curve(frame, kind):
    curves, scores = [], []
    for _, fold in frame.groupby('group'):
        y, p = fold['Label'].to_numpy(), fold['Probability'].to_numpy()
        if kind == 'roc':
            fpr, tpr, _ = roc_curve(y, p)
            c = np.interp(GRID, fpr, tpr); c[0] = 0
            scores.append(roc_auc_score(y, p))
        else:
            precision, recall, _ = precision_recall_curve(y, p)
            order = np.argsort(recall); recall, precision = recall[order], precision[order]
            c = np.array([precision[recall >= g].max() for g in GRID])
            scores.append(average_precision_score(y, p))
        curves.append(c)
    return np.mean(curves, 0), np.mean(scores) * 100


def plot(root, output_dir):
    plt.rcParams.update({'font.size': 16, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    summary = []
    for prefix, (task, models) in PANELS.items():
        for modality in ['t1', 't2']:
            frames = [(load_test(root, 'internal', task, m[1], m[2], modality), label, color) for m, label, color in models]
            for kind in ['roc', 'prc']:
                fig, ax = plt.subplots(figsize=(6.2, 5.8))
                for frame, label, color in frames:
                    curve, score = mean_curve(frame, kind)
                    metric = 'AUC' if kind == 'roc' else 'AP'
                    ax.plot(GRID, curve, color=color, lw=2, label=f'{label} ({metric} = {score:.2f}%)')
                    summary.append((task, modality, kind, label, round(score, 2)))
                if kind == 'roc':
                    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.2, label='Chance')
                    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); loc = 'lower right'
                else:
                    prevalence = frames[0][0]['Label'].mean() * 100
                    ax.axhline(prevalence / 100, ls='--', color='gray', lw=1.2, label=f'Baseline ({prevalence:.2f}%)')
                    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); loc = 'upper right'
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(loc=loc, fontsize=13)
                fig.tight_layout()
                name = f"{prefix + '_' if prefix else ''}{modality}_{kind}_mean.pdf"
                fig.savefig(Path(output_dir) / name); plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in plot(args.classification_root, args.output_dir):
        print(*row)
