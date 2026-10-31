"""Check second-round revision outputs against the accepted reference results.

Usage: python verify_round2.py <output-dir>
Compares every numeric column of federated_auc_ci.csv, center_adjusted_or.csv and
calibration_dca.csv with reference_results/round2/ and checks the values quoted in
the manuscript.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-6
FILES = ['federated_auc_ci.csv', 'center_adjusted_or.csv', 'calibration_dca.csv']

# Values reported in the manuscript (rounded as printed).
MANUSCRIPT = {
    'center_adjusted_or.csv': [  # Section 2.2.1 and Limitations
        (dict(scope='internal', mri='T1W'), (2.31, 1.79, 2.98)),
        (dict(scope='internal', mri='T2W'), (1.91, 1.49, 2.45)),
        (dict(scope='external', mri='T1W'), (1.85, 1.44, 2.37)),
        (dict(scope='external', mri='T2W'), (2.08, 1.60, 2.71)),
    ],
}


def compare(output, reference):
    out, ref = pd.read_csv(output), pd.read_csv(reference)
    if out.shape != ref.shape or list(out.columns) != list(ref.columns):
        return f'shape or columns differ: {out.shape} vs {ref.shape}'
    numeric = ref.select_dtypes('number').columns
    text = [c for c in ref.columns if c not in numeric]
    if not out[text].equals(ref[text]):
        return 'text columns differ'
    gap = np.nanmax(np.abs(out[numeric].to_numpy(float) - ref[numeric].to_numpy(float)))
    if not (np.isfinite(gap) and gap <= TOLERANCE):
        return f'max absolute difference {gap:.3g}'
    return None


def main(output_dir):
    output_dir = Path(output_dir); failures = 0
    for name in FILES:
        problem = compare(output_dir / name, HERE / 'reference_results' / 'round2' / name)
        print(f"{'OK  ' if problem is None else 'FAIL'} {name}" + ('' if problem is None else f': {problem}'))
        failures += problem is not None
    table = pd.read_csv(output_dir / 'center_adjusted_or.csv')
    for selector, expected in MANUSCRIPT['center_adjusted_or.csv']:
        row = table[(table.task == 'low_vs_high') & (table.scope == selector['scope']) & (table.mri == selector['mri'])].iloc[0]
        got = tuple(round(float(row[c]), 2) for c in ['odds_ratio_per_sd', 'lower', 'upper'])
        ok = got == expected
        print(f"{'OK  ' if ok else 'FAIL'} manuscript OR {selector['scope']} {selector['mri']}: {got} (reported {expected})")
        failures += not ok
    print('All checks passed.' if not failures else f'{failures} check(s) failed.')
    return failures


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(1 if main(sys.argv[1]) else 0)
