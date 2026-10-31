#!/usr/bin/env bash
# Reproduce the second-round revision analyses from saved predictions (CPU only).
#   bash Classification/additional_analyses/run_round2.sh [classification-root] [output-dir]
# Defaults: ./Classification and ./results/round2 (run from the repository root).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-./Classification}"
OUT="${2:-./results/round2}"
mkdir -p "$OUT"
ROOT="$(cd "$ROOT" && pwd)"
OUT="$(cd "$OUT" && pwd)"
cd "$HERE"

python federated_auc_ci.py   --classification-root "$ROOT" --output-dir "$OUT"   # Section 2.4
python center_adjusted_or.py --classification-root "$ROOT" --output-dir "$OUT"   # Section 2.2.1, Limitations
python calibration_dca.py    --classification-root "$ROOT" --output-dir "$OUT"   # Supplementary Tables A19-A20
python plot_calibration_dca.py --classification-root "$ROOT" --output-dir "$OUT" # Supplementary Fig. A2
python plot_roc_pr.py        --classification-root "$ROOT" --output-dir "$OUT/figure4"  # Figure 4
python verify_round2.py "$OUT"
