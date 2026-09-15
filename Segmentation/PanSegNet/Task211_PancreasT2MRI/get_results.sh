#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/Segmentation_eval.py"
INTERNAL_DIR="${SCRIPT_DIR}/nnTransUNetTrainerV2__nnUNetPlansv2.1"
EXTERNAL_DIR="${SCRIPT_DIR}/nnTransUNetTrainerV2__nnUNetPlansv2.1_external_T2"
PRED_SUBDIR="validation_raw_postprocessed"
NUM_WORKERS="${NUM_WORKERS:-32}"
SKIP_EVAL=0
RUN_INTERNAL=0
RUN_EXTERNAL=0
GT_REL="gt_niftis"

usage() {
  cat <<EOF
Usage: $(basename "$0") --internal|--external [--skip-eval] [--gt DIR] [--workers N]

Evaluate fold_*/${PRED_SUBDIR} with Segmentation_eval.py,
then aggregate class_1 metrics by center into t2_result.csv.

Options:
  --internal    5-fold internal split: nnTransUNetTrainerV2__nnUNetPlansv2.1
  --external    leave-one-center-out: nnTransUNetTrainerV2__nnUNetPlansv2.1_external_T2
  --skip-eval   Reuse existing fold_*/*_metrics.csv files
  --gt DIR      GT folder relative to the experiment dir (default: gt_niftis)
  --workers N   Parallel workers for evaluation (default: ${NUM_WORKERS})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --internal) RUN_INTERNAL=1; shift ;;
    --external) RUN_EXTERNAL=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --gt) GT_REL="$2"; shift 2 ;;
    --workers) NUM_WORKERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${RUN_INTERNAL}" -eq 0 && "${RUN_EXTERNAL}" -eq 0 ]]; then
  echo "Specify --internal and/or --external." >&2
  usage
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "Missing eval script: ${EVAL_PY}" >&2
  exit 1
fi

run_one() {
  local exp_dir="$1"
  local tag="$2"
  local gt="${GT_REL}"

  if [[ ! -d "${exp_dir}" ]]; then
    echo "Experiment folder not found: ${exp_dir}" >&2
    exit 1
  fi

  mapfile -t fold_dirs < <(find "${exp_dir}" -maxdepth 1 -type d -name 'fold_*' | sort -V)
  if [[ ${#fold_dirs[@]} -eq 0 ]]; then
    echo "No fold_* directories under ${exp_dir}" >&2
    exit 1
  fi

  if [[ "${gt}" != /* ]]; then
    gt="${exp_dir}/${gt}"
  fi
  if [[ ! -d "${gt}" ]]; then
    echo "Ground-truth folder not found: ${gt}" >&2
    exit 1
  fi

  local out_csv="${exp_dir}/t2_result.csv"
  echo "===== ${tag} ====="
  echo "Exp dir    : ${exp_dir}"
  echo "Eval python: ${EVAL_PY}"
  echo "GT folder  : ${gt}"
  echo "Workers    : ${NUM_WORKERS}"
  echo "Output CSV : ${out_csv}"

  if [[ "${SKIP_EVAL}" -eq 0 ]]; then
    for fold_dir in "${fold_dirs[@]}"; do
      local pred_dir="${fold_dir}/${PRED_SUBDIR}"
      if [[ ! -d "${pred_dir}" ]]; then
        echo "Skip $(basename "${fold_dir}"): missing ${PRED_SUBDIR}" >&2
        continue
      fi
      echo "Evaluating $(basename "${fold_dir}") ..."
      python "${EVAL_PY}" \
        -s "${pred_dir}" \
        -g "${gt}" \
        -w "${NUM_WORKERS}"
    done
  else
    echo "Skip evaluation; using existing metrics CSVs."
  fi

  python - "${exp_dir}" "${out_csv}" <<'PY'
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

exp_dir, out_csv = sys.argv[1], sys.argv[2]
center_order = ["NYU", "MCF", "NU", "MCA", "EMC", "IU", "AHN", "Total"]
metric_cols = [
    ("Dice", "dice_class_1", True),
    ("Jaccard", "jaccard_class_1", True),
    ("Precision", "precision_class_1", True),
    ("Recall", "recall_class_1", True),
    ("Hausdorff Distance 95", "hd95_class_1", False),
    ("Avg. Symmetric Surface Distance", "assd_class_1", False),
]


def center_of(name):
    n = name.lower()
    if n.startswith("nyu"):
        return "NYU"
    if n.startswith("cad") or n.startswith("mcf"):
        return "MCF"
    if n.startswith("mca"):
        return "MCA"
    if n.startswith("emc"):
        return "EMC"
    if n.startswith("iu"):
        return "IU"
    if n.startswith("ahn"):
        return "AHN"
    if n.startswith("nu") or "northwestern" in n:
        return "NU"
    return "UNK"


def parse(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "inf", "-inf"}:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if math.isnan(number) or math.isinf(number):
        return np.nan
    return number


def fmt(mean, std):
    if np.isnan(mean):
        return "nan"
    if np.isnan(std):
        return f"{mean:.2f} ± nan"
    return f"{mean:.2f} ± {std:.2f}"


rows_by_center = defaultdict(list)
csv_paths = []
for name in sorted(os.listdir(exp_dir)):
    fold_dir = os.path.join(exp_dir, name)
    metrics_csv = os.path.join(fold_dir, "validation_raw_postprocessed_metrics.csv")
    if name.startswith("fold_") and os.path.isfile(metrics_csv):
        csv_paths.append(metrics_csv)

if not csv_paths:
    raise SystemExit(f"No validation_raw_postprocessed_metrics.csv under {exp_dir}")

for path in csv_paths:
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_center[center_of(row["name"])].append(row)

known = [c for c in center_order if c != "Total"]
all_rows = []
for center in known:
    all_rows.extend(rows_by_center.get(center, []))
rows_by_center["Total"] = all_rows

print("Cases per center:")
for center in center_order:
    print(f"  {center}: {len(rows_by_center.get(center, []))}")

unknown = rows_by_center.get("UNK", [])
if unknown:
    print(f"Warning: {len(unknown)} cases with unknown center")

header = ["center"] + [col for col, _, _ in metric_cols]
table = []
for center in center_order:
    rows = rows_by_center.get(center, [])
    if not rows:
        continue
    record = {"center": center}
    for label, key, percent in metric_cols:
        values = np.array([parse(row.get(key)) for row in rows], dtype=float)
        if key == "jaccard_class_1" and np.all(np.isnan(values)):
            values = np.array([parse(row.get("iou_class_1")) for row in rows], dtype=float)
        if percent:
            values = values * 100.0
        if values.size == 0 or np.all(np.isnan(values)):
            record[label] = "nan"
        else:
            record[label] = fmt(np.nanmean(values), np.nanstd(values))
    table.append(record)

with open(out_csv, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(table)

print(f"Saved {out_csv}")
metric_names = [col for col, _, _ in metric_cols]
print("  " + " | ".join(["center"] + metric_names))
for record in table:
    parts = [f"{record['center']:6s}"] + [record[name] for name in metric_names]
    print("  " + " | ".join(parts))
PY
}

if [[ "${RUN_INTERNAL}" -eq 1 ]]; then
  run_one "${INTERNAL_DIR}" "internal"
fi
if [[ "${RUN_EXTERNAL}" -eq 1 ]]; then
  run_one "${EXTERNAL_DIR}" "external"
fi
