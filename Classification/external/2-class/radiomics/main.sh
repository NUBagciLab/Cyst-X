#!/bin/bash
set -e

data_path="../../../../Cyst-X-resampled/classification"
t="all"
while [[ $# -gt 0 ]]; do
    if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 1; fi
    case $1 in
        --data-path|-data_path|--data_path) data_path="$2"; shift 2 ;;
        -t|--t) t="$2"; shift 2 ;;
        *) echo "Usage: bash main.sh [--data-path root] [-t 1|2|all]" >&2; exit 1 ;;
    esac
done
case "$t" in
    1|2) t_list=("$t") ;;
    all) t_list=(1 2) ;;
    *) echo "Modality must be 1, 2, or all." >&2; exit 1 ;;
esac
for current_t in "${t_list[@]}"; do
    for fold in {0..6}; do
        python train.py --data-path "$data_path" -t "$current_t" -f "$fold"
    done
    python calibrate.py -t "$current_t"
    python fold_test.py --data-path "$data_path" -t "$current_t"
    python fold_test.py --data-path "$data_path" -t "$current_t" -c
done
