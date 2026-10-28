#!/bin/bash
set -e
model="densenet121"
data_path="../../../../Cyst-X-resampled/classification"
output_dir="./saved"
while [[ $# -gt 0 ]]; do
    if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 1; fi
    case $1 in
        --model|-model) model="$2"; shift 2 ;;
        --data-path|-data_path|--data_path) data_path="$2"; shift 2 ;;
        --output-dir|-output-dir|-o) output_dir="$2"; shift 2 ;;
        *) echo "Usage: bash main.sh [--model name] [--data-path root] [--output-dir dir]" >&2; exit 1 ;;
    esac
done
for fold in {0..6}; do
    python train.py --model "$model" -f "$fold" -s 42 --data-path "$data_path" --output-dir "$output_dir" -j 4
done
python calibrate.py --model "$model" --data-path "$data_path" --output-dir "$output_dir"
python fold_test.py --model "$model" --data-path "$data_path" --output-dir "$output_dir"
python fold_test.py --model "$model" --data-path "$data_path" --output-dir "$output_dir" -c
