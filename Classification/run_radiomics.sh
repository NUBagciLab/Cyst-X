#!/usr/bin/env bash
set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_path="$script_dir/../Cyst-X-resampled/classification"
t="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: bash run_radiomics.sh [--data-path root] [-t 1|2|all]"
            exit 0
            ;;
        --data-path|-t|--t)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            case $1 in
                --data-path) data_path="$2" ;;
                -t|--t) t="$2" ;;
            esac
            shift 2
            ;;
        *)
            echo "Usage: bash run_radiomics.sh [--data-path root] [-t 1|2|all]" >&2
            exit 1
            ;;
    esac
done

case "$t" in
    1|2|all) ;;
    *) echo "Modality must be 1, 2, or all." >&2; exit 1 ;;
esac

# Resolve a relative input path before entering each experiment folder.
data_path="$(cd -- "$data_path" && pwd)"

for evaluation in internal external; do
    for cohort in 2-class histology low_vs_high; do
        echo "Running radiomics: $evaluation / $cohort / t=$t"
        (
            cd -- "$script_dir/$evaluation/$cohort/radiomics"
            bash main.sh --data-path "$data_path" -t "$t"
        )
    done
done
