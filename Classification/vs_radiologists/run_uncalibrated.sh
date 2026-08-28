#!/usr/bin/env bash

# Set default base directory
DEFAULT_BASE_DIR="../results_calibration"
base_dir="$DEFAULT_BASE_DIR"
extra_flags=()

# Parse arguments: non-flag is treated as base_dir, flags are collected
while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--histology|-hist)
            extra_flags+=("$1")
            shift
            ;;
        -*)
            extra_flags+=("$1")
            shift
            ;;
        *)
            base_dir="$1"
            shift
            ;;
    esac
done

base_dir="${base_dir%/}"

echo "Using Base Directory: $base_dir"
if [ ${#extra_flags[@]} -gt 0 ]; then
    echo "Only evaluating on histology‑confirmed cases"
fi

# Run radiologist.py with flags
python radiologist.py "${extra_flags[@]}"

# Define tasks: "Display Label|Relative Path"
tasks=(
    "3D Radiomics Internal T1|Internal 2 Classes/3D Radiomics/t1.xlsx"
    "3D Radiomics Internal T2|Internal 2 Classes/3D Radiomics/t2.xlsx"
    "DenseNet-121 Internal T1|Internal 2 Classes/DenseNet-121/t1.xlsx"
    "DenseNet-121 Internal T2|Internal 2 Classes/DenseNet-121/t2.xlsx"
    "DenseNet-121 Internal Early Feature Concatenation|Internal 2 Classes/fusion_shared2/result.xlsx"
    "DenseNet-121 Internal Early Feature Addition|Internal 2 Classes/fusion_add_shared2/result.xlsx"
    "DenseNet-121 Internal Late Feature Concatenation|Internal 2 Classes/fusion2/result.xlsx"
    "DenseNet-121 Internal Late Feature Addition|Internal 2 Classes/fusion_add2/result.xlsx"
    "DenseNet-121 Internal Probability Fusion|Internal 2 Classes/fusion_prob/result.xlsx"
    "3D Radiomics External T1|External 2 Classes/3D Radiomics/t1.xlsx"
    "3D Radiomics External T2|External 2 Classes/3D Radiomics/t2.xlsx"
    "DenseNet-121 External T1|External 2 Classes/DenseNet-121/t1.xlsx"
    "DenseNet-121 External T2|External 2 Classes/DenseNet-121/t2.xlsx"
    "DenseNet-121 External Early Feature Concatenation|External 2 Classes/fusion_shared/result.xlsx"
    "DenseNet-121 External Early Feature Addition|External 2 Classes/fusion_add_shared/result.xlsx"
    "DenseNet-121 External Late Feature Concatenation|External 2 Classes/fusion/result.xlsx"
    "DenseNet-121 External Late Feature Addition|External 2 Classes/fusion_add/result.xlsx"
    "DenseNet-121 External Probability Fusion|External 2 Classes/fusion_prob/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"
    
    echo -n "$label "
    python classification.py -i "$base_dir/$rel_path" "${extra_flags[@]}"
done