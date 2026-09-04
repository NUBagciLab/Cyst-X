#!/usr/bin/env bash

# Set default base directory
DEFAULT_BASE_DIR="../results_calibration"
base_dir="$DEFAULT_BASE_DIR"
extra_flags=()

# Parse arguments: non-flag is treated as base_dir, flags are collected
while [[ $# -gt 0 ]]; do
    case "$1" in
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

# Run radiologist.py with flags
python radiologist.py "${extra_flags[@]}" -H

# Define tasks: "Display Label|Relative Path"
tasks=(
    "3D Radiomics Internal T1|Internal 2 Classes/3D Radiomics_histology/t1.xlsx"
    "3D Radiomics Internal T2|Internal 2 Classes/3D Radiomics_histology/t2.xlsx"
    "DenseNet-121 Internal T1|Internal 2 Classes/DenseNet-121_histology/t1.xlsx"
    "DenseNet-121 Internal T2|Internal 2 Classes/DenseNet-121_histology/t2.xlsx"
    "DenseNet-121 Internal Early Feature Concatenation|Internal 2 Classes/fusion_shared2_histology/result.xlsx"
    "DenseNet-121 Internal Late Feature Concatenation|Internal 2 Classes/fusion2_histology/result.xlsx"
    "3D Radiomics External T1|External 2 Classes/3D Radiomics_histology/t1.xlsx"
    "3D Radiomics External T2|External 2 Classes/3D Radiomics_histology/t2.xlsx"
    "DenseNet-121 External T1|External 2 Classes/DenseNet-121_histology/t1.xlsx"
    "DenseNet-121 External T2|External 2 Classes/DenseNet-121_histology/t2.xlsx"
    "DenseNet-121 External Early Feature Concatenation|External 2 Classes/fusion_shared_histology/result.xlsx"
    "DenseNet-121 External Late Feature Concatenation|External 2 Classes/fusion_histology/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"
    
    echo -n "$label "
    python classification.py -i "$base_dir/$rel_path" "${extra_flags[@]}"
done