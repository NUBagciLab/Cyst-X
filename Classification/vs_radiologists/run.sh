#!/usr/bin/env bash

# Set default base directory if not passed as an argument
DEFAULT_BASE_DIR="../results_calibration"
base_dir="${1:-$DEFAULT_BASE_DIR}"
base_dir="${base_dir%/}"

echo "Using Base Directory: $base_dir"

python radiologist.py

# Define tasks: "Display Label|Relative Path"
tasks=(
    "3D Radiomics Internal T1|Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx"
    "3D Radiomics Internal T2|Internal 2 Classes Calibrated/3D Radiomics/t2.xlsx"
    "DenseNet-121 Internal T1|Internal 2 Classes Calibrated/DenseNet-121/t1.xlsx"
    "DenseNet-121 Internal T2|Internal 2 Classes Calibrated/DenseNet-121/t2.xlsx"
    "DenseNet-121 Internal Early Feature Concatenation|Internal 2 Classes Calibrated/fusion_shared2/result.xlsx"
    "DenseNet-121 Internal Early Feature Addition|Internal 2 Classes Calibrated/fusion_add_shared2/result.xlsx"
    "DenseNet-121 Internal Late Feature Concatenation|Internal 2 Classes Calibrated/fusion2/result.xlsx"
    "DenseNet-121 Internal Late Feature Addition|Internal 2 Classes Calibrated/fusion_add2/result.xlsx"
    "DenseNet-121 Internal Probaility Fusion|Internal 2 Classes Calibrated/fusion_prob/result.xlsx"
    "3D Radiomics External T1|External 2 Classes Calibrated/3D Radiomics/t1.xlsx"
    "3D Radiomics External T2|External 2 Classes Calibrated/3D Radiomics/t2.xlsx"
    "DenseNet-121 External T1|External 2 Classes Calibrated/DenseNet-121/t1.xlsx"
    "DenseNet-121 External T2|External 2 Classes Calibrated/DenseNet-121/t2.xlsx"
    "DenseNet-121 External Late Fusion|External 2 Classes Calibrated/fusion/result.xlsx"
    "DenseNet-121 External Early Feature Concatenation|External 2 Classes Calibrated/fusion_shared2/result.xlsx"
    "DenseNet-121 External Early Feature Addition|External 2 Classes Calibrated/fusion_add_shared2/result.xlsx"
    "DenseNet-121 External Late Feature Concatenation|External 2 Classes Calibrated/fusion2/result.xlsx"
    "DenseNet-121 External Late Feature Addition|External 2 Classes Calibrated/fusion_add2/result.xlsx"
    "DenseNet-121 External Probaility Fusion|External 2 Classes Calibrated/fusion_prob/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"
    
    echo -n "$label "
    python classification.py -i "$base_dir/$rel_path"
done