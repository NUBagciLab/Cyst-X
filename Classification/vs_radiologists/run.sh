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
echo "Radiologist Results"

# Run radiologist.py with flags
python radiologist.py "${extra_flags[@]}"

# Define tasks: "Display Label|Relative Path"
tasks=(
    "3D Radiomics Internal T1|Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx"
    "3D Radiomics Internal T2|Internal 2 Classes Calibrated/3D Radiomics/t2.xlsx"
    "DenseNet-121 Internal T1|Internal 2 Classes Calibrated/DenseNet-121/t1.xlsx"
    "DenseNet-121 Internal T2|Internal 2 Classes Calibrated/DenseNet-121/t2.xlsx"
    "DenseNet-121 Internal Early Fusion|Internal 2 Classes Calibrated/early_fusion/result.xlsx"
    "DenseNet-121 Internal Late Fusion|Internal 2 Classes Calibrated/late_fusion/result.xlsx"
    "DenseNet-121 Internal Siamese Fusion|Internal 2 Classes Calibrated/siamese_fusion/result.xlsx"
    "DenseNet-121 Internal Logit Fusion|Internal 2 Classes Calibrated/logit_fusion/result.xlsx"
    "3D Radiomics External T1|External 2 Classes Calibrated/3D Radiomics/t1.xlsx"
    "3D Radiomics External T2|External 2 Classes Calibrated/3D Radiomics/t2.xlsx"
    "DenseNet-121 External T1|External 2 Classes Calibrated/DenseNet-121/t1.xlsx"
    "DenseNet-121 External T2|External 2 Classes Calibrated/DenseNet-121/t2.xlsx"
    "DenseNet-121 External Early Fusion|External 2 Classes Calibrated/early_fusion/result.xlsx"
    "DenseNet-121 External Late Fusion|External 2 Classes Calibrated/late_fusion/result.xlsx"
    "DenseNet-121 External Siamese Fusion|External 2 Classes Calibrated/siamese_fusion/result.xlsx"
    "DenseNet-121 External Logit Fusion|External 2 Classes Calibrated/logit_fusion/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"
    
    echo -n "$label "
    python classification.py -i "$base_dir/$rel_path" "${extra_flags[@]}"
done


echo "Radiologist Results (Histology)"

# Run radiologist.py with flags
python radiologist.py "${extra_flags[@]}"  -H

# Define tasks: "Display Label|Relative Path"
tasks=(
    "3D Radiomics Internal T1 (Histology)|Internal 2 Classes Calibrated/3D Radiomics_histology/t1.xlsx"
    "3D Radiomics Internal T2 (Histology)|Internal 2 Classes Calibrated/3D Radiomics_histology/t2.xlsx"
    "DenseNet-121 Internal T1 (Histology)|Internal 2 Classes Calibrated/DenseNet-121_histology/t1.xlsx"
    "DenseNet-121 Internal T2 (Histology)|Internal 2 Classes Calibrated/DenseNet-121_histology/t2.xlsx"
    "DenseNet-121 Internal Early Fusion (Histology)|Internal 2 Classes Calibrated/early_fusion_histology/result.xlsx"
    "DenseNet-121 Internal Late Fusion (Histology)|Internal 2 Classes Calibrated/late_fusion_histology/result.xlsx"
    "DenseNet-121 Internal Siamese Fusion (Histology)|Internal 2 Classes Calibrated/siamese_fusion_histology/result.xlsx"
    "DenseNet-121 Internal Logit Fusion (Histology)|Internal 2 Classes Calibrated/logit_fusion_histology/result.xlsx"
    "3D Radiomics External T1 (Histology)|External 2 Classes Calibrated/3D Radiomics_histology/t1.xlsx"
    "3D Radiomics External T2 (Histology)|External 2 Classes Calibrated/3D Radiomics_histology/t2.xlsx"
    "DenseNet-121 External T1 (Histology)|External 2 Classes Calibrated/DenseNet-121_histology/t1.xlsx"
    "DenseNet-121 External T2 (Histology)|External 2 Classes Calibrated/DenseNet-121_histology/t2.xlsx"
    "DenseNet-121 External Early Fusion (Histology)|External 2 Classes Calibrated/early_fusion_histology/result.xlsx"
    "DenseNet-121 External Late Fusion (Histology)|External 2 Classes Calibrated/late_fusion_histology/result.xlsx"
    "DenseNet-121 External Siamese Fusion (Histology)|External 2 Classes Calibrated/siamese_fusion_histology/result.xlsx"
    "DenseNet-121 External Logit Fusion (Histology)|External 2 Classes Calibrated/logit_fusion_histology/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"
    
    echo -n "$label "
    python classification.py -i "$base_dir/$rel_path" "${extra_flags[@]}"
done