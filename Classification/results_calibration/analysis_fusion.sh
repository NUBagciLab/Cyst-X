#!/usr/bin/env bash

# Set default source and destination base directories
DEFAULT_SRC_INTERNAL="./Internal 2 Classes"
DEFAULT_OUT_INTERNAL="./Internal 2 Classes Calibrated"
DEFAULT_SRC_EXTERNAL="./External 2 Classes"
DEFAULT_OUT_EXTERNAL="./External 2 Classes Calibrated"
src_base_internal="$DEFAULT_SRC_INTERNAL"
out_base_internal="$DEFAULT_OUT_INTERNAL"
src_base_external="$DEFAULT_SRC_EXTERNAL"
out_base_external="$DEFAULT_OUT_EXTERNAL"
calib_flags=()
positional_args=()

# Parse flags and positional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--no-calibration)
            calib_flags+=("$1")
            shift
            ;;
        *)
            positional_args+=("$1")
            shift
            ;;
    esac
done

# Assign positional directories if provided
if [ ${#positional_args[@]} -ge 1 ]; then
    src_base="${positional_args[0]}"
fi
if [ ${#positional_args[@]} -ge 2 ]; then
    out_base="${positional_args[1]}"
fi

tasks=(
    "DenseNet-121 Early Fusion|early_fusion/result.xlsx"
    "DenseNet-121 Late Fusion|late_fusion/result.xlsx"
    "DenseNet-121 Siamese Fusion|siamese_fusion/result.xlsx"
    "DenseNet-121 Logit Fusion|logit_fusion/result.xlsx"
    "DenseNet-121 Early Fusion (Histology)|early_fusion_histology/result.xlsx"
    "DenseNet-121 Late Fusion (Histology)|late_fusion_histology/result.xlsx"
    "DenseNet-121 Siamese Fusion (Histology)|siamese_fusion_histology/result.xlsx"
    "DenseNet-121 Logit Fusion (Histology)|logit_fusion_histology/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"

    echo "Running Internal Task: $label"

    input_file="$src_base_internal/$rel_path"
    output_file="$out_base_internal/$rel_path"

    # Ensure the target output subdirectory exists before writing
    mkdir -p "$(dirname "$output_file")"

    python analysis_fusion_internal.py -i "$input_file" -o "$output_file" "${calib_flags[@]}"
done

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"

    echo "Running External Task: $label"

    input_file="$src_base_external/$rel_path"
    output_file="$out_base_external/$rel_path"

    # Ensure the target output subdirectory exists before writing
    mkdir -p "$(dirname "$output_file")"

    python analysis_external.py -i "$input_file" -o "$output_file" "${calib_flags[@]}"
done
