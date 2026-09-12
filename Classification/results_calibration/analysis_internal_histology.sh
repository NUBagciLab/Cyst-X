#!/usr/bin/env bash

# Set default source and destination base directories
DEFAULT_SRC="./Internal 2 Classes"
DEFAULT_OUT="./Internal 2 Classes Calibrated"

src_base="$DEFAULT_SRC"
out_base="$DEFAULT_OUT"
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
    "3D Radiomics T1 Histology|3D Radiomics_histology/t1.xlsx"
    "3D Radiomics T2 Histology|3D Radiomics_histology/t2.xlsx"
    "DenseNet-121 T1 Histology|DenseNet-121_histology/t1.xlsx"
    "DenseNet-121 T2 Histology|DenseNet-121_histology/t2.xlsx"
    "DenseNet-121 Early Feature Concatenation Histology|fusion_shared2_histology/result.xlsx"
    "DenseNet-121 Late Feature Concatenation Histology|fusion2_histology/result.xlsx"
)

for task in "${tasks[@]}"; do
    label="${task%%|*}"
    rel_path="${task##*|}"

    echo "Running Task: $label"

    input_file="$src_base/$rel_path"
    output_file="$out_base/$rel_path"

    # Ensure the target output subdirectory exists before writing
    mkdir -p "$(dirname "$output_file")"

    python analysis_internal.py -i "$input_file" -o "$output_file" "${calib_flags[@]}"
done