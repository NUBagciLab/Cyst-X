#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -gt 1 ]]; then
    echo "Usage: bash run_preprocessing.sh [aligned_segmentation_folder]" >&2
    exit 2
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash run_preprocessing.sh [aligned_segmentation_folder]"
    echo "Runs RAS reorientation, N4 correction, and cropping (steps 1-3)."
    exit 0
fi

if [[ "$#" -eq 1 ]]; then
    python "$script_dir/step1_reorientation_to_RAS.py" --input "$1"
else
    python "$script_dir/step1_reorientation_to_RAS.py"
fi
python "$script_dir/step2_n4biascorrection.py"
python "$script_dir/step3_crop.py"
