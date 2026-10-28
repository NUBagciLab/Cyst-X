#!/usr/bin/env bash
set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 0 ]; then
    python "$script_dir/step0_reorientation.py" --input "$1"
else
    python "$script_dir/step0_reorientation.py"
fi
python "$script_dir/step1_reorientation_to_RAS.py"
python "$script_dir/step2_n4biascorrection.py"
python "$script_dir/step3_crop.py"
