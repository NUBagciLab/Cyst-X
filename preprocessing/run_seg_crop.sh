#!/usr/bin/env bash
set -euo pipefail

network="${1:?Specify PanSegNet or Swin}"
modality="${2:?Specify 1 or 2}"
case "$network" in PanSegNet|Swin) ;; *) exit 2 ;; esac
case "$modality" in 1|2) ;; *) exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "$script_dir")"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/preprocessing/.cache/matplotlib"
export XDG_CACHE_HOME="$project_root/preprocessing/.cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$project_root/logs/preprocessing"
log_file="$project_root/logs/preprocessing/crop_${network,,}_t${modality}.log"
set +e
python "$script_dir/step3_crop.py" \
    --input "$project_root/Cyst-X-n4/segmentation" \
    --mask-input "$project_root/seg/${network}-RAS/segmentation" \
    --output "$project_root/seg/${network}-n4/classification" \
    -t "$modality" > "$log_file" 2>&1
status=$?
printf '\n%s T%s cropping exited with status %s\n' "$network" "$modality" "$status"
exit "$status"
