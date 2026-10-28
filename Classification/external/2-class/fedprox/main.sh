#!/bin/bash
set -e

# Default configurations
model="densenet121"
data_path="../../../../Cyst-X-resampled/classification"
output_dir="./saved/"
t="all"
mu=0.1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -model|--model)
            model="$2"
            shift 2
            ;;
        --data-path|-data_path|--data_path)
            data_path="$2"
            shift 2
            ;;
        -output-dir|--output-dir|-o)
            output_dir="$2"
            shift 2
            ;;
        -mu|--mu)
            mu="$2"
            shift 2
            ;;
        -t|--t)
            t="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./main.sh [--model model_name] [--data-path data_path] [--output-dir output_dir] [-t 1|2|all] [--mu mu]"
            exit 1
            ;;
    esac
done

# Determine list of modalities to run
if [[ "$t" == "1" ]]; then
    t_list=(1)
elif [[ "$t" == "2" ]]; then
    t_list=(2)
else
    echo "[Info] Modality -t not specified as 1 or 2 (received: '$t'). Running both: t1 first, then t2."
    t_list=(1 2)
fi

# Execute training and evaluation for each selected modality
for current_t in "${t_list[@]}"; do
    echo "=========================================="
    echo " Running IPMN External Training & Testing"
    echo " Model      : $model"
    echo " FedProx mu : $mu"
    echo " Modality   : t$current_t"
    echo " Data Path  : $data_path"
    echo " Output Dir : $output_dir"
    echo "=========================================="

    # Run seven leave-one-center-out experiments
    for f in {0..6}; do
        echo ">>> Training: Modality t$current_t | Fold $f"
        python train.py --model "$model" --t "$current_t" --f "$f" -s 42 --data-path "$data_path" --output-dir "$output_dir" --mu "$mu" -j 4
    done

    # Run calibration
    echo ">>> Running Calibration for t$current_t"
    python calibrate.py --model "$model" --t "$current_t" --data-path "$data_path" --output-dir "$output_dir" --mu "$mu"

    # Run fold aggregation / evaluation
    echo ">>> Running Fold Evaluation for t$current_t"
    python fold_test.py --model "$model" --t "$current_t" --data-path "$data_path" --output-dir "$output_dir" --mu "$mu"

    # Run fold aggregation / evaluation with calibration
    echo ">>> Running Fold Evaluation for t$current_t"
    python fold_test.py --model "$model" --t "$current_t" --data-path "$data_path" --output-dir "$output_dir" --mu "$mu" -c

    echo "Completed training and fold test for t$current_t."
    echo ""
done

echo "All scheduled experiments completed."
