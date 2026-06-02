#!/bin/bash

# Default model
model="densenet121"
data_path="/data2/pky0507/dataset/IPMN_Classification/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -model|--model)
            model="$2"
            shift 2
            ;;
        -data_path|--data_path)
            data_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./main.sh [-model model_name] [-data_path data_path]"
            exit 1
            ;;
    esac
done


for f in {0..3}; do
    python train.py --model "$model" --f "$f" -s 42 --data-path "$data_path"
done

python fold_test.py --model "$model" --data-path "$data_path"
