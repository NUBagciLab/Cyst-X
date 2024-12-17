#!/bin/bash
# Iterate over --t values from 1 to 2
for t in {1..2}; do
  # Iterate over --f values from 0 to 3
  for f in {0..4}; do
    echo "Running: python train.py $DATA_PATH --t $t --f $f"
    python train.py $DATA_PATH --t "$t" --f "$f" -s 42 --data-path /data/pky0507/IPMN_Classification/
  done
done
