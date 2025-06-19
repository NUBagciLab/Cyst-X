#!/bin/bash

# Read t from the command-line argument
t=$1  # First argument passed to the script

# Check if t is provided
if [ -z "$t" ]; then
    echo "Error: Please provide a value for t. Usage: ./train.sh <t>"
    exit 1
fi

# Loop over f values
for f in {0..4}  # f ranges from 0 to 4
do
    echo "Running: python train.py --t $t --f $f --seed 42 -j 4"
    python train.py --t $t --f $f --seed 42 -j 4
done
