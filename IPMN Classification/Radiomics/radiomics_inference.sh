#!/usr/bin/env bash

script="/home/pyq6817/New_IPMN_Classification/Radiomics/radiomics_inference.py"

modality="t2"
split="/home/pyq6817/New_IPMN_Classification/Radiomics/splits_internal_$modality.json"
features="/home/pyq6817/New_IPMN_Classification/Radiomics/features/radiomics_features_$modality.csv"

for fold in {0..3}; do
    output="/data2/pyq6817/CystX/radiomics_results/${modality}_mRMR_internal/predictions_fold$fold.csv"
    python "$script" -f "$fold" --split "$split" --features "$features" -o "$output"
done