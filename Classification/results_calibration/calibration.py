# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:21:54 2026

@author: pky0507
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
def calibrate(probabilities, labels):
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    acc_list = []
    sens_list = []
    spec_list = []
    threshold_list = [0]+list(set(probabilities))+[1]
    for threshold in threshold_list:
        output = probabilities >= threshold
        acc = accuracy_score(labels, output)
        tn, fp, fn, tp = confusion_matrix(labels, output).ravel()
        sens = recall_score(labels, output)
        spec = tn / (tn + fp)
        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)
    
    acc_list = np.array(acc_list)
    sens_list = np.array(sens_list)
    spec_list = np.array(spec_list)
    threshold_list = np.array(threshold_list)

    # Filter indices where both conditions are satisfied
    valid_indices = np.where((sens_list > 0.35) & (spec_list > 0.85))[0]

    if len(valid_indices) > 0:
        # Get index with the highest accuracy among valid candidates
        best_idx = valid_indices[np.argmax(acc_list[valid_indices])]
        best_threshold = threshold_list[best_idx]
    else:
        valid_indices = np.where((sens_list > 0) & (spec_list > 0))[0]
        if len(valid_indices) > 0:
            # Get index with the highest accuracy among valid candidates
            best_idx = valid_indices[np.argmax(acc_list[valid_indices])]
            best_threshold = threshold_list[best_idx]
        else:
            best_threshold = 0.5
            print("No valid threshold found. Defaulting to 0.5.")
    return best_threshold

def highlight_errors(row):
    # If Prediction != Label, color the row light red
    if row['Label'] != row['Prediction']:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)