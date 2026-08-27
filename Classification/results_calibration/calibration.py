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
    best_threshold = 0
    for threshold in [0]+list(set(probabilities))+[1]:
        output = probabilities >= threshold
        acc = accuracy_score(labels, output)
        tn, fp, fn, tp = confusion_matrix(labels, output).ravel()
        sens = recall_score(labels, output)
        spec = tn / (tn + fp)
        if threshold == 0:
            best_threshold = threshold
            acc_best = acc
        else:
            # if acc > acc_best:
            if acc > acc_best and sens > 0.35 and spec > 0.85:
                best_threshold = threshold
                acc_best = acc
        
    if best_threshold == 0:
        for threshold in [0]+list(set(probabilities))+[1]:
            output = probabilities >= threshold
            acc = accuracy_score(labels, output)
            tn, fp, fn, tp = confusion_matrix(labels, output).ravel()
            sens = recall_score(labels, output)
            spec = tn / (tn + fp)
            if threshold == 0:
                best_threshold = threshold
                acc_best = acc
            else:
                if acc > acc_best and sens > 0 and spec > 0:
                    best_threshold = threshold
                    acc_best = acc
    return best_threshold

def highlight_errors(row):
    # If Prediction != Label, color the row light red
    if row['Label'] != row['Prediction']:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)