# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:25:55 2026

@author: pky0507
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd

n_center = 7

center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
df_ref = pd.read_csv('../../Cyst-X_bigdata_risk_assessment.csv')
names_ref = df_ref['Patient ID'].tolist()
thresholds = [0 for i in range(n_center)]
for path in [os.path.join('3D Radiomics', 't1.xlsx'),
             os.path.join('3D Radiomics', 't2.xlsx'),
             os.path.join('DenseNet-121', 't1.xlsx'),
             os.path.join('DenseNet-121', 't2.xlsx'),
             os.path.join('fusion_shared2', 'result.xlsx'),
             os.path.join('fusion_add_shared2', 'result.xlsx'),
             os.path.join('fusion2', 'result.xlsx'),
             os.path.join('fusion_add2', 'result.xlsx'),
             os.path.join('fusion_prob', 'result.xlsx')]:
    print(path)
    df = pd.read_excel(path)
    for c in range(n_center):
        filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
        epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
        for threshold in [0]+list(set(filtered_df['Probability'].to_numpy()))+[1]:
            output = epoch_y['pred'] >= threshold
            acc = accuracy_score(epoch_y['true'], output)
            tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
            sens = recall_score(epoch_y['true'], output)
            spec = tn / (tn + fp)
            if threshold == 0:
                thresholds[c] = threshold
                acc_best = acc
                sens_spec_sum_best = sens+spec
            else:
                if acc > acc_best and sens > 0.35 and spec > 0.85:
                # if sens_spec_sum_best > sens+spec and sens > 0.35 and spec > 0.85:
                    thresholds[c] = threshold
                    acc_best = acc
                    sens_spec_sum_best = sens+spec
            
        if thresholds[c] == 0:
            for threshold in [0]+list(set(filtered_df['Probability'].to_numpy()))+[1]:
                output = epoch_y['pred'] >= threshold
                acc = accuracy_score(epoch_y['true'], output)
                tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
                sens = recall_score(epoch_y['true'], output)
                spec = tn / (tn + fp)
                if threshold == 0:
                    thresholds[c] = threshold
                    acc_best = acc
                    sens_spec_sum_best = sens+spec
                else:
                    if acc > acc_best and sens > 0 and spec > 0:
                    # if sens_spec_sum_best > sens+spec and sens > 0 and spec > 0:
                        thresholds[c] = threshold
                        acc_best = acc
                        sens_spec_sum_best = sens+spec
    
    df2 = df[df['ID'].isin(names_ref)]
    
    y_all = []
    pred_all = []
    output_all = []
    for c in range(n_center):
        filtered_df = df2[df2['ID'].str.contains(center_names[c], na=False)]
        epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
        output = epoch_y['pred'] >= thresholds[c]
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        output_all.extend(output)
    tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
    sens1 = recall_score(y_all, output_all)
    spec1 = tn / (tn + fp)
    print(f'Sens:{sens1*100:.2f}, Spec:{spec1*100:.2f}')
    
    thresholds = [0.5 for i in range(n_center)]
    y_all = []
    pred_all = []
    output_all = []
    for c in range(n_center):
        filtered_df = df2[df2['ID'].str.contains(center_names[c], na=False)]
        epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
        output = epoch_y['pred'] >= thresholds[c]
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        output_all.extend(output)
    tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
    sens2 = recall_score(y_all, output_all)
    spec2 = tn / (tn + fp)

    print(f'Sens:{sens2*100:.2f}, Spec:{spec2*100:.2f}')
    print(f'{sens1*100:.2f}({sens2*100:.2f}) & {spec1*100:.2f}({spec2*100:.2f})')
    print()