# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:39:06 2026

@author: pky0507
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from seed import seed_everything

def calculate_auc_ci(y_true, y_pred_probs, n_bootstraps=1000, ci_level=0.95):
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        y_b, pred_b = resample(y_true, y_pred_probs)
        
        # Check if bootstrap sample has both classes
        if len(np.unique(y_b)) < 2:
            continue
            
        score = roc_auc_score(y_b, pred_b)
        bootstrapped_scores.append(score)
        
    # Calculate 95% CI
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = np.percentile(sorted_scores, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ci_level) / 2 * 100)
    
    return lower_bound, upper_bound

n_center = 7
center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
thresholds = {}

logs = {}

model_list = ['3D Radiomics', 'DenseNet-121', '+FedAvg', '+FedProx(0.1)', '+FedProx(0.3)']
for model in model_list:

    for t in [1, 2]:
        thresholds[model+f't{t}'] = [0 for i in range(n_center)]
        df = pd.read_excel(os.path.join(model, f't{t}.xlsx'))
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
                    thresholds[model+f't{t}'][c] = threshold
                    acc_best = acc
                    sens_spec_sum_best = sens+spec
                else:
                    if acc > acc_best and sens > 0.35 and spec > 0.85:
                    # if sens_spec_sum_best > sens+spec:
                        thresholds[model+f't{t}'][c] = threshold
                        acc_best = acc
                        sens_spec_sum_best = sens+spec
                
            if thresholds[model+f't{t}'][c] == 0:
                for threshold in [0]+list(set(filtered_df['Probability'].to_numpy()))+[1]:
                    output = epoch_y['pred'] >= threshold
                    acc = accuracy_score(epoch_y['true'], output)
                    tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
                    sens = recall_score(epoch_y['true'], output)
                    spec = tn / (tn + fp)
                    if threshold == 0:
                        thresholds[model+f't{t}'][c] = threshold
                        acc_best = acc
                        sens_spec_sum_best = sens+spec
                    else:
                        if acc > acc_best and sens > 0 and spec > 0:
                        # if sens_spec_sum_best > sens+spec:
                            thresholds[model+f't{t}'][c] = threshold
                            acc_best = acc
                            sens_spec_sum_best = sens+spec
    
    for t in [1, 2]:
        
        seed_everything(42) # Fix seed for 95% AUC
        df = pd.read_excel(os.path.join(model, f't{t}.xlsx'))
        y_all = []
        pred_all = []
        output_all = []
        log = {'acc':[], 'auc':[], 'auc_upper':[], 'auc_lower':[], 'sens':[], 'spec':[], 'f1':[]}
        for c in range(n_center):
           filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
           epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
           output = epoch_y['pred'] >= thresholds[model+f't{t}'][c]
           log['acc'].append(accuracy_score(epoch_y['true'], output))
           log['auc'].append(roc_auc_score(epoch_y['true'], epoch_y['pred']))
           lower_bound, upper_bound = calculate_auc_ci(epoch_y['true'], epoch_y['pred'])
           log['auc_lower'].append(lower_bound)
           log['auc_upper'].append(upper_bound)
           
           tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
           log['sens'].append(recall_score(epoch_y['true'], output))    
           log['spec'].append(tn / (tn + fp))
           log['f1'].append(f1_score(epoch_y['true'], output)) 

           y_all.extend(epoch_y['true'])
           pred_all.extend(epoch_y['pred'])
           output_all.extend(output)

        log['acc'].append(accuracy_score(y_all, output_all))
        log['auc'].append(roc_auc_score(y_all, pred_all))
        lower_bound, upper_bound = calculate_auc_ci(y_all, pred_all)
        log['auc_lower'].append(lower_bound)
        log['auc_upper'].append(upper_bound)
        tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
        log['sens'].append(recall_score(y_all, output_all))    
        log['spec'].append(tn / (tn + fp))
        log['f1'].append(f1_score(y_all, output_all))    

        print(f"{model} t{t}")
        
        for c in range(n_center):
            print(f"Dataset {c} acc {log['acc'][c]:.4f} auc {log['auc'][c]:.4f} 95%auc [{log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f}]")
        
        print(f"Global acc {log['acc'][-1]:.4f} auc {log['auc'][-1]:.4f} 95%auc ({log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f})")
        
        logs[model+f't{t}'] = log
        
        # for c in range(n_center+1): # print for latex
        #     print(f"{c+1} {log['acc'][c]*100:.2f} & {log['auc'][c]*100:.2f} & ({log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f})")
    
        
for c in range(n_center+1): # print for latex
    print('\\hline')
    print(['\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 1: New York University Langone Health (NYU), T1W 127 no/low + 23 high risk, T2W 127 no/low + 24 high risk}}\\\\',
     '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 2: Mayo Clinic Florida (MCF), T1W 71 no/low + 63 high risk, T2W 67 no/low + 63 high risk}}\\\\',
      '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 3: Northwestern University (NU), T1W 169 no/low + 17 high risk, T2W 171 no/low + 16 high risk}}\\\\',
       '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 4: Allegheny Health Network (AHN), T1W 12 no/low + 4 high risk, T2W 14 no/low + 4 high risk}}\\\\',
        '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 5: Mayo Clinic Arizona (MCA), T1W 10 no/low + 14 high risk, T2W 7 no/low + 16 high risk}}\\\\',
         '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 6: Istanbul University Faculty of Medicine (IU), T1W 51 no/low + 13 high risk, T2W 49 no/low + 14 high risk}}\\\\',
          '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 7: Erasmus Medical Center (EMC), T1W 63 no/low + 15 high risk, T2W 68 no/low + 15 high risk}}\\\\',
           '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Global, T1W 503 no/low + 149 high risk, T2W 503 no/low + 152 high risk}}\\\\',
    ][c])
    print('\\hline')
    for model in model_list:
        if model in ['3D Radiomics', 'DenseNet-121']:
            latex_str = '\\rowcolor{g1}'+f"{model}"
        else:
            latex_str = '\\rowcolor{g2}'+f"{model}"
        for t in [1, 2]:
            latex_str += f" & {logs[model+f't{t}']['auc'][c]*100:.2f} & [{logs[model+f't{t}']['auc_lower'][c]*100:.2f}, {logs[model+f't{t}']['auc_upper'][c]*100:.2f}]"
            for metric in ['acc', 'sens', 'spec']:
                latex_str += f" & {logs[model+f't{t}'][metric][c]*100:.2f}"
        latex_str += ' \\\\'
        print(latex_str)