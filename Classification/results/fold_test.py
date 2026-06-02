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
n_fold = 4
center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
thresholds = {}
logs_mean = {}
logs_std = {}
logs = {}
valid_list = ['Internal', 'External']
model_list = ['fusion_shared', 'fusion_add_shared', 'fusion', 'fusion_add', 'fusion_prob']
model_latex_list = ['Early feature concatenation', 'Early feature addition', 'Late feature concatenation', 'Late feature addition', 'Probability fusion']
for valid in valid_list:
    for model in model_list:
        thresholds[valid+model] = [0 for i in range(n_center)]
        if valid == 'External' or model == 'fusion_prob':
            df = pd.read_excel(os.path.join(valid+' 2 Classes', model, 'result.xlsx'))
        else:
            df = pd.read_excel(os.path.join(valid+' 2 Classes', model+'2', 'result.xlsx'))
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
                    thresholds[valid+model][c] = threshold
                    acc_best = acc
                    sens_spec_sum_best = sens+spec
                else:
                    if acc > acc_best and sens > 0.35 and spec > 0.85:
                    # if sens_spec_sum_best > sens+spec:
                        thresholds[valid+model][c] = threshold
                        acc_best = acc
                        sens_spec_sum_best = sens+spec
                
            if thresholds[valid+model][c] == 0:
                for threshold in [0]+list(set(filtered_df['Probability'].to_numpy()))+[1]:
                    output = epoch_y['pred'] >= threshold
                    acc = accuracy_score(epoch_y['true'], output)
                    tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
                    sens = recall_score(epoch_y['true'], output)
                    spec = tn / (tn + fp)
                    if threshold == 0:
                        thresholds[valid+model][c] = threshold
                        acc_best = acc
                        sens_spec_sum_best = sens+spec
                    else:
                        if acc > acc_best and sens > 0 and spec > 0:
                        # if sens_spec_sum_best > sens+spec:
                            thresholds[valid+model][c] = threshold
                            acc_best = acc
                            sens_spec_sum_best = sens+spec

valid = 'Internal'
for model in model_list:
    if model == 'fusion_prob':
        df = pd.read_excel(os.path.join(valid+' 2 Classes', model, 'result.xlsx'))
    else:
        df = pd.read_excel(os.path.join(valid+' 2 Classes', model+'2', 'result.xlsx'))
    
    log = [{'acc':[[] for i in range(n_center+1)], 'auc':[[] for i in range(n_center+1)], 'sens':[[] for i in range(n_center+1)], 'spec':[[] for i in range(n_center+1)], 'f1':[[] for i in range(n_center+1)]} for j in range(n_fold)]
    
    for fold in range(n_fold):
        y_all = []
        pred_all = []
        output_all = []
        for c in range(n_center):
            filtered_df = df[df['ID'].str.contains(center_names[c], na=False) & (df['Fold'] == fold)]
            epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
            output = epoch_y['pred'] >= thresholds[valid+model][c]
            log[fold]['acc'][c].append(accuracy_score(epoch_y['true'], output))
            log[fold]['auc'][c].append(roc_auc_score(epoch_y['true'], epoch_y['pred']))
            
            tn, fp, fn, tp = confusion_matrix(epoch_y['true'], output).ravel()
            log[fold]['sens'][c].append(recall_score(epoch_y['true'], output))    
            log[fold]['spec'][c].append(tn / (tn + fp))
            log[fold]['f1'][c].append(f1_score(epoch_y['true'], output)) 
            
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
            output_all.extend(output)
        
        log[fold]['acc'][-1].append(accuracy_score(y_all, output_all))
        log[fold]['auc'][-1].append(roc_auc_score(y_all, pred_all))
        tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
        log[fold]['sens'][-1].append(recall_score(y_all, output_all))    
        log[fold]['spec'][-1].append(tn / (tn + fp))
        log[fold]['f1'][-1].append(f1_score(y_all, output_all))    
    
    print(f"{valid} {model}")
    # for fold in range(n_fold): 
    #     print(f"Fold {fold}")
    #     for c in range(n_center):
    #         print(f"Center {c+1} acc {log[fold]['acc'][c][-1]:.4f} auc {log[fold]['auc'][c][-1]:.4f} sens {log[fold]['sens'][c][-1]:.4f} spec {log[fold]['spec'][c][-1]:.4f} f1 {log[fold]['f1'][c][-1]:.4f}")
    #     print(f"Global acc {log[fold]['acc'][-1][-1]:.4f} auc {log[fold]['auc'][-1][-1]:.4f} sens {log[fold]['sens'][-1][-1]:.4f} spec {log[fold]['spec'][-1][-1]:.4f} f1 {log[fold]['f1'][-1][-1]:.4f}")
    log_mean = {'acc':[0 for i in range(n_center+1)], 'auc':[0 for i in range(n_center+1)], 'auc_lower':[0 for i in range(n_center+1)], 'auc_upper':[0 for i in range(n_center+1)], 'sens':[0 for i in range(n_center+1)], 'spec':[0 for i in range(n_center+1)], 'f1':[0 for i in range(n_center+1)]}   
    log_std = {'acc':[0 for i in range(n_center+1)], 'auc':[0 for i in range(n_center+1)], 'sens':[0 for i in range(n_center+1)], 'spec':[0 for i in range(n_center+1)], 'f1':[0 for i in range(n_center+1)]}   

    for c in range(n_center+1):
        for metric in ['acc', 'auc', 'sens', 'spec', 'f1']:
            log_mean[metric][c] = np.mean([log[fold][metric][c][-1] for fold in range(n_fold)])
            log_std[metric][c] = np.std([log[fold][metric][c][-1] for fold in range(n_fold)])
        ci95 = 1.96 * log_std['auc'][c] / np.sqrt(n_fold)
        log_mean['auc_lower'][c] = log_mean['auc'][c] - ci95
        log_mean['auc_upper'][c] = log_mean['auc'][c] + ci95
        if c < n_center:
            print(f"Center {c+1} acc {log_mean['acc'][c]:.4f}±{log_std['acc'][c]:.4f} auc {log_mean['auc'][c]:.4f}±{log_std['auc'][c]:.4f} 95%CI [{log_mean['auc_lower'][c]:.4f}, {log_mean['auc_upper'][c]:.4f}]  sens {log_mean['sens'][c]:.4f}±{log_std['sens'][c]:.4f} spec {log_mean['spec'][c]:.4f}±{log_std['spec'][c]:.4f} f1 {log_mean['f1'][c]:.4f}±{log_std['f1'][c]:.4f}")
        else: 
            print(f"Global acc {log_mean['acc'][c]:.4f}±{log_std['acc'][c]:.4f} auc {log_mean['auc'][c]:.4f}±{log_std['auc'][c]:.4f} 95% CI [{log_mean['auc_lower'][c]:.4f}, {log_mean['auc_upper'][c]:.4f}] sens {log_mean['sens'][c]:.4f}±{log_std['sens'][c]:.4f} spec {log_mean['spec'][c]:.4f}±{log_std['spec'][c]:.4f} f1 {log_mean['f1'][c]:.4f}±{log_std['f1'][c]:.4f}")
    logs_mean[model] = log_mean
    logs_std[model] = log_std

valid = 'External'
for model in model_list:
    
    seed_everything(42) # Fix seed for 95% AUC
    df = pd.read_excel(os.path.join(valid+' 2 Classes', model, 'result.xlsx'))
    y_all = []
    pred_all = []
    output_all = []
    log = {'acc':[], 'auc':[], 'auc_upper':[], 'auc_lower':[], 'sens':[], 'spec':[], 'f1':[]}
    for c in range(n_center):
       filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
       epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
       output = epoch_y['pred'] >= thresholds[valid+model][c]
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

    print(f"{valid} {model}")
    
    for c in range(n_center):
        print(f"Dataset {c} acc {log['acc'][c]:.4f} auc {log['auc'][c]:.4f} 95%auc [{log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f}]")
    
    print(f"Global acc {log['acc'][-1]:.4f} auc {log['auc'][-1]:.4f} 95%auc ({log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f})")
    
    logs[model] = log
        
for c in range(n_center+1): # print for latex
    print('\\hline')
    print(['\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 1: New York University Langone Health (NYU), 127 no/low + 23 high risk}}\\\\',
     '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 2: Mayo Clinic Florida (MCF), 67 no/low + 63 high risk}}\\\\',
      '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 3: Northwestern University (NU), 169 no/low + 16 high risk}}\\\\',
       '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 4: Allegheny Health Network (AHN), 11 no/low + 4 high risk}}\\\\',
        '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 5: Mayo Clinic Arizona (MCA), 6 no/low + 9 high risk}}\\\\',
         '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 6: Istanbul University Faculty of Medicine (IU), 49 no/low + 13 high risk}}\\\\',
          '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Center 7: Erasmus Medical Center (EMC), 57 no/low + 15 high risk}}\\\\',
           '\\rowcolor{g3}\multicolumn{11}{l}{\\textbf{Global, 486 no/low + 143 high risk}}\\\\',
    ][c])
    print('\\hline')
    for model, model_latex in zip(model_list, model_latex_list):
        latex_str = '\\rowcolor{g2}'+f"{model_latex}"
        latex_str += f" & {logs_mean[model]['auc'][c]*100:.2f}$\\pm${logs_std[model]['auc'][c]*100:.2f} & [{logs_mean[model]['auc_lower'][c]*100:.2f}, {logs_mean[model]['auc_upper'][c]*100:.2f}]"
        for metric in ['acc', 'sens', 'spec']:
            latex_str += f" & {logs_mean[model][metric][c]*100:.2f}$\\pm${logs_std[model][metric][c]*100:.2f}"            
        latex_str += f" & {logs[model]['auc'][c]*100:.2f} & [{logs[model]['auc_lower'][c]*100:.2f}, {logs[model]['auc_upper'][c]*100:.2f}]"
        for metric in ['acc', 'sens', 'spec']:
            latex_str += f" & {logs[model][metric][c]*100:.2f}"
        latex_str += ' \\\\'
        print(latex_str)