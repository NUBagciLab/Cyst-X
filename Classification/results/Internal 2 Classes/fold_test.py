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

n_center = 7
n_fold = 4
center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
plt.rcParams.update({'font.size': 16})
thresholds = {}

logs_mean = {}
logs_std = {}

model_list = ['3D Radiomics', 'ResNet-34', 'ResNet-50', 'EfficientNet-B0', 'DenseNet-121', '+FedAvg', '+FedProx(0.1)', '+FedProx(0.3)']
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
        df = pd.read_excel(os.path.join(model, f't{t}.xlsx'))
        
    # --- Setup for ROC ---
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # --- Setup for PR ---
        precisions_list = []
        aps = []
        mean_recall = np.linspace(0, 1, 100)
        
        # Create figures
        fig_roc, ax_roc = plt.subplots(figsize=(7, 7))    
        fig_pr, ax_pr = plt.subplots(figsize=(7, 7))    
        
        log = [{'acc':[[] for i in range(n_center+1)], 'auc':[[] for i in range(n_center+1)], 'sens':[[] for i in range(n_center+1)], 'spec':[[] for i in range(n_center+1)], 'f1':[[] for i in range(n_center+1)]} for j in range(n_fold)]
        
        for fold in range(n_fold):
            y_all = []
            pred_all = []
            output_all = []
            for c in range(n_center):
                filtered_df = df[df['ID'].str.contains(center_names[c], na=False) & (df['Fold'] == fold)]
                epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
                output = epoch_y['pred'] >= thresholds[model+f't{t}'][c]
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
            
            # --- ROC Calculation ---
            fpr, tpr, _ = roc_curve(y_all, pred_all)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            ax_roc.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} ROC (AUC={roc_auc:.4f})')

            # --- PR Calculation ---
            precision, recall, _ = precision_recall_curve(y_all, pred_all)
            # Flip recall/precision because np.interp needs x-axis (recall) to be increasing
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions_list.append(interp_precision)
            ap = average_precision_score(y_all, pred_all)
            aps.append(ap)
            ax_pr.plot(recall, precision, alpha=0.3, label=f'Fold {fold+1} PR (AP={ap:.4f})')
        
        print(f"{model} t{t}")
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
        logs_mean[model+f't{t}'] = log_mean
        logs_std[model+f't{t}'] = log_std
        
    # --- Finalize ROC Plot ---
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        ax_roc.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})', lw=2)
        ax_roc.plot([0, 1], [0, 1], '--r', label='Chance')
        ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f"Mean ROC - T{t}W")
        ax_roc.legend(loc='lower right')
        ax_roc.axis([0, 1, 0, 1])
        ax_roc.grid()
        fig_roc.savefig(os.path.join(model, f"T{t}roc.pdf"), format="pdf", bbox_inches='tight')

        # --- Finalize PR Plot ---
        mean_precision = np.mean(precisions_list, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        
        ax_pr.plot(mean_recall, mean_precision, color='green', label=f'Mean PR (AP={mean_ap:.4f}±{std_ap:.4f})', lw=2)
        # The baseline for PR is the proportion of positive samples
        baseline = sum(y_all) / len(y_all) if len(y_all) > 0 else 0
        ax_pr.axhline(baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
        ax_pr.set(xlabel='Recall', ylabel='Precision', title=f"Mean PR Curve - T{t}W")
        ax_pr.legend(loc='lower left')
        ax_pr.axis([0, 1, 0, 1])
        ax_pr.grid()
        fig_pr.savefig(os.path.join(model, f"T{t}pr_curve.pdf"), format="pdf", bbox_inches='tight')
             
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
            latex_str += f" & {logs_mean[model+f't{t}']['auc'][c]*100:.2f}$\\pm${logs_std[model+f't{t}']['auc'][c]*100:.2f} & [{logs_mean[model+f't{t}']['auc_lower'][c]*100:.2f}, {logs_mean[model+f't{t}']['auc_upper'][c]*100:.2f}]"
            for metric in ['acc', 'sens', 'spec']:
                latex_str += f" & {logs_mean[model+f't{t}'][metric][c]*100:.2f}$\\pm${logs_std[model+f't{t}'][metric][c]*100:.2f}"
        latex_str += ' \\\\'
        print(latex_str)