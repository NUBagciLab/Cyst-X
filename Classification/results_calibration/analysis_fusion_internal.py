import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from seed import seed_everything
from calibration import calibrate, highlight_errors
from auc_ci import calculate_auc_ci_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    parser.add_argument("-i", "--input", default="./Internal 2 Classes/early_fusion/result.xlsx", type=str, help="input path")
    parser.add_argument("-o", "--output", default="./out.xlsx", type=str, help="ouput path")
    parser.add_argument("-n", "--no-calibration", action="store_true", help="no calibration, use threshold=0.5")
    args = parser.parse_args()

    seed_everything(42) # Fix seed for 95% AUC

    df = pd.read_excel(args.input)

    n_center = 7
    n_fold = 5
    center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
    plt.rcParams.update({'font.size': 16})
    thresholds = [0.5 for i in range(n_center)]
    if not args.no_calibration:
        for c in range(n_center):
            filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
            epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
            thresholds[c] = calibrate(epoch_y['pred'], epoch_y['true'])
              
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
    
    log = [{} for j in range(n_fold)]
    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_predictions = []
    folds_data = []
    for fold in range(n_fold):
        y_all = []
        pred_all = []
        output_all = []
        for c in range(n_center):
            filtered_df = df[df['ID'].str.contains(center_names[c], na=False) & (df['Fold'] == fold)]
            epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
            output = epoch_y['pred'] >= thresholds[c]
            
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
            output_all.extend(output)

            csv_images.extend(filtered_df['ID'])
            csv_labels.extend(filtered_df['Label'])
            csv_probabilities.extend(filtered_df['Probability'])
            csv_predictions.extend([int(i) for i in output])

        folds_data.append((y_all, pred_all))
        
        log[fold]['acc'] = accuracy_score(y_all, output_all)
        log[fold]['auc'] = roc_auc_score(y_all, pred_all)
        tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
        log[fold]['sens'] = recall_score(y_all, output_all) 
        log[fold]['spec'] = tn / (tn + fp) 

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

    log_mean = {}   
    log_std = {}   
    
    for metric in ['acc', 'auc', 'sens', 'spec']:
        log_mean[metric] = np.mean([log[fold][metric] for fold in range(n_fold)])
        log_std[metric] = np.std([log[fold][metric] for fold in range(n_fold)])
    epoch_y = {'true': df['Label'].to_numpy(), 'pred': df['Probability'].to_numpy()}
    log_mean['auc_lower'], log_mean['auc_upper']  = calculate_auc_ci_cv(folds_data)

    print(f"Global auc {log_mean['auc']*100:.2f}±{log_std['auc']*100:.2f} 95%CI [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}] acc {log_mean['acc']*100:.2f}±{log_std['acc']*100:.2f} sens {log_mean['sens']*100:.2f}±{log_std['sens']*100:.2f} spec {log_mean['spec']*100:.2f}±{log_std['spec']*100:.2f}")

    if not args.no_calibration:
    # --- Finalize ROC Plot ---
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        ax_roc.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})', lw=2)
        ax_roc.plot([0, 1], [0, 1], '--r', label='Chance')
        ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f"Mean ROC")
        ax_roc.legend(loc='lower right')
        ax_roc.axis([0, 1, 0, 1])
        ax_roc.grid()
        fig_roc.savefig(args.output.replace('.xlsx', "roc.pdf"), format="pdf", bbox_inches='tight')

        # --- Finalize PR Plot ---
        mean_precision = np.mean(precisions_list, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        
        ax_pr.plot(mean_recall, mean_precision, color='green', label=f'Mean PR (AP={mean_ap:.4f}±{std_ap:.4f})', lw=2)
        # The baseline for PR is the proportion of positive samples
        baseline = sum(y_all) / len(y_all) if len(y_all) > 0 else 0
        ax_pr.axhline(baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
        ax_pr.set(xlabel='Recall', ylabel='Precision', title=f"Mean PR Curve")
        ax_pr.legend(loc='lower left')
        ax_pr.axis([0, 1, 0, 1])
        ax_pr.grid()
        fig_pr.savefig(args.output.replace('.xlsx', "pr_curve.pdf"), format="pdf", bbox_inches='tight')

        mapping = {value: i for i, value in enumerate(csv_images)}
        indices = [mapping[value] for value in df['ID']]
        csv_images = [csv_images[i] for i in indices]
        csv_labels = [csv_labels[i] for i in indices]
        csv_probabilities = [csv_probabilities[i] for i in indices]
        csv_predictions = [csv_predictions[i] for i in indices]

        df = pd.DataFrame({
            'ID': csv_images,
            'Risk Assessment': df['Risk Assessment'],
            'Label': csv_labels,
            'Prediction': csv_predictions,
            'Probability': csv_probabilities,
            'Center': pd.Series(center_names),
            'Threshold': pd.Series(thresholds)
        })
        df.style.apply(highlight_errors, axis=1).to_excel(args.output, index=False)

