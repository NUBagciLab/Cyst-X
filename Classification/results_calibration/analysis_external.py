import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
from seed import seed_everything
from calibration import calibrate, highlight_errors
from auc_ci import calculate_auc_ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    #parser.add_argument("-i", "--input", default="./External 2 Classes/3D Radiomics/t1.xlsx", type=str, help="input path")
    parser.add_argument("-i", "--input", default="./External 2 Classes/fusion/result.xlsx", type=str, help="input path")
    parser.add_argument("-o", "--output", default="./out.xlsx", type=str, help="dataset path")
    parser.add_argument("-n", "--no-calibration", action="store_true", help="no calibration, use threshold=0.5")
    args = parser.parse_args()

    seed_everything(42) # Fix seed for 95% AUC

    df = pd.read_excel(args.input)

    n_center = 7
    n_fold = 4
    center_names = ['nyu', 'CAD|MCF', 'northwestern|NU', 'AHN|ahn', 'mca', 'IU', 'EMC']
    thresholds = [0.5 for i in range(n_center)]
    for c in range(n_center):
        filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
        epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
        if not args.no_calibration:
            thresholds[c] = calibrate(epoch_y['pred'], epoch_y['true'])
        print(f"Center {c+1} threshold {thresholds[c]*100:.2f}%")

    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_predictions = []
    y_all = []
    pred_all = []
    output_all = []
    log = {'acc':[], 'auc':[], 'auc_upper':[], 'auc_lower':[], 'sens':[], 'spec':[], 'f1':[]}
    for c in range(n_center):
        filtered_df = df[df['ID'].str.contains(center_names[c], na=False)]
        epoch_y = {'true': filtered_df['Label'].to_numpy(), 'pred': filtered_df['Probability'].to_numpy()}
        output = epoch_y['pred'] >= thresholds[c]
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

        csv_images.extend(filtered_df['ID'])
        csv_labels.extend(filtered_df['Label'])
        csv_probabilities.extend(filtered_df['Probability'])
        csv_predictions.extend(output)

    log['acc'].append(accuracy_score(y_all, output_all))
    log['auc'].append(roc_auc_score(y_all, pred_all))
    lower_bound, upper_bound = calculate_auc_ci(y_all, pred_all)
    log['auc_lower'].append(lower_bound)
    log['auc_upper'].append(upper_bound)
    tn, fp, fn, tp = confusion_matrix(y_all, output_all).ravel()
    log['sens'].append(recall_score(y_all, output_all))    
    log['spec'].append(tn / (tn + fp))
    log['f1'].append(f1_score(y_all, output_all))    
    
    for c in range(n_center):
        print(f"Center {c+1} auc {log['auc'][c]:.4f} 95%auc [{log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f}] acc {log['acc'][c]:.4f} sens {log['sens'][c]:.4f} spec {log['spec'][c]:.4f} f1 {log['f1'][c]:.4f}")
    print(f"Global auc {log['auc'][-1]:.4f} 95%auc [{log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f}] acc {log['acc'][-1]:.4f} sens {log['sens'][-1]:.4f} spec {log['spec'][-1]:.4f} f1 {log['f1'][-1]:.4f}")

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
    })
    df.style.apply(highlight_errors, axis=1).to_excel(args.output, index=False)