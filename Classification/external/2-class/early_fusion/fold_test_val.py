"""Evaluate saved models on validation data; write val_result TXT/XLSX only.

Use -c to load existing thresholds.json. Missing thresholds cause an error.
Rows are predictions per fold; a case may occur in multiple validation folds.
"""
import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from sklearn.metrics import roc_auc_score, accuracy_score
from train import load_data, test_fn
from data_loader import get_pair_table
import pandas as pd
from auc_ci import calculate_auc_ci, highlight_errors
from sklearn.metrics import accuracy_score, confusion_matrix


def write_validation_excel(frame, path):
    """Write one worksheet per model fold, preserving predictions and highlighting."""
    fold_column = 'Held-out Center'
    with pd.ExcelWriter(path) as writer:
        for fold in range(7):
            sheet = frame.loc[frame[fold_column] == fold].reset_index(drop=True)
            if sheet["ID"].duplicated().any():
                raise ValueError(f"Duplicate validation case within fold {fold}.")
            sheet.style.apply(highlight_errors, axis=1).to_excel(
                writer, sheet_name=f"Center_{fold + 1}", index=False)


def validation_case_rows(loader, predictions):
    """Read IDs in exactly the order used by the unshuffled validation loader."""
    dataset = loader.dataset
    if hasattr(dataset, "image_files"):
        images, labels = dataset.image_files, dataset.labels
    else:
        records = dataset.data
        images = [record["t1"] for record in records]
        labels = [record["label"] for record in records]
    targets = np.asarray(labels).reshape(-1)
    inferred = np.asarray(predictions["true"]).reshape(-1)
    probabilities = np.asarray(predictions["pred"]).reshape(-1)
    if len(images) != len(probabilities) or not np.array_equal(targets, inferred):
        raise ValueError("Validation IDs/labels do not align with model predictions.")
    return images, np.asarray(labels).reshape(-1, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/Cyst-X-resampled/classification", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("-c", "--calibrate", action="store_true", help="Use existing thresholds.json; never recalibrate")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    result_lines = []

    def print_result(message):
        print(message)
        result_lines.append(message)


    device = torch.device(args.device)
    
    n_center = 7

    if args.calibrate:
        threshold_path = os.path.join(args.output_dir, "thresholds.json")
        with open(threshold_path, 'r') as threshold_file:
            thresholds = json.load(threshold_file)
        if len(thresholds) != n_center or not np.isfinite(np.asarray(thresholds, dtype=float)).all():
            raise ValueError("Expected one finite saved threshold per validation fold.")
        print_result("Loading existing thresholds; calibration is never rerun.")
        for fold, threshold in enumerate(thresholds):
            print_result(f"Fold {fold} threshold: {threshold:.4f}")
    else:
        print_result("Calibration not requested. Using default threshold of 0.5 for all folds.")
        thresholds = [0.5 for _ in range(n_center)]

    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    y_all = []
    pred_all = []
    output_all = []
    log = {'loss':[], 'acc':[], 'auc':[], 'sens':[], 'spec':[], 'auc_lower':[], 'auc_upper':[]}
    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_predictions = []
    csv_centers = []
    csv_thresholds = []
    for c in range(n_center):
        args.fold = c      
        _, val_dataloader, _ = load_data(args)        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        epoch_log, epoch_y = test_fn(val_dataloader, model, loss_fn, device)

        # Compute balanced accuracy and F1 score if not already provided by test_fn
        y_true_fold = np.array(epoch_y['true'])
        y_pred_fold = np.array(epoch_y['pred']) >= thresholds[c]
        epoch_log['acc'] = accuracy_score(y_true_fold, y_pred_fold)

        tn, fp, fn, tp = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1]).ravel()
        epoch_log['sens'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        epoch_log['spec'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        for metric in ['loss', 'acc', 'sens', 'spec', 'auc']:
            log[metric].append(epoch_log[metric])
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        output_all.extend(y_pred_fold)
        lower_bound, upper_bound = calculate_auc_ci(epoch_y['true'], epoch_y['pred'])
        log['auc_lower'].append(lower_bound)
        log['auc_upper'].append(upper_bound)
        
        val_image, val_label = validation_case_rows(val_dataloader, epoch_y)
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in val_image])
        csv_labels.extend([i[0] for i in val_label])
        csv_probabilities.extend([i[0] for i in epoch_y['pred']])
        csv_predictions.extend(y_pred_fold.ravel().astype(int).tolist())
        csv_centers.extend([c] * len(val_image))
        csv_thresholds.extend([thresholds[c]] * len(val_image))
        
    log['acc'].append(accuracy_score(y_all, output_all))
    log['auc'].append(roc_auc_score(y_all, pred_all))
    tn, fp, fn, tp = confusion_matrix(y_all, output_all, labels=[0, 1]).ravel()
    log['sens'].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    log['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    lower_bound, upper_bound = calculate_auc_ci(y_all, pred_all)
    log['auc_lower'].append(lower_bound)
    log['auc_upper'].append(upper_bound)
    
    metadata = get_pair_table(args.data_path)[["ID", "Risk Assessment"]].copy()
    if metadata["ID"].duplicated().any():
        raise ValueError("Duplicate case IDs in risk metadata.")
    risk_by_id = metadata.set_index("ID")["Risk Assessment"].to_dict()
    order = {name: index for index, name in enumerate(metadata["ID"])}
    if not set(csv_images).issubset(order):
        raise ValueError("Validation IDs are missing from risk metadata.")
    # Stable workbook order; retain separate rows for each model's validation set.
    indices = sorted(range(len(csv_images)), key=lambda i: order[csv_images[i]])
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities[i] for i in indices]
    csv_predictions = [csv_predictions[i] for i in indices]
    csv_centers = [csv_centers[i] for i in indices]
    csv_thresholds = [csv_thresholds[i] for i in indices]
    risks = np.asarray([risk_by_id[name] for name in csv_images], dtype=np.float32)
    if not np.array_equal((risks // 2).astype(int), np.asarray(csv_labels).astype(int)):
        raise ValueError("Risk metadata differs from validation labels.")

    if args.calibrate:
        df = pd.DataFrame({
        'ID': pd.Series(csv_images),
        'Risk Assessment': pd.Series(risks),
        'Label': pd.Series(csv_labels),
        'Prediction': pd.Series(csv_predictions),
        'Probability': pd.Series(csv_probabilities),
        'Held-out Center': pd.Series(csv_centers),
        'Thresholds': pd.Series(csv_thresholds)
    })
        write_validation_excel(df, os.path.join(args.output_dir, 'val_result_calibrated.xlsx'))
        
    else:
        df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability': csv_probabilities,
        'Held-out Center': csv_centers,
    })
        write_validation_excel(df, os.path.join(args.output_dir, 'val_result.xlsx'))
    
    for c in range(n_center):
        print_result(f"Dataset {c} validation loss {log['loss'][c]:.4f} acc {log['acc'][c]:.4f} sens {log['sens'][c]:.4f} spec {log['spec'][c]:.4f} auc {log['auc'][c]:.4f} 95%auc ({log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f})")
    
    print_result(f"Global validation acc {log['acc'][-1]:.4f} sens {log['sens'][-1]:.4f} spec {log['spec'][-1]:.4f} auc {log['auc'][-1]:.4f} ({np.mean([log['auc'][i] for i in range(n_center)])*100:.2f}) 95%auc ({log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f})")
    
    for c in range(n_center): # print for latex
        print_result(f"{c+1} {log['auc'][c]*100:.2f} & [{log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f}] & {log['acc'][c]*100:.2f} & {log['sens'][c]*100:.2f} & {log['spec'][c]*100:.2f}")
    c += 1
    print_result(f"{c+1} {log['auc'][c]*100:.2f}({np.mean([log['auc'][i] for i in range(n_center)])*100:.2f}) & [{log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f}] & {log['acc'][c]*100:.2f} & {log['sens'][c]*100:.2f} & {log['spec'][c]*100:.2f}")

    report_name = "val_result_calibrated.txt" if args.calibrate else "val_result.txt"
    with open(os.path.join(args.output_dir, report_name), "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(result_lines) + "\n")
