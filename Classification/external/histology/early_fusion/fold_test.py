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
from calibrate import calibrate_and_save_thresholds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/Cyst-X-resampled/classification", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("-c", "--calibrate", action="store_true", help="perform calibration")
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
        if not os.path.exists(os.path.join(args.output_dir, "thresholds.json")):
            print_result("Calibrating thresholds...")
            thresholds = calibrate_and_save_thresholds(args)
        else:
            print_result("Loading existing thresholds...")
            with open(os.path.join(args.output_dir, "thresholds.json"), 'r') as f:
                thresholds = json.load(f)
            for fold in range(n_center):
                print_result(f"Fold {fold} threshold: {thresholds[fold]:.4f}")
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
        _, _, test_dataloader = load_data(args)        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)

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
        
        records = test_dataloader.dataset.data
        test_image = [record['t1'] for record in records]
        test_label = [record['label'] for record in records]
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i[0] for i in test_label])
        csv_probabilities.extend([i[0] for i in epoch_y['pred']])
        csv_predictions.extend(y_pred_fold.ravel().astype(int).tolist())
        csv_centers.extend([c] * len(test_image))
        csv_thresholds.extend([thresholds[c]] * len(test_image))
        
    log['acc'].append(accuracy_score(y_all, output_all))
    log['auc'].append(roc_auc_score(y_all, pred_all))
    tn, fp, fn, tp = confusion_matrix(y_all, output_all, labels=[0, 1]).ravel()
    log['sens'].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    log['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    lower_bound, upper_bound = calculate_auc_ci(y_all, pred_all)
    log['auc_lower'].append(lower_bound)
    log['auc_upper'].append(upper_bound)
    
    pairs = get_pair_table(args.data_path)
    mapping = {value: i for i, value in enumerate(csv_images)}
    pairs = pairs[pairs['ID'].isin(mapping)]
    indices = [mapping[value] for value in pairs['ID']]
    risks = pairs['Risk Assessment'].to_numpy(dtype=np.float32)
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities[i] for i in indices]
    csv_predictions = [csv_predictions[i] for i in indices]
    csv_centers = [csv_centers[i] for i in indices]
    csv_thresholds = [csv_thresholds[i] for i in indices]

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
        df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result_calibrated.xlsx'), index=False)
        
    else:
        df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability': csv_probabilities,
        'Held-out Center': csv_centers,
    })
        df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result.xlsx'), index=False)
    
    for c in range(n_center):
        print_result(f"Dataset {c} test loss {log['loss'][c]:.4f} acc {log['acc'][c]:.4f} sens {log['sens'][c]:.4f} spec {log['spec'][c]:.4f} auc {log['auc'][c]:.4f} 95%auc ({log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f})")
    
    print_result(f"Global test acc {log['acc'][-1]:.4f} sens {log['sens'][-1]:.4f} spec {log['spec'][-1]:.4f} auc {log['auc'][-1]:.4f} ({np.mean([log['auc'][i] for i in range(n_center)])*100:.2f}) 95%auc ({log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f})")
    
    for c in range(n_center): # print for latex
        print_result(f"{c+1} {log['auc'][c]*100:.2f} & [{log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f}] & {log['acc'][c]*100:.2f} & {log['sens'][c]*100:.2f} & {log['spec'][c]*100:.2f}")
    c += 1
    print_result(f"{c+1} {log['auc'][c]*100:.2f}({np.mean([log['auc'][i] for i in range(n_center)])*100:.2f}) & [{log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f}] & {log['acc'][c]*100:.2f} & {log['sens'][c]*100:.2f} & {log['spec'][c]*100:.2f}")

    report_name = "result_calibrated.txt" if args.calibrate else "result.txt"
    with open(os.path.join(args.output_dir, report_name), "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(result_lines) + "\n")
