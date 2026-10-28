import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from sklearn.metrics import roc_curve, auc
from train import load_data, test_fn
import matplotlib.pyplot as plt
from data_loader import get_data_list, get_fold
import pandas as pd
from auc_ci import calculate_auc_ci_cv, highlight_errors
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
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-c", "--calibrate", action="store_true", help="perform calibration")
    parser.add_argument("--mu", default=0.1, type=float, help="FedProx mu (default: 0.1)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.mu), 't'+str(args.t))
    os.makedirs(args.output_dir, exist_ok=True)
    result_lines = []

    def print_result(message):
        print(message)
        result_lines.append(message)

    
    device = torch.device(args.device)
    
    n_center = 7
    n_fold = 5

    if args.calibrate:
        if not os.path.exists(os.path.join(args.output_dir, "thresholds.json")):
            print_result("Calibrating thresholds...")
            thresholds = calibrate_and_save_thresholds(args)
        else:
            print_result("Loading existing thresholds...")
            with open(os.path.join(args.output_dir, "thresholds.json"), 'r') as f:
                thresholds = json.load(f)
            for fold in range(n_fold):
                print_result(f"Fold {fold} threshold: {thresholds[fold]:.4f}")
    else:
        print_result("Calibration not requested. Using default threshold of 0.5 for all folds.")
        thresholds = [0.5 for _ in range(n_fold)]

    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    y_all = []
    pred_all = []
    folds_data = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    log = [{} for j in range(n_fold)]   
    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_predictions = []
    csv_folds = []
    for fold in range(n_fold):
        args.fold = fold
        _, _, test_dataloader = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)

        # Compute balanced accuracy and F1 score if not already provided by test_fn
        y_true_fold = np.array(epoch_y['true'])
        y_pred_fold = np.array(epoch_y['pred']) >= thresholds[fold]
        epoch_log['acc'] = accuracy_score(y_true_fold, y_pred_fold)
        tn, fp, fn, tp = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1]).ravel()
        epoch_log['sens'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        epoch_log['spec'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        for metric in ['loss', 'acc', 'sens', 'spec', 'auc']:
            log[fold]['test_'+metric] = epoch_log[metric]
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        folds_data.append((epoch_y['true'], epoch_y['pred']))
        test_image = []
        test_label = []       
        for c in range(n_center):
            image_list_c, label_list_c = get_data_list(root=args.data_path, t=args.t, center=c)
            _, _, test_image_c, test_label_c = get_fold(image=image_list_c, label=label_list_c, root=args.data_path, t=args.t, fold=fold)
            test_image.extend(test_image_c)
            test_label.extend(test_label_c)  
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i[0] for i in test_label])
        csv_probabilities.extend([i[0] for i in epoch_y['pred']])
        csv_predictions.extend(y_pred_fold.ravel().astype(int).tolist())
        csv_folds.extend([fold for i in range(len(epoch_y['pred']))])
        
        fpr, tpr, _ = roc_curve(epoch_y['true'], epoch_y['pred'])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPRs
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} ROC (AUC={roc_auc:.4f})')
        
    for fold in range(n_fold): 
        print_result(f"Fold {fold} test loss {log[fold]['test_loss']:.4f} acc {log[fold]['test_acc']:.4f} sens {log[fold]['test_sens']:.4f} spec {log[fold]['test_spec']:.4f} auc {log[fold]['test_auc']:.4f}")
            
    log_mean = {}   
    log_std = {}
    for metric in ['loss', 'acc', 'sens', 'spec', 'auc']:
        log_mean['test_'+metric] = np.mean([log[fold]['test_'+metric] for fold in range(n_fold)])
        log_std['test_'+metric] = np.std([log[fold]['test_'+metric] for fold in range(n_fold)])
    print_result(f"Test loss {log_mean['test_loss']:.4f}±{log_std['test_loss']:.4f} acc {log_mean['test_acc']:.4f}±{log_std['test_acc']:.4f} sens {log_mean['test_sens']:.4f}±{log_std['test_sens']:.4f} spec {log_mean['test_spec']:.4f}±{log_std['test_spec']:.4f} auc {log_mean['test_auc']:.4f}±{log_std['test_auc']:.4f}")
    
    lower_bound, upper_bound = calculate_auc_ci_cv(folds_data)
    log_mean['auc_lower'] = lower_bound
    log_mean['auc_upper'] = upper_bound
    print_result(f"95%CI: [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}]")

    print_result(f"{log_mean['test_auc']*100:.2f}$\\pm${log_std['test_auc']*100:.2f} & [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}] & {log_mean['test_acc']*100:.2f}$\\pm${log_std['test_acc']*100:.2f} & {log_mean['test_sens']*100:.2f}$\\pm${log_std['test_sens']*100:.2f} & {log_mean['test_spec']*100:.2f}$\\pm${log_std['test_spec']*100:.2f}")
    
    report_name = "result_calibrated.txt" if args.calibrate else "result.txt"
    with open(os.path.join(args.output_dir, report_name), "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(result_lines) + "\n")

    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    mean_auc = log_mean['test_auc']
    std_auc = log_std['test_auc']
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})',
             lw=2, alpha=0.8)
    
    # Plot std deviation
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                     label='±1 std. dev.')
    
    # Add plot details
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.title("Mean ROC Curve on Dual Modalities")
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(os.path.join(args.output_dir, "roc.pdf"), format="pdf", bbox_inches='tight')

    df = pd.read_excel(os.path.join(args.data_path, 'IPMN_labels_t'+str(args.t)+'_total.xlsx'), usecols=[0, 5])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = [i.replace('.nii.gz', '') for i in df_cleaned.iloc[:, 0].values]
    risks =  df_cleaned.iloc[:, 1].to_numpy(dtype=np.float32)
    mapping = {value: i for i, value in enumerate(csv_images)}
    indices = [mapping[value] for value in names]
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities[i] for i in indices]
    csv_folds = [csv_folds[i] for i in indices]
    csv_predictions = [csv_predictions[i] for i in indices]

    if args.calibrate:
        plt.savefig(os.path.join(args.output_dir, "roc_calibrated.pdf"), format="pdf", bbox_inches='tight')
        df = pd.DataFrame({
        'ID': pd.Series(csv_images),
        'Risk Assessment': pd.Series(risks),
        'Label': pd.Series(csv_labels),
        'Prediction': pd.Series(csv_predictions),
        'Probability': pd.Series(csv_probabilities),
        'Fold': pd.Series(csv_folds),
        'Thresholds': pd.Series(thresholds)
    })
        df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result_calibrated.xlsx'), index=False)
        
    else:
        plt.savefig(os.path.join(args.output_dir, "roc.pdf"), format="pdf", bbox_inches='tight')
        df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability': csv_probabilities,
        'Fold': csv_folds
    })
        df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result.xlsx'), index=False)