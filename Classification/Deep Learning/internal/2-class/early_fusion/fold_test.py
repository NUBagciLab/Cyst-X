import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from sklearn.metrics import roc_curve, auc
from train import load_data, test_fn
import matplotlib.pyplot as plt
from data_loader import get_data_list, get_fold
import pandas as pd

def highlight_errors(row):
    # If Prediction != Label, color the row light red
    if row['Label'] != row['Prediction']:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model)
    
    device = torch.device(args.device)
            
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    n_center = 7
    n_fold = 5
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    log = [{'test_loss':[], 'test_acc':[], 'test_auc':[]} for j in range(n_fold)]   
    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_folds = []
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log[fold]['test_'+metric].append(epoch_log[metric])
        y_all = epoch_y['true']
        pred_all = epoch_y['pred']
        
        test_image = []
        test_label = []       
        for c in range(n_center):
            image1_list_c, image2_list_c, label_list_c = get_data_list(root=args.data_path, center=c)
            _, _, _, test_image1_c, _, test_label_c = get_fold(image1_list_c, image2_list_c, label_list_c, fold = args.fold)
            test_image.extend(test_image1_c)
            test_label.extend(test_label_c)
            
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i[0] for i in test_label])
        csv_probabilities.extend([i[0] for i in epoch_y['pred']])
        csv_folds.extend([fold for i in range(len(epoch_y['pred']))])
        
        fpr, tpr, _ = roc_curve(y_all, pred_all)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPRs
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} ROC (AUC={roc_auc:.4f})')
    
    df = pd.read_excel(os.path.join(args.data_path, 'IPMN_labels_total.xlsx'),  usecols=[0, 1, 6])
    df_cleaned = df.dropna(subset=[df.columns[2]]) # remove NaN
    names = [i.replace('.nii.gz', '') for i in df_cleaned.iloc[:, 0].values]
    risks =  df_cleaned.iloc[:, 2].to_numpy(dtype=np.float32)
    mapping = {value: i for i, value in enumerate(csv_images)}
    indices = [mapping[value] for value in names]
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities[i] for i in indices]
    csv_folds = [csv_folds[i] for i in indices]
    csv_predictions = [int(i>=0.5) for i in csv_probabilities]
    
    df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability': csv_probabilities,
        'Fold': csv_folds
    })
    df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result.xlsx'), index=False)
        
    for fold in range(n_fold): 
        print(f"Fold {fold} test loss {log[fold]['test_loss'][-1]:.4f} acc {log[fold]['test_acc'][-1]:.4f} auc {log[fold]['test_auc'][-1]:.4f}")
    log_mean = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)]}   
    log_std = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)]}   

    for metric in ['loss', 'acc', 'auc']:
        log_mean['test_'+metric] = np.mean([log[fold]['test_'+metric][-1] for fold in range(n_fold)])
        log_std['test_'+metric] = np.std([log[fold]['test_'+metric][-1] for fold in range(n_fold)])
    print(f"Global test loss {log_mean['test_loss']:.4f}±{log_std['test_loss']:.4f} acc {log_mean['test_acc']:.4f}±{log_std['test_acc']:.4f} auc {log_mean['test_auc']:.4f}±{log_std['test_auc']:.4f}")

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