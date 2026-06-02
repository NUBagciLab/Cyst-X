import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from sklearn.metrics import roc_auc_score, roc_curve, auc
from train import load_data, test_fn
import matplotlib.pyplot as plt
from data_loader import get_data_list, get_fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="../../../preprocessing/PanSegNet/", type=str, help="dataset path")
    # parser.add_argument("--data-path", default="../../../preprocessing/Swin-UNETR/", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 't'+str(args.t))
    
    device = torch.device(args.device)
            
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    n_center = 7
    n_fold = 4
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    log = [{'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]} for j in range(n_fold)]   
    csv_images = []
    csv_labels = []
    csv_probabilities = []
    csv_folds = []
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args, n_center=n_center)
        test_ds = [test_dataloader[i].dataset for i in range(n_center)]
        n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
        n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   
        y_all = []
        pred_all = []
        for c in range(n_center):
            epoch_log, epoch_y = test_fn(test_dataloader[c], model, loss_fn, device)
            for metric in ['loss', 'acc', 'auc']:
                log[fold]['test_'+metric][c].append(epoch_log[metric])
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
            
            image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
            _, _, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
            csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
            csv_labels.extend([i[0] for i in test_label])
            csv_probabilities.extend([i[0] for i in epoch_y['pred']])
            csv_folds.extend([fold for i in range(len(epoch_y['pred']))])
        
        log[fold]['test_loss'][-1].append(sum([log[fold]['test_loss'][i][-1]*len(test_dataloader[i]) for i in range(n_center)])/n_test_dataloader)
        log[fold]['test_acc'][-1].append(sum([log[fold]['test_acc'][i][-1]*len(test_ds[i]) for i in range(n_center)])/n_test_ds)
        log[fold]['test_auc'][-1].append(roc_auc_score(y_all, pred_all))
        
        fpr, tpr, _ = roc_curve(y_all, pred_all)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPRs
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} ROC (AUC={roc_auc:.4f})')

    for fold in range(n_fold): 
        print(f"Fold {fold}")
        for c in range(n_center):
            print(f"Center {c+1} test loss {log[fold]['test_loss'][c][-1]:.4f} acc {log[fold]['test_acc'][c][-1]:.4f} auc {log[fold]['test_auc'][c][-1]:.4f}")
        print(f"Global test loss {log[fold]['test_loss'][-1][-1]:.4f} acc {log[fold]['test_acc'][-1][-1]:.4f} auc {log[fold]['test_auc'][-1][-1]:.4f}")
    log_mean = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)], 'test_auc_lower':[0 for i in range(n_center+1)], 'test_auc_upper':[0 for i in range(n_center+1)]}   
    log_std = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)]}   
    for c in range(n_center+1):
        for metric in ['loss', 'acc', 'auc']:
            log_mean['test_'+metric][c] = np.mean([log[fold]['test_'+metric][c][-1] for fold in range(n_fold)])
            log_std['test_'+metric][c] = np.std([log[fold]['test_'+metric][c][-1] for fold in range(n_fold)])
        ci95 = 1.96 * log_std['test_auc'][c] / np.sqrt(n_fold)
        log_mean['test_auc_lower'][c] = log_mean['test_auc'][c] - ci95
        log_mean['test_auc_upper'][c] = log_mean['test_auc'][c] + ci95
        if c < n_center:
            print(f"Center {c+1} test loss {log_mean['test_loss'][c]:.4f}±{log_std['test_loss'][c]:.4f} acc {log_mean['test_acc'][c]:.4f}±{log_std['test_acc'][c]:.4f} auc {log_mean['test_auc'][c]:.4f}±{log_std['test_auc'][c]:.4f} 95%CI [{log_mean['test_auc_lower'][c]:.4f}, {log_mean['test_auc_upper'][c]:.4f}]")
        else: 
            print(f"Global test loss {log_mean['test_loss'][c]:.4f}±{log_std['test_loss'][c]:.4f} acc {log_mean['test_acc'][c]:.4f}±{log_std['test_acc'][c]:.4f} auc {log_mean['test_auc'][c]:.4f}±{log_std['test_auc'][c]:.4f} 95%CI [{log_mean['test_auc_lower'][c]:.4f}, {log_mean['test_auc_upper'][c]:.4f}]")
    for c in range(n_center+1): # print for latex
        print(f"{c+1} {log_mean['test_acc'][c]*100:.2f}$\\pm${log_std['test_acc'][c]*100:.2f} & {log_mean['test_auc'][c]*100:.2f}$\\pm${log_std['test_auc'][c]*100:.2f} & [{log_mean['test_auc_lower'][c]*100:.2f}, {log_mean['test_auc_upper'][c]*100:.2f}]")