import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from data_loader import get_data_list, get_fold
from model import get_model
from sklearn.metrics import roc_auc_score, roc_curve, auc
from monai.data import DataLoader, ImageDataset
from monai.transforms import Resize, EnsureChannelFirst, Compose, ScaleIntensity
from train import test_fn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t))
    
    device = torch.device(args.device)
    n_center = 7
    n_fold = 4
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    log = [{'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]} for j in range(n_fold)]   

    for fold in range(n_fold):
        image_lists = []
        label_lists = []
        test_ds = []
        test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
        for c in range(n_center):
            image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
            _, _, test_image, test_label = get_fold(image_list, label_list, fold = fold)
            print(f"Center {c+1} has {len(test_image)} testing images")
            test_ds.append(ImageDataset(image_files=test_image, labels=test_label, transform=test_transforms))
        test_dataloader = []
        for c in range(n_center):
            test_dataloader.append(DataLoader(test_ds[c], batch_size=args.batch_size, shuffle=False, num_workers=args.workers))
        n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
        n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])
    
        model = get_model(out_channels = 2)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        
        y_all = []
        pred_all = []
        for c in range(n_center):
            epoch_log, epoch_y = test_fn(test_dataloader[c], model, loss_fn, device)
            for metric in ['loss', 'acc', 'auc']:
                log[fold]['test_'+metric][c].append(epoch_log[metric])
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
        
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
    log_mean = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)]}   
    log_std = {'test_loss':[0 for i in range(n_center+1)], 'test_acc':[0 for i in range(n_center+1)], 'test_auc':[0 for i in range(n_center+1)]}   
    for c in range(n_center+1):
        for metric in ['loss', 'acc', 'auc']:
            log_mean['test_'+metric][c] = np.mean([log[fold]['test_'+metric][c][-1] for fold in range(n_fold)])
            log_std['test_'+metric][c] = np.std([log[fold]['test_'+metric][c][-1] for fold in range(n_fold)])
        if c < n_center:
            print(f"Center {c+1} test loss {log_mean['test_loss'][c]:.4f}±{log_std['test_loss'][c]:.4f} acc {log_mean['test_acc'][c]:.4f}±{log_std['test_acc'][c]:.4f} auc {log_mean['test_auc'][c]:.4f}±{log_std['test_auc'][c]:.4f}")
        else: 
            print(f"Global test loss {log_mean['test_loss'][c]:.4f}±{log_std['test_loss'][c]:.4f} acc {log_mean['test_acc'][c]:.4f}±{log_std['test_acc'][c]:.4f} auc {log_mean['test_auc'][c]:.4f}±{log_std['test_auc'][c]:.4f}")
    for c in range(n_center+1): # print for latex
        print(f"{c+1} {log_mean['test_acc'][c]:.4f}$\\pm${log_std['test_acc'][c]:.4f} & {log_mean['test_auc'][c]:.4f}$\\pm${log_std['test_auc'][c]:.4f}")

    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    mean_auc = log_mean['test_auc'][-1]
    std_auc = log_std['test_auc'][-1]
    
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
    plt.title("Mean ROC Curve on T"+str(args.t)+" Modality")
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig("roc"+str(args.t)+".pdf", format="pdf", bbox_inches='tight')