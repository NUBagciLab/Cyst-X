import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from train import load_data, test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="../../../preprocessing/PanSegNet/", type=str, help="dataset path")
    # parser.add_argument("--data-path", default="../../../preprocessing/Swin-UNETR/", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 't'+str(args.t))
    
    device = torch.device(args.device)
    
    model = get_model(name = args.model, num_classes = 3)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    n_center = 7
    n_fold = 5
    log = [{'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]} for j in range(n_fold)]   

    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   
        
        epoch_log, _ = test_fn(test_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log[fold]['test_'+metric].append(epoch_log[metric])        
        
    for fold in range(n_fold): 
        print(f"Fold {fold} test loss {log[fold]['test_loss'][-1]:.4f} acc {log[fold]['test_acc'][-1]:.4f} auc {log[fold]['test_auc'][-1]:.4f}")
    log_mean = {'test_loss':0, 'test_acc':0, 'test_auc':0}   
    log_std = {'test_loss':0, 'test_acc':0, 'test_auc':0}   
    for metric in ['loss', 'acc', 'auc']:
        log_mean['test_'+metric] = np.mean([log[fold]['test_'+metric][-1] for fold in range(n_fold)])
        log_std['test_'+metric] = np.std([log[fold]['test_'+metric][-1] for fold in range(n_fold)])
    print(f"Test loss {log_mean['test_loss']:.4f}±{log_std['test_loss']:.4f} acc {log_mean['test_acc']:.4f}±{log_std['test_acc']:.4f} auc {log_mean['test_auc']:.4f}±{log_std['test_auc']:.4f}")
   
    ci95 = 1.96 * log_std['test_auc'] / np.sqrt(n_fold)
    log_mean['auc_lower'] = log_mean['test_auc'] - ci95
    log_mean['auc_upper'] = log_mean['test_auc'] + ci95
    print(f"95%CI: [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}]")
    
    print(f"{log_mean['test_acc']*100:.2f}$\\pm${log_std['test_acc']*100:.2f} & {log_mean['test_auc']*100:.2f}$\\pm${log_std['test_auc']*100:.2f} & [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}]")