import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from train import load_data, test_fn
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
    y_all = []
    pred_all = []
    log = [{'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]} for j in range(n_fold)]   
    csv_images = []
    csv_labels = []
    csv_folds = []
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   

        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log[fold]['test_'+metric].append(epoch_log[metric])  
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        
        image_list, label_list = get_data_list(root=args.data_path, t = args.t)
        _, _, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i for i in test_label])
        csv_folds.extend([fold for i in range(len(epoch_y['pred']))])

        
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


    csv_probabilities = pred_all
    csv_predictions = [i.argmax() for i in pred_all]
    df = pd.read_excel(os.path.join(args.data_path, 'IPMN_labels_t'+str(args.t)+'_total.xlsx'), usecols=[0, 5])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = [i.replace('.nii.gz', '') for i in df_cleaned.iloc[:, 0].values]
    risks =  df_cleaned.iloc[:, 1].to_numpy(dtype=np.float32)
    mapping = {value: i for i, value in enumerate(csv_images)}
    indices = [mapping[value] for value in names]
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities for i in indices]
    df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability No Risk': csv_probabilities[0],
        'Probability Low Risk': csv_probabilities[1],
        'Probability High Risk': csv_probabilities[2],
    })
    df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'result.xlsx'), index=False)