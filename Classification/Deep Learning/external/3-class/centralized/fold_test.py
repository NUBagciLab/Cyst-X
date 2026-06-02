import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from seed import seed_everything
from sklearn.metrics import roc_auc_score, accuracy_score
from train import load_data, test_fn
from sklearn.utils import resample
from data_loader import get_data_list
import pandas as pd

def highlight_errors(row):
    # If Prediction != Label, color the row light red
    if row['Label'] != row['Prediction']:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

def calculate_auc_ci(y_true, y_pred_probs, n_bootstraps=1000, ci_level=0.95):
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        y_b, pred_b = resample(y_true, y_pred_probs)
        
        # Check if bootstrap sample has both classes
        if len(np.unique(y_b)) < 3:
            continue
            
        score = roc_auc_score(y_b, pred_b, average="macro", multi_class="ovr", labels=[0, 1, 2])
        bootstrapped_scores.append(score)
        
    # Calculate 95% CI
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = np.percentile(sorted_scores, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ci_level) / 2 * 100)
    
    return lower_bound, upper_bound

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 't'+str(args.t))
    
    seed_everything(42) # Fix seed for 95% AUC

    device = torch.device(args.device)
            
    model = get_model(name = args.model, num_classes = 3)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    n_center = 7
    y_all = []
    pred_all = []
    log = {'loss':[], 'acc':[], 'auc':[], 'auc_lower':[], 'auc_upper':[]}
    csv_images = []
    csv_labels = []
    for c in range(n_center):
        args.fold = c      
        _, test_dataloader = load_data(args)        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log[metric].append(epoch_log[metric])
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        if c == 4: #MCA only have two classes
            log['auc_lower'].append(float('nan'))
            log['auc_upper'].append(float('nan'))
        else:
            lower_bound, upper_bound = calculate_auc_ci(epoch_y['true'], epoch_y['pred'])
            log['auc_lower'].append(lower_bound)
            log['auc_upper'].append(upper_bound)
        
        test_image, test_label = get_data_list(root=args.data_path, t = args.t, center=c)
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i for i in test_label])
        
    log['acc'].append(accuracy_score(y_all, [i.argmax() for i in pred_all]))
    log['auc'].append(roc_auc_score(y_all, pred_all, average="macro", multi_class="ovr", labels=[0, 1, 2]))
    lower_bound, upper_bound = calculate_auc_ci(y_all, pred_all)
    log['auc_lower'].append(lower_bound)
    log['auc_upper'].append(upper_bound)
    
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
    
    for c in range(n_center):
        print(f"Dataset {c} test loss {log['loss'][c]:.4f} acc {log['acc'][c]:.4f} auc {log['auc'][c]:.4f} 95%auc ({log['auc_lower'][c]:.4f}, {log['auc_upper'][c]:.4f})")
    
    print(f"Global test acc {log['acc'][-1]:.4f} auc {log['auc'][-1]:.4f} 95%auc ({log['auc_lower'][-1]:.4f}, {log['auc_upper'][-1]:.4f})")
    
    for c in range(n_center+1): # print for latex
        print(f"{c+1} {log['acc'][c]*100:.2f} & {log['auc'][c]*100:.2f} & ({log['auc_lower'][c]*100:.2f}, {log['auc_upper'][c]*100:.2f})")