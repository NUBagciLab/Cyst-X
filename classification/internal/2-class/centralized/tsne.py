import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from model import get_model
from sklearn.metrics import roc_auc_score, roc_curve, auc
from train import load_data, test_fn
import matplotlib.pyplot as plt
from data_loader import get_data_list, get_fold
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def test_fn(dataloader, model, loss_fn, device):
    model.eval()
    input_to_classifier = {}
    # Hook function to capture input
    def hook_fn(module, input, output):
        input_to_classifier['classifier_input'] = input[0].detach()
    
    hook = model.class_layers[-1].register_forward_hook(hook_fn)
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    y_all = []
    pred_all = []
    hidden_state = []
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Testing")
        for X, y in progress_bar:
            y_all.extend(y)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_all.extend(torch.sigmoid(pred).cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = ((pred>0) == (y>=0.5)).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(X)
            hidden_state.extend(input_to_classifier['classifier_input'].cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    auc_score = roc_auc_score(y_all, pred_all)
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count, 'auc': auc_score}, {'true': y_all, 'pred': pred_all}, hidden_state

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
    
    device = torch.device(args.device)
            
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    n_center = 7
    n_fold = 4
    
    y_all = []
    pred_all = []
    hidden_state_all = []
    
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args, n_center=n_center)
        test_ds = [test_dataloader[i].dataset for i in range(n_center)]
        n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
        n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   

        for c in range(n_center):
            epoch_log, epoch_y, hidden_state = test_fn(test_dataloader[c], model, loss_fn, device)
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
            hidden_state_all.extend(hidden_state)

    hidden_state_all = np.stack(hidden_state_all, axis=0)
    y_all = np.array([y.cpu().numpy().squeeze() for y in y_all], dtype=int)
    X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(hidden_state_all)

    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_all, palette='tab10', legend='full', s=60)
    # Create custom legend
    plt.legend(handles=[
        mpatches.Patch(color='C0', label='IPMN no/low-risk'),
        mpatches.Patch(color='C1', label='IPMN high-risk'),
    ], loc='upper left')
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(X_tsne, y_all):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    plt.grid()
    # plt.show()
    plt.savefig(args.model+'_'+'t'+str(args.t)+"tsne.pdf", format="pdf", bbox_inches='tight')
    print(f'Silhouette Coefficient: {silhouette_score(X_tsne, y_all):.4f}')
    print(f'Calinski-Harabasz index: {calinski_harabasz_score(X_tsne, y_all):.4f}')
    print(f'Davies-Bouldin index: {davies_bouldin_score(X_tsne, y_all):.4f}')
