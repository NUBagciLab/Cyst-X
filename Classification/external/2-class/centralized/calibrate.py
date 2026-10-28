# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:21:54 2026

@author: pky0507
"""
import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from train import load_data, test_fn
from sklearn.metrics import confusion_matrix

def calibrate(probabilities, labels):
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    threshold_list = sorted([0]+list(set(probabilities))+[1])
    for threshold in threshold_list:
        output = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, output).ravel()
        spec = tn / (tn + fp)
        if spec >= 0.9:
            return float(threshold)

    print("No valid threshold found. Defaulting to 0.5.")
    return 0.5


def calibrate_and_save_thresholds(args):
    device = torch.device(args.device)
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    n_center = 7
    n_fold = n_center
    thresholds = []
    for fold in range(n_fold):
        args.fold = fold
        _, val_dataloader, _ = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   
        _, epoch_y = test_fn(val_dataloader, model, loss_fn, device)
        thresholds.append(calibrate(np.array(epoch_y['pred']).squeeze(), np.array(epoch_y['true']).squeeze()))
    for fold in range(n_fold):
        print(f"Fold {fold} threshold: {thresholds[fold]:.4f}")
    with open(os.path.join(args.output_dir, "thresholds.json"), 'w') as f:
        json.dump(thresholds, f)
    return thresholds

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
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 't'+str(args.t))
    calibrate_and_save_thresholds(args)
  