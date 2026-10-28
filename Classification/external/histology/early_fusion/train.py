import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from data_loader import get_data_list
from model import get_model
from seed import seed_everything
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, RandAxisFlipd, RandRotate90d

def load_data(args, n_center=7, val_ratio=0.2, random_state=42):
    # Hold out one entire center; reserve validation only within the other six.
    if args.fold not in range(n_center):
        raise ValueError(f'Held-out center must be between 0 and {n_center-1}.')
    train_records, val_records, test_records = [], [], []
    train_transforms = Compose([
        LoadImaged(keys=["t1", "t2"]),
        EnsureChannelFirstd(keys=["t1", "t2"]),
        Resized(keys=["t1", "t2"], spatial_size=(96, 96, 96)),
        RandAxisFlipd(keys=["t1", "t2"]),
        RandRotate90d(keys=["t1", "t2"]),
    ])
    eval_transforms = Compose([
        LoadImaged(keys=["t1", "t2"]),
        EnsureChannelFirstd(keys=["t1", "t2"]),
        Resized(keys=["t1", "t2"], spatial_size=(96, 96, 96)),
    ])
    for center in range(n_center):
        image1, image2, labels = get_data_list(root=args.data_path, center=center)
        if center == args.fold:
            testing = [{"t1": a, "t2": b, "label": y} for a, b, y in zip(image1, image2, labels)]
            test_records.extend(testing)
            print(f"Center {center+1}: {len(testing)} external test pairs (held out)")
            continue
        source1, source2, source_labels = image1, image2, labels
        records = [{"t1": a, "t2": b, "label": y} for a, b, y in zip(source1, source2, source_labels)]
        if len(records) < 2:
            training, validation = records, []
        else:
            classes, counts = np.unique(np.asarray(source_labels).ravel(), return_counts=True)
            can_stratify = len(classes) > 1 and counts.min() >= 2
            # Stratification also requires at least one example per class in each partition.
            n_val = int(np.ceil(len(records) * val_ratio))
            can_stratify = can_stratify and n_val >= len(classes) and len(records)-n_val >= len(classes)
            training, validation = train_test_split(
                records, test_size=val_ratio, random_state=random_state, shuffle=True,
                stratify=np.asarray(source_labels).ravel() if can_stratify else None)
        testing = []
        train_records.extend(training)
        val_records.extend(validation)
        test_records.extend(testing)
        print(f"Center {center+1}: {len(training)} train | {len(validation)} val | {len(testing)} test | "
              f"{len(training)+len(validation)+len(testing)} total pairs")
    print(f"Total: {len(train_records)} train | {len(val_records)} val | {len(test_records)} test pairs")
    if not train_records or not val_records or not test_records:
        raise ValueError('Training, validation and held-out test must all contain paired cases.')
    train_ds = Dataset(train_records, transform=train_transforms)
    val_ds = Dataset(val_records, transform=eval_transforms)
    test_ds = Dataset(test_records, transform=eval_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, val_loader, test_loader

def train_fn(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (batch_data) in enumerate(progress_bar):
        X1 = batch_data["t1"]
        X2 = batch_data["t2"]
        y = batch_data["label"]
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        pred = model([X1, X2])
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct = ((pred>0) == (y>=0.5)).type(torch.float).sum().item()
        total_correct += correct
        batch_count += 1
        sample_count += len(y)
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(y):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")   
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count}

def test_fn(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    y_all = []
    pred_all = []
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Testing")
        for batch_data in progress_bar:
            X1 = batch_data["t1"]
            X2 = batch_data["t2"]
            y = batch_data["label"]
            y_all.extend(y.detach().cpu().numpy())
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            pred = model([X1, X2])
            pred_all.extend(torch.sigmoid(pred).cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = ((pred>0) == (y>=0.5)).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(y)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(y):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    auc_score = roc_auc_score(y_all, pred_all) if len(np.unique(y_all)) > 1 else float("nan")
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count, 'auc': auc_score}, {'true': y_all, 'pred': pred_all}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired T1/T2 histology leave-one-center-out external training.")
    parser.add_argument("--data-path", default="/dataset/Cyst-X-resampled/classification", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("-f", "--fold", default=0, type=int, help="held-out center index (0 to 6)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 'fold'+str(args.fold))

    if args.seed is not None:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    n_center = 7
    train_dataloader, val_dataloader, _ = load_data(args, n_center=n_center)
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    log = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_auc':[]}    
    best_ind = {'loss': 0, 'acc': 0, 'auc': 0}
    for epoch in range(args.epochs):
        epoch_log = train_fn(train_dataloader, model, loss_fn, optimizer, device)
        for metric in ['loss', 'acc']:
            log['train_'+metric].append(epoch_log[metric])
        scheduler.step()
        y_all = []
        pred_all = []
        epoch_log, _ = test_fn(val_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log['val_'+metric].append(epoch_log[metric])
        print(f"Epoch {epoch+1} train loss {log['train_loss'][-1]:.4f} acc {log['train_acc'][-1]:.4f} val loss {log['val_loss'][-1]:.4f} acc {log['val_acc'][-1]:.4f} auc {log['val_auc'][-1]:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
        if log['val_loss'][-1] <= min(log['val_loss']):    
            best_ind['loss'] = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_loss.pth"))
        for metric, metric2 in (['acc', 'auc'], ['auc', 'acc']): # save model when metric improves or both metric and metric2 are maximized. The second condition avoids one best again but another worse
            if epoch == 0 or log['val_'+metric][-1] > max(log['val_'+metric][:-1]) or log['val_'+metric][-1] == max(log['val_'+metric][:-1]) and log['val_'+metric2][-1] >= max(log['val_'+metric2][:-1]):
                best_ind[metric] = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_'+metric+'.pth'))
    
    for metric in ['acc', 'auc']:
        print(f"{metric} best model reached at epoch {best_ind[metric]+1}")
        print(f"val loss {log['val_loss'][best_ind[metric]]:.4f} acc {log['val_acc'][best_ind[metric]]:.4f} auc {log['val_auc'][best_ind[metric]]:.4f}")
