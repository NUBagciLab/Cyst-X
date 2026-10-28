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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, RandAxisFlip
import copy
from fl import fedavg, difference_models_norm_2

def load_data(args, n_center=7, val_ratio=0.2, random_state=42):
    if args.fold not in range(n_center):
        raise ValueError(f"Held-out center must be between 0 and {n_center-1}.")
    train_ds = []
    all_val_images, all_val_labels = [], []
    all_test_images, all_test_labels = [], []

    train_transforms = Compose([
        EnsureChannelFirst(), 
        Resize((96, 96, 96)), 
        RandAxisFlip(), 
        RandRotate90()
    ])
    val_test_transforms = Compose([
        EnsureChannelFirst(), 
        Resize((96, 96, 96))
    ])

    for c in range(n_center):
        image_list, label_list = get_data_list(root=args.data_path, t=args.t, center=c)
        if c == args.fold:
            all_test_images.extend(image_list)
            all_test_labels.extend(label_list)
            print(f"Center {c+1}: {len(image_list)} external test images (held out)")
            continue
        train_image, train_label = image_list, label_list

        # Flatten in case labels are shape (N, 1) or a list of 1-element arrays
        lbl_arr = np.squeeze(np.asarray(train_label))

        # Get unique classes and their sample counts
        classes, counts = np.unique(lbl_arr, return_counts=True)
        can_stratify = len(classes) > 1 and counts.min() >= 2

        tr_img, val_img, tr_lbl, val_lbl = train_test_split(
            train_image,
            train_label,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_label if can_stratify else None,
            shuffle=True
        )

        print(f"Center {c+1}: {len(tr_img)} train | {len(val_img)} val | 0 test | {len(tr_img)+len(val_img)} total")

        train_ds.append(ImageDataset(image_files=tr_img, labels=tr_lbl, transform=train_transforms))
        all_val_images.extend(val_img)
        all_val_labels.extend(val_lbl)
    n_train = sum(len(ds) for ds in train_ds)
    print(f"Total: {n_train} train | {len(all_val_images)} val | {len(all_test_images)} test | {n_train+len(all_val_images)+len(all_test_images)} total")
    val_ds = ImageDataset(image_files=all_val_images, labels=all_val_labels, transform=val_test_transforms)
    test_ds = ImageDataset(image_files=all_test_images, labels=all_test_labels, transform=val_test_transforms)

    train_dataloader = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        for ds in train_ds
    ]
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_dataloader, val_dataloader, test_dataloader

def train_fn(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct = ((pred>0) == (y>=0.5)).type(torch.float).sum().item()
        total_correct += correct
        batch_count += 1
        sample_count += len(X)
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")   
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count}

def train_fn_fedprox(dataloader, model, model_0, mu, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y) + mu/2*difference_models_norm_2(model, model_0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct = ((pred>0) == (y>=0.5)).type(torch.float).sum().item()
        total_correct += correct
        batch_count += 1
        sample_count += len(X)
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")   
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
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    auc_score = roc_auc_score(y_all, pred_all)
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count, 'auc': auc_score}, {'true': y_all, 'pred': pred_all}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN FedProx leave-one-center-out training with validation-based checkpoint selection.")
    parser.add_argument("--data-path", default="/dataset/Cyst-X-resampled/classification", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of FedProx rounds (one local epoch per center per round)")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="held-out center index (0 to 6)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    parser.add_argument("--mu", default=0.1, type=float, help="FedProx mu (default: 0.1)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.mu), 't'+str(args.t), 'fold'+str(args.fold))

    if args.seed:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    n_center = 7
    train_dataloader, val_dataloader, _ = load_data(args, n_center=n_center)
    n_center = len(train_dataloader)  # Six source-center clients; target center is excluded.
    train_ds = [train_dataloader[i].dataset for i in range(n_center)]
    n_train_dataloader = sum([len(train_dataloader[i]) for i in range(n_center)])
    n_train_ds = sum([len(train_ds[i]) for i in range(n_center)])
    
    aggregation_weights = [len(d) for d in train_ds]
    aggregation_weights = [aggregation_weights[i]/sum(aggregation_weights) for i in range(n_center)]
    
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    model_local = [copy.deepcopy(model) for c in range(n_center)]
    # The proximal reference is fixed during local training.
    model.requires_grad_(False)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer_local = [torch.optim.AdamW(model_local[i].parameters(), lr=args.lr) for i in range(n_center)]
    scheduler_local = [torch.optim.lr_scheduler.StepLR(optimizer_local[i], step_size=args.lr_step_size, gamma=args.lr_gamma) for i in range(n_center)]
    
    log = {
        'train_loss': [[] for i in range(n_center+1)],
        'train_acc': [[] for i in range(n_center+1)],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
    }
    best_ind = {'loss': 0, 'acc': 0, 'auc': 0}
    for epoch in range(args.epochs):
        for c in range(n_center):
            epoch_log = train_fn_fedprox(train_dataloader[c], model_local[c], model, args.mu, loss_fn, optimizer_local[c], device)
            for metric in ['loss', 'acc']:
                log['train_'+metric][c].append(epoch_log[metric])
            scheduler_local[c].step()
        log['train_loss'][-1].append(sum([log['train_loss'][i][-1]*len(train_dataloader[i]) for i in range(n_center)])/n_train_dataloader)
        log['train_acc'][-1].append(sum([log['train_acc'][i][-1]*len(train_ds[i]) for i in range(n_center)])/n_train_ds)
        fedavg(model, model_local, aggregation_weights)

        # Select the global model using pooled validation data only.
        epoch_log, _ = test_fn(val_dataloader, model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log['val_'+metric].append(epoch_log[metric])
        print(f"Epoch {epoch+1} train loss {log['train_loss'][-1][-1]:.4f} acc {log['train_acc'][-1][-1]:.4f} val loss {log['val_loss'][-1]:.4f} acc {log['val_acc'][-1]:.4f} auc {log['val_auc'][-1]:.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
        if log['val_loss'][-1] <= min(log['val_loss']):
            best_ind['loss'] = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_loss.pth"))
        for metric, metric2 in (['acc', 'auc'], ['auc', 'acc']):
            if epoch == 0 or log['val_'+metric][-1] > max(log['val_'+metric][:-1]) or log['val_'+metric][-1] == max(log['val_'+metric][:-1]) and log['val_'+metric2][-1] >= max(log['val_'+metric2][:-1]):
                best_ind[metric] = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_'+metric+'.pth'))

    for metric in ['acc', 'auc']:
        print(f"{metric} best model reached at epoch {best_ind[metric]+1}")
        print(f"Global val loss {log['val_loss'][best_ind[metric]]:.4f} acc {log['val_acc'][best_ind[metric]]:.4f} auc {log['val_auc'][best_ind[metric]]:.4f}")
