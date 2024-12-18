import os
import argparse
import json
from tqdm import tqdm
import torch
import numpy as np
from monai import transforms
from monai.losses import DiceCELoss
from data_loader import get_data_list, get_fold
from seed import seed_everything
from monai.data import DataLoader, Dataset
from synergynet import SynVNet_8h2s
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

def train_fn(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.
    progress_bar = tqdm(loader, desc="Training")
    batch_count = 0
    for data in progress_bar:
        x = data['image'].to(device)
        y = data['label'].to(device)
        optimizer.zero_grad()
        # y_hat = model(x)
        # loss = criterion(y_hat, y)
        y_hat, l = model(x)
        loss = criterion(y_hat, y) + l
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}")
    return total_loss/batch_count

def test_fn(model, loader, device, roi):
    model.eval()
    total_dice = 0.0
    batch_count = 0
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.0)
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for data in progress_bar:
            x = data['image'].to(device)
            y = data['label'].to(device)
            y_hat = sliding_window_inference(
                inputs=x,
                roi_size=roi,
                sw_batch_size=8,
                predictor=model.forward,
                overlap=0.5,
                mode="gaussian"
            )
            y_hat = post_pred(y_hat)
            dice_score = dice_metric(y_pred=y_hat, y=y).item()
            total_dice += dice_score
            batch_count += 1
            progress_bar.set_postfix(dice=f"{dice_score:.4f}", dice_avg = f"{total_dice/batch_count:.4f}")
    return total_dice/batch_count
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/data/pky0507/IPMN_images_masks/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--t", default=1, type=int, help="Modalities (1 or 2)")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--val-freq", default=10, type=int, help="validation frequency")
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 4)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t), 'fold'+str(args.fold))
    
    if args.seed:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    n_center = 7
    train_ds = []
    test_ds = []
    roi = (128, 128, 16)
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),       # Load images and labels
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channel-first format
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),  # Pad to ROI size
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],              
            label_key="label",                    
            spatial_size=roi,            
            pos=0.7,                             
            neg=0.3,                              
            num_samples=8                        
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channel-first format
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    for c in range(n_center):
        image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
        train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
        print(f"Center {c+1} has {len(train_image)} training images and {len(test_image)} testing images")
        train_ds.append(Dataset(data=[{'image': image, 'label': label} for image, label in zip(train_image, train_label)], transform=train_transforms))
        test_ds.append(Dataset(data=[{'image': image, 'label': label} for image, label in zip(test_image, test_label)], transform=test_transforms))
    train_dataloader = DataLoader(torch.utils.data.ConcatDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = []
    for c in range(n_center):
        test_dataloader.append(DataLoader(test_ds[c], batch_size=1, shuffle=False, num_workers=0))
    n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
    n_test_ds = sum([len(test_ds[i]) for i in range(n_center)]) 
    
    model = SynVNet_8h2s().to(device)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    log = {'train_loss':[], 'test_dice':[[] for i in range(n_center+1)]}
    test_dice_max = 0.0
    for epoch in range(1, args.epochs+1):
        log['train_loss'].append(train_fn(model, optimizer, criterion, train_dataloader, device))
        scheduler.step()
        if epoch % args.val_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, str(epoch)+".pth"))
            for i in range(n_center):
                log['test_dice'][i].append(test_fn(model, test_dataloader[i], device, roi=roi))
            log['test_dice'][-1].append(sum([log['test_dice'][i][-1]*len(test_dataloader[i]) for i in range(n_center)])/n_test_dataloader)
            print(f"Epoch {epoch} global train loss {log['train_loss'][-1]:.4f} test dice {log['test_dice'][-1][-1]:.4f}")
            if log['test_dice'][-1][-1] == log['test_dice'][-1][-1] and log['test_dice'][-1][-1] >= test_dice_max: #If dice is not NaN but maximized
                test_dice_max = log['test_dice'][-1][-1]
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model_dice.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
    print(f"Best test dice {test_dice_max} reached at epoch {args.val_freq*np.argmax(log['test_dice'][-1])+1}")