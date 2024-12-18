import os
import argparse
import torch
import numpy as np
from data_loader import get_data_list, get_fold
from monai import transforms
from monai.data import DataLoader, Dataset
from synergynet import SynVNet_8h2s
from test import test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg cross validation test.")
    parser.add_argument("--data-path", default="/data/pky0507/IPMN_images_masks/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--t", default=1, type=int, help="Modalities (1 or 2)")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 8)")
    parser.add_argument("--resume", default="model_dice.pth", type=str, help="path of checkpoint")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t))
    
    device = torch.device(args.device)
    n_center = 7
    n_fold = 5
    roi = (128, 128, 16)
    
    model = SynVNet_8h2s().to(device)
    
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),  # Ensure channel-first format
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    fold_result = [{'dice': [], 'jaccard': [], 'precision': [], 'recall': [], 'hd95': [], 'assd': []} for c in range(n_center+1)]
    for fold in range(n_fold):  
        output_dir = os.path.join(args.output_dir, 'fold'+str(fold))
        output_image_dir = os.path.join(output_dir, 'output')
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
            
        test_images = []
        test_labels = []
        test_ds = []
        for c in range(n_center):
            image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
            train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = fold)
            print(f"Center {c+1} has {len(train_image)} training images and {len(test_image)} testing images")
            test_ds.append(Dataset(data=[{'image': image, 'label': label} for image, label in zip(test_image, test_label)], transform=test_transforms))
            test_images.append(test_image)
            test_labels.append(test_label)
        test_dataloader = []
        for c in range(n_center):
            test_dataloader.append(DataLoader(test_ds[c], batch_size=1, shuffle=False, num_workers=0))
        n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
        n_test_ds = sum([len(test_ds[i]) for i in range(n_center)]) 
        
        model.load_state_dict(torch.load(os.path.join(output_dir, args.resume), map_location='cpu', weights_only=True))
        global_result = {'dice': 0.0, 'jaccard': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0, 'assd': 0.0}
        for i in range(n_center):
            result = test_fn(model, test_dataloader[i], device, roi, output_image_dir, test_images[i], test_labels[i])
            print(f"Fold {fold} center {i} dice {result['dice']:.4f} jaccard {result['jaccard']:.4f} precision {result['precision']:.4f} recall {result['recall']:.4f} HD95 {result['hd95']:.4f} ASSD {result['assd']:.4f}") 
            for metric in ['dice', 'jaccard', 'precision', 'recall', 'hd95', 'assd']:
                global_result[metric] += result[metric] * len(test_ds[i])
                fold_result[i][metric].append(result[metric])
        for metric in ['dice', 'jaccard', 'precision', 'recall', 'hd95', 'assd']:
            global_result[metric] /= n_test_ds
            fold_result[-1][metric].append(global_result[metric])
        print(f"Fold {fold} global dice {global_result['dice']:.4f} jaccard {global_result['jaccard']:.4f} precision {global_result['precision']:.4f} recall {global_result['recall']:.4f} HD95 {global_result['hd95']:.4f} ASSD {global_result['assd']:.4f}")
    
    fold_mean = [{'dice': 0.0, 'jaccard': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0, 'assd': 0.0} for c in range(n_center+1)]
    fold_std = [{'dice': 0.0, 'jaccard': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0, 'assd': 0.0} for c in range(n_center+1)]
    for c in range(n_center+1):
        for metric in ['dice', 'jaccard', 'precision', 'recall', 'hd95', 'assd']:
            fold_mean[c][metric] = np.mean(fold_result[c][metric])
            fold_std[c][metric] = np.std(fold_result[c][metric])
        print(f"{c+1} {fold_mean[c]['dice']*100:.2f}$\\pm${fold_std[c]['dice']*100:.2f} & {fold_mean[c]['jaccard']*100:.2f}$\\pm${fold_std[c]['jaccard']*100:.2f} & {fold_mean[c]['precision']*100:.2f}$\\pm${fold_std[c]['precision']*100:.2f} & {fold_mean[c]['recall']*100:.2f}$\\pm${fold_std[c]['recall']*100:.2f} & {fold_mean[c]['hd95']:.2f}$\\pm${fold_std[c]['hd95']:.2f} & {fold_mean[c]['assd']:.2f}$\\pm${fold_std[c]['assd']:.2f}")
