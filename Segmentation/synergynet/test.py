import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
from data_loader import get_data_list, get_fold
from monai import transforms
from monai.data import DataLoader, Dataset
from synergynet import SynVNet_8h2s
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
import nibabel as nib
from collections import OrderedDict

def test_fn(model, loader, device, roi, output_dir, image_path, label_path):
    model.eval()
    total_dice = 0.0
    total_jaccard = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_hd95 = 0.0
    total_assd = 0.0
    batch_count = 0
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    jaccard_metric = MeanIoU(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    assd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=False, threshold=0.0)
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for data in progress_bar:
            x = data['image'].to(device)
            y = data['label'].to(device)
            image = nib.load(image_path[batch_count])
            spacing = np.sqrt(np.sum(image.affine[:3, :3] ** 2, axis=0))  # Diagonal elements
            y_hat = sliding_window_inference(
                inputs=x,
                roi_size=roi,
                sw_batch_size=4,
                predictor=model.forward,
                overlap=0.5,
                mode="gaussian"
            )
            y_hat = post_pred(y_hat)
            nifti_img = nib.Nifti1Image(y_hat.cpu().numpy()[0].squeeze(0), image.affine, image.header)
            nib.save(nifti_img, os.path.join(output_dir, os.path.basename(label_path[batch_count])))
            dice = dice_metric(y_pred=y_hat, y=y).item()
            jaccard = jaccard_metric(y_pred=y_hat, y=y).item()        
            TP = (y_hat * y).sum().item()  # True positives
            FP = (y_hat * (1 - y)).sum().item()  # False positives
            FN = ((1 - y_hat) * y).sum().item()  # False negatives
            # Compute precision and recall
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            hd95 = hd95_metric(y_pred=y_hat, y=y, spacing=spacing).item()
            assd = assd_metric(y_pred=y_hat, y=y, spacing=spacing).item()
            total_dice += dice
            total_jaccard += jaccard
            total_precision += precision
            total_recall += recall
            total_hd95 += hd95
            total_assd += assd
            batch_count += 1
            metrics = OrderedDict([
            ("dice", f"{dice:.4f}({total_dice/batch_count:.4f})"),
            ("jaccard", f"{jaccard:.4f}({total_jaccard/batch_count:.4f})"),
            ("precision", f"{precision:.4f}({total_precision/batch_count:.4f})"),
            ("recall", f"{recall:.4f}({total_recall/batch_count:.4f})"),
            ("hd95", f"{hd95:.2f}({total_hd95/batch_count:.4f})"),
            ("assd", f"{assd:.3f}({total_assd/batch_count:.4f})")])
            progress_bar.set_postfix(metrics)
    return {'dice': total_dice/batch_count, 'jaccard': total_jaccard/batch_count, 'precision': total_precision/batch_count, 'recall': total_recall/batch_count, 'hd95': total_hd95/batch_count, 'assd': total_assd/batch_count}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Test.")
    parser.add_argument("--data-path", default="/data/pky0507/IPMN_images_masks/", type=str, help="dataset path")
    parser.add_argument("--data-path", default="/dataset/IPMN_images_masks/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--t", default=1, type=int, help="Modalities (1 or 2)")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 4)")
    parser.add_argument("--resume", default="model_dice.pth", type=str, help="path of checkpoint")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t), 'fold'+str(args.fold))
    output_image_dir = os.path.join(args.output_dir, 'output')
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    
    device = torch.device(args.device)
    n_center = 7
    test_images = []
    test_labels = []
    test_ds = []
    roi = (128, 128, 32)
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
        test_ds.append(Dataset(data=[{'image': image, 'label': label} for image, label in zip(test_image, test_label)], transform=test_transforms))
        test_images.append(test_image)
        test_labels.append(test_label)
    test_dataloader = []
    for c in range(n_center):
        test_dataloader.append(DataLoader(test_ds[c], batch_size=1, shuffle=False, num_workers=0))
    n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
    n_test_ds = sum([len(test_ds[i]) for i in range(n_center)]) 
    
    model = SynVNet_8h2s().to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))    
    center_result = []
    global_result = {'dice': 0.0, 'jaccard': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0, 'assd': 0.0}
    for i in range(n_center):
        result = test_fn(model, test_dataloader[i], device, roi, output_image_dir, test_images[i], test_labels[i])
        center_result.append(result)
        for metric in ['dice', 'jaccard', 'precision', 'recall', 'hd95', 'assd']:
            global_result[metric] += result[metric] * len(test_ds[i])
    for metric in ['dice', 'jaccard', 'precision', 'recall', 'hd95', 'assd']:
        global_result[metric] /= n_test_ds
    for i in range(n_center):
        print(f"Center {i+1} dice {center_result[i]['dice']*100:.2f} jaccard {center_result[i]['jaccard']*100:.2f} precision {center_result[i]['precision']*100:.2f} recall {center_result[i]['recall']*100:.2f} HD95 {center_result[i]['hd95']:.4f} ASSD {center_result[i]['assd']:.2f}") 
    print(f"Global dice {global_result['dice']*100:.2f} jaccard {global_result['jaccard']*100:.2f} precision {global_result['precision']*100:.2f} recall {global_result['recall']*100:.2f} HD95 {global_result['hd95']:.4f} ASSD {global_result['assd']:.2f}") 