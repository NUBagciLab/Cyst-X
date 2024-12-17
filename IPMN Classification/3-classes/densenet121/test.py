import os
import argparse
import torch
import torch.nn as nn
from data_loader import get_data_list, get_fold
from model import get_model
from monai.data import DataLoader, ImageDataset
from monai.transforms import Resize, EnsureChannelFirst, Compose, ScaleIntensity
from train import test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification test.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 4)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t), 'fold'+str(args.fold))
    
    device = torch.device(args.device)
    n_center = 7
    n_class = 3
    image_lists = []
    label_lists = []
    test_ds = []
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    for c in range(n_center):
        image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
        train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
        print(f"Center {c+1} has {len(train_image)} training images and {len(test_image)} testing images")
        test_ds.append(ImageDataset(image_files=test_image, labels=test_label, transform=test_transforms))
    test_dataloader = DataLoader(torch.utils.data.ConcatDataset(test_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = get_model(out_channels = n_class)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    log = {'test_loss':[], 'test_acc':[], 'test_auc':[]}    
    epoch_log, _ = test_fn(test_dataloader, model, loss_fn, device)
    for metric in ['loss', 'acc', 'auc']:
        log['test_'+metric].append(epoch_log[metric])
    print(f"Test loss {log['test_loss'][-1]:.4f} acc {log['test_acc'][-1]:.4f} auc {log['test_auc'][-1]:.4f}")