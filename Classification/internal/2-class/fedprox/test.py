import os
import argparse
import torch
import torch.nn as nn
from model import get_model
from sklearn.metrics import roc_auc_score, roc_curve
from train import load_data, test_fn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/dataset/Cyst-X-resampled/classification", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 4)")
    parser.add_argument("--mu", default=0.1, type=float, help="FedProx mu (default: 0.1)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.mu), 't'+str(args.t), 'fold'+str(args.fold))
    
    device = torch.device(args.device)
    n_center = 7
    _, _, test_dataloader = load_data(args, n_center=n_center)

    model = get_model(name = args.model, num_classes = 1)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
    fpr, tpr, thresholds = roc_curve(epoch_y['true'], epoch_y['pred'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=4, label='Global')
    print(f"Global test loss {epoch_log['loss']:.4f} acc {epoch_log['acc']:.4f} auc {epoch_log['auc']:.4f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.pdf", format="pdf", bbox_inches='tight')
    #plt.show()