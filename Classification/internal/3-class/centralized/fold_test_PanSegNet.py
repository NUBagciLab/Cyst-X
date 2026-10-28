import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from train import load_data, test_fn
from data_loader import get_data_list, get_fold
import pandas as pd
from auc_ci import calculate_auc_ci_cv, highlight_errors
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="../../../preprocessing/PanSegNet/", type=str, help="dataset path")
    parser.add_argument("--model", default="densenet121", type=str, help="model name")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, 't'+str(args.t))
    os.makedirs(args.output_dir, exist_ok=True)
    result_lines = []

    def print_result(message):
        print(message)
        result_lines.append(message)

    
    device = torch.device(args.device)
    
    model = get_model(name = args.model, num_classes = 3)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    n_center = 7
    n_fold = 5
    y_all = []
    pred_all = []
    folds_data = []
    log = [{} for j in range(n_fold)]   
    csv_images = []
    csv_labels = []
    csv_folds = []
    for fold in range(n_fold):
        args.fold = fold
        _, _, test_dataloader = load_data(args, n_center=n_center)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(fold), args.resume), map_location='cpu', weights_only=True))   

        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)

        y_true_fold = np.array(epoch_y['true'])
        y_pred_fold = np.argmax(np.array(epoch_y['pred']), axis=-1)
        if 'sens' not in epoch_log or 'spec' not in epoch_log:
              y_true_high = (y_true_fold == 2).astype(int)
              y_pred_high = (y_pred_fold == 2).astype(int)
              tn, fp, fn, tp = confusion_matrix(y_true_high, y_pred_high, labels=[0, 1]).ravel()
              epoch_log['sens'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
              epoch_log['spec'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        for metric in ['loss', 'acc', 'sens', 'spec', 'auc']:
            log[fold]['test_'+metric] = epoch_log[metric]
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        folds_data.append((epoch_y['true'], epoch_y['pred']))
        test_image = []
        test_label = []       
        for c in range(n_center):
            image_list_c, label_list_c = get_data_list(root=args.data_path, t=args.t, center=c)
            _, _, test_image_c, test_label_c = get_fold(image=image_list_c, label=label_list_c, root=args.data_path, t=args.t, fold=fold)
            test_image.extend(test_image_c)
            test_label.extend(test_label_c)
        csv_images.extend([os.path.basename(i).replace('.nii.gz', '') for i in test_image])
        csv_labels.extend([i for i in test_label])
        csv_folds.extend([fold for i in range(len(epoch_y['pred']))])

    for fold in range(n_fold): 
        print_result(f"Fold {fold} test loss {log[fold]['test_loss']:.4f} acc {log[fold]['test_acc']:.4f} sens {log[fold]['test_sens']:.4f} spec {log[fold]['test_spec']:.4f} auc {log[fold]['test_auc']:.4f}")
    
    log_mean = {}   
    log_std = {}   
    for metric in ['loss', 'acc', 'sens', 'spec', 'auc']:
        log_mean['test_'+metric] = np.mean([log[fold]['test_'+metric] for fold in range(n_fold)])
        log_std['test_'+metric] = np.std([log[fold]['test_'+metric] for fold in range(n_fold)])
    print_result(f"Test loss {log_mean['test_loss']:.4f}±{log_std['test_loss']:.4f} acc {log_mean['test_acc']:.4f}±{log_std['test_acc']:.4f} sens {log_mean['test_sens']:.4f}±{log_std['test_sens']:.4f} spec {log_mean['test_spec']:.4f}±{log_std['test_spec']:.4f} auc {log_mean['test_auc']:.4f}±{log_std['test_auc']:.4f}")

    lower_bound, upper_bound = calculate_auc_ci_cv(folds_data)
    log_mean['auc_lower'] = lower_bound
    log_mean['auc_upper'] = upper_bound
    print_result(f"95%CI: [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}]")
    
    print_result(f"{log_mean['test_auc']*100:.2f}$\\pm${log_std['test_auc']*100:.2f} & [{log_mean['auc_lower']*100:.2f}, {log_mean['auc_upper']*100:.2f}] & {log_mean['test_acc']*100:.2f}$\\pm${log_std['test_acc']*100:.2f} & {log_mean['test_sens']*100:.2f}$\\pm${log_std['test_sens']*100:.2f} & {log_mean['test_spec']*100:.2f}$\\pm${log_std['test_spec']*100:.2f}")

    with open(os.path.join(args.output_dir, "PanSegNet_result.txt"), "w", encoding="utf-8") as result_file:
        result_file.write("\n".join(result_lines) + "\n")

    csv_probabilities = pred_all
    df = pd.read_excel(os.path.join(args.data_path, 'IPMN_labels_t'+str(args.t)+'_total.xlsx'), usecols=[0, 5])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = [i.replace('.nii.gz', '') for i in df_cleaned.iloc[:, 0].values]
    risks =  df_cleaned.iloc[:, 1].to_numpy(dtype=np.float32)
    mapping = {value: i for i, value in enumerate(csv_images)}
    indices = [mapping[value] for value in names]
    csv_images = [csv_images[i] for i in indices]
    csv_labels = [csv_labels[i] for i in indices]
    csv_probabilities = [csv_probabilities[i] for i in indices]
    csv_predictions = [np.argmax(probabilities) for probabilities in csv_probabilities]
    csv_folds = [csv_folds[i] for i in indices]
    df = pd.DataFrame({
        'ID': csv_images,
        'Risk Assessment': risks,
        'Label': csv_labels,
        'Prediction': csv_predictions,
        'Probability No Risk': [p[0] for p in csv_probabilities],
        'Probability Low Risk': [p[1] for p in csv_probabilities],
        'Probability High Risk': [p[2] for p in csv_probabilities],
        'Fold': csv_folds
    })
    df.style.apply(highlight_errors, axis=1).to_excel(os.path.join(args.output_dir, 'PanSegNet_result.xlsx'), index=False)