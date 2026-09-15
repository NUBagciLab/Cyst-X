import os
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", message=r".*always_return_as_numpy.*")

import torch
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    ConfusionMatrixMetric,
)
from monai.transforms import AsDiscrete


def format_sig4(value):
    """Format a number with 4 significant figures."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "nan"
    if value == 0:
        return "0"
    return f"{value:.4g}"


def format_mean_std(mean, std):
    return f"{format_sig4(mean)}±{format_sig4(std)}"


def get_num_classes(seg_array, gt_array):
    """Infer number of classes (including background 0) from seg and gt."""
    max_label = int(max(seg_array.max(), gt_array.max()))
    return max_label + 1


def surface_distances(result, reference, spacing, connectivity=1):
    """MedPy-compatible directed surface distances in physical units."""
    result = np.asarray(result).astype(bool)
    reference = np.asarray(reference).astype(bool)
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return None
    footprint = generate_binary_structure(result.ndim, connectivity)
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    dt = distance_transform_edt(~reference_border, sampling=spacing)
    return dt[result_border]


def hd95_mm(pred, gt, spacing):
    d1 = surface_distances(pred, gt, spacing)
    d2 = surface_distances(gt, pred, spacing)
    if d1 is None or d2 is None or d1.size == 0 or d2.size == 0:
        return np.nan
    return float(np.percentile(np.hstack((d1, d2)), 95))


def assd_mm(pred, gt, spacing):
    d1 = surface_distances(pred, gt, spacing)
    d2 = surface_distances(gt, pred, spacing)
    if d1 is None or d2 is None or d1.size == 0 or d2.size == 0:
        return np.nan
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def evaluate_sample(name, segmentation_path, ground_truth_path):
    """Evaluate a single sample. Returns metrics dict or None if skipped."""
    torch.set_num_threads(1)

    seg_path = os.path.join(segmentation_path, name)
    gt_path = os.path.join(ground_truth_path, name)
    if not os.path.isfile(gt_path):
        print(f"Skipping {name}: Ground truth file not found")
        return None

    try:
        seg_img = nib.load(seg_path)
        gt_img = nib.load(gt_path)
    except Exception as e:
        print(f"Skipping {name}: {e}")
        return None

    seg_array = seg_img.get_fdata().astype(np.int32)
    gt_array = gt_img.get_fdata().astype(np.int32)
    spacing = tuple(float(x) for x in gt_img.header.get_zooms()[: gt_array.ndim])

    num_classes = get_num_classes(seg_array, gt_array)
    labels = list(range(1, num_classes))

    seg_array = np.clip(seg_array, 0, num_classes - 1)
    gt_array = np.clip(gt_array, 0, num_classes - 1)

    seg_tensor = torch.tensor(seg_array).unsqueeze(0)
    gt_tensor = torch.tensor(gt_array).unsqueeze(0)

    if seg_tensor.shape != gt_tensor.shape:
        print(f"Skipping {name}: Shape mismatch (Seg: {seg_tensor.shape}, GT: {gt_tensor.shape})")
        return None

    post_label = AsDiscrete(to_onehot=num_classes)
    seg_onehot = post_label(seg_tensor)
    gt_onehot = post_label(gt_tensor)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
    jaccard_metric = MeanIoU(include_background=False, reduction="mean_batch")
    confusion = ConfusionMatrixMetric(
        include_background=False, reduction="mean_batch", metric_name=["sensitivity", "precision"]
    )

    dice_metric(y_pred=[seg_onehot], y=[gt_onehot])
    iou_metric(y_pred=[seg_onehot], y=[gt_onehot])
    jaccard_metric(y_pred=[seg_onehot], y=[gt_onehot])
    confusion(y_pred=[seg_onehot], y=[gt_onehot])

    dice_scores = dice_metric.aggregate().tolist()
    iou_scores = iou_metric.aggregate().tolist()
    jaccard_scores = jaccard_metric.aggregate().tolist()
    sensitivity, precision = confusion.aggregate()[0].tolist(), confusion.aggregate()[1].tolist()

    hd95_scores = []
    assd_scores = []
    for label in labels:
        pred_bin = seg_array == label
        gt_bin = gt_array == label
        hd95_scores.append(hd95_mm(pred_bin, gt_bin, spacing))
        assd_scores.append(assd_mm(pred_bin, gt_bin, spacing))

    metrics_dict = {"name": name, "num_classes": num_classes}
    for i, label in enumerate(labels):
        metrics_dict[f"dice_class_{label}"] = dice_scores[i]
        metrics_dict[f"hd95_class_{label}"] = hd95_scores[i]
        metrics_dict[f"assd_class_{label}"] = assd_scores[i]
        metrics_dict[f"iou_class_{label}"] = iou_scores[i]
        metrics_dict[f"jaccard_class_{label}"] = jaccard_scores[i]
        metrics_dict[f"precision_class_{label}"] = precision[i]
        metrics_dict[f"recall_class_{label}"] = sensitivity[i]

    metrics_dict["dice_avg"] = float(np.mean(dice_scores))
    metrics_dict["hd95_avg"] = float(np.mean(hd95_scores))
    metrics_dict["assd_avg"] = float(np.mean(assd_scores))
    metrics_dict["iou_avg"] = float(np.mean(iou_scores))
    metrics_dict["jaccard_avg"] = float(np.mean(jaccard_scores))
    metrics_dict["precision_avg"] = float(np.mean(precision))
    metrics_dict["recall_avg"] = float(np.mean(sensitivity))

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Segmentation Evaluation")
    parser.add_argument('--segmentation_folder', '-s', required=True, help='Path to the segmentation folder')
    parser.add_argument('--ground_truth_folder', '-g', required=True, help='Path to the ground truth folder')
    parser.add_argument(
        '--num_workers', '-w', type=int, default=32,
        help='Number of parallel workers (default: min(cpu_count, 8))'
    )
    args = parser.parse_args()

    segmentation_path = args.segmentation_folder
    ground_truth_path = args.ground_truth_folder
    num_workers = args.num_workers or min(os.cpu_count() or 1, 8)

    names = [f for f in os.listdir(segmentation_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
    metrics_list = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_sample, name, segmentation_path, ground_truth_path): name
            for name in names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Segmentation Results"):
            result = future.result()
            if result is not None:
                metrics_list.append(result)

    metrics_df = pd.DataFrame(metrics_list)
    output_folder = os.path.dirname(args.segmentation_folder)
    folder_name = args.segmentation_folder.rstrip("/").split("/")[-1]
    metrics_csv = os.path.join(output_folder, f"{folder_name}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary_rows = []
    for col in numeric_cols:
        col_data = metrics_df[col].dropna()
        if len(col_data) == 0:
            continue
        mean_val = col_data.mean()
        std_val = col_data.std(ddof=1) if len(col_data) > 1 else 0.0
        summary_rows.append({
            "metric": col,
            "mean": mean_val,
            "std": std_val,
            "mean±std": format_mean_std(mean_val, std_val),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_folder, f"{folder_name}_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Parallel workers: {num_workers}")
    print(f"Per-sample metrics saved to: {metrics_csv}")
    print(f"Summary (mean±std) saved to: {summary_csv}")


if __name__ == "__main__":
    main()
