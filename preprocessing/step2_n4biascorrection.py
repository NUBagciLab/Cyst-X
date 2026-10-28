# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:01:22 2026

@author: pky0507
"""

import argparse
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_n4_visualization(original_sitk, bias_field_sitk, corrected_sitk, out_png_path):
    """
    Saves a 3-panel comparison (Original, Estimated Bias Field, Corrected)
    on the middle axial slice.
    """
    orig_np = sitk.GetArrayFromImage(original_sitk)
    bias_np = sitk.GetArrayFromImage(bias_field_sitk)
    corr_np = sitk.GetArrayFromImage(corrected_sitk)

    slice_idx = orig_np.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original Image
    axes[0].imshow(np.rot90(orig_np[slice_idx], 2), cmap="gray")
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # 2. Estimated Bias Field
    im1 = axes[1].imshow(np.rot90(bias_np[slice_idx], 2), cmap="gray")
    axes[1].set_title("Estimated Bias Field", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. N4 Corrected Image
    axes[2].imshow(np.rot90(corr_np[slice_idx], 2), cmap="gray")
    axes[2].set_title("N4 Corrected", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_single_scan(task_args):
    """Worker function executed across multiple CPU processes."""
    img_path, out_path, vis_path, shrink_factor = task_args

    if os.path.exists(out_path) and (vis_path is None or os.path.exists(vis_path)):
        return Path(img_path).name, True, "Already exists"

    try:
        # Prevent CPU over-subscription per process
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

        input_img = sitk.ReadImage(img_path, sitk.sitkFloat32)

        # Downsample for faster bias field estimation
        if shrink_factor > 1:
            img_resampled = sitk.Shrink(
                input_img, [shrink_factor] * input_img.GetDimension()
            )
        else:
            img_resampled = input_img

        # Foreground mask via Otsu thresholding
        mask = sitk.OtsuThreshold(img_resampled, 0, 1, 200)

        # Multi-resolution N4 filter
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.SetSplineOrder(3)
        corrector.SetConvergenceThreshold(1e-4)

        _ = corrector.Execute(img_resampled, mask)

        # Evaluate B-spline field at full resolution
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
        bias_field = sitk.Exp(log_bias_field)

        # Correct original image
        corrected_img = input_img / bias_field

        # Save corrected volume
        sitk.WriteImage(corrected_img, out_path)

        # Generate visualization
        if vis_path is not None:
            save_n4_visualization(input_img, bias_field, corrected_img, vis_path)

        return Path(img_path).name, True, "Success"

    except Exception as e:
        return Path(img_path).name, False, str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Multiprocessing N4 Bias Field Correction.")
    parser.add_argument("-t", "--t", default=None, type=int, choices=[1, 2], help="Modality (1 or 2). Omit for both.")
    parser.add_argument("--dataset", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-RAS/segmentation"), help="Dataset root directory")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-n4/segmentation"), help="Base output directory")
    parser.add_argument("--shrink-factor", type=int, default=4, help="Downsample factor for N4")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel CPU workers")
    parser.add_argument("--no-vis", action="store_true", help="Disable comparison visualization export")
    return parser.parse_args()


def main():
    args = parse_args()
    t_list = [1, 2] if args.t is None else [args.t]

    for t in t_list:
        images_dir = os.path.join(args.dataset, f"t{t}", "images")
        output_dir = os.path.join(args.output, f"t{t}", "images")
        vis_dir = os.path.join(args.output, f"t{t}", "visualizations") if not args.no_vis else None

        os.makedirs(output_dir, exist_ok=True)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

        patterns = [os.path.join(images_dir, "*.nii"), os.path.join(images_dir, "*.nii.gz")]
        img_files = []
        for p in patterns:
            img_files.extend(glob.glob(p))
        img_files = sorted(list(set(img_files)))

        print(f"Total scans to process: {len(img_files)}")
        print(f"Using {args.num_workers} parallel CPU workers (shrink_factor={args.shrink_factor}).")

        # Build task list
        tasks = []
        for img_path in img_files:
            filename = Path(img_path).name
            save_name = filename.replace(".nii", ".nii.gz") if not filename.endswith(".nii.gz") else filename
            out_path = os.path.join(output_dir, save_name)

            vis_path = None
            if vis_dir:
                vis_filename = save_name.replace(".nii.gz", ".png")
                vis_path = os.path.join(vis_dir, vis_filename)

            tasks.append((img_path, out_path, vis_path, args.shrink_factor))

        # Parallel Execution
        failed = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_scan, task): task for task in tasks}

            pbar = tqdm(as_completed(futures), total=len(tasks), desc=f"N4 Correcting t{t}")
            for future in pbar:
                name, success, msg = future.result()
                if not success:
                    failed.append((name, msg))
                    pbar.write(f"[ERROR] Failed {name}: {msg}")

        print(f"\nProcessing complete. Corrected images saved to: '{output_dir}'")
        if vis_dir:
            print(f"Visualizations saved to: '{vis_dir}'")
        if failed:
            print(f"Warning: {len(failed)} scans encountered errors.")


if __name__ == "__main__":
    main()
