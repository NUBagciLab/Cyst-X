# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:20:04 2026

@author: pky0507
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
from concurrent.futures import ProcessPoolExecutor
import SimpleITK as sitk
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt


def rename_ahn_files(target_dir):
    """
    Renames files like 'AHN05.nii.gz' or 'AHN_05.nii.gz' to 'ahn_05.nii.gz'
    """
    if not os.path.exists(target_dir):
        return

    count = 0
    for filename in sorted(os.listdir(target_dir)):
        if filename.startswith("AHN"):
            new_name = re.sub(r"^AHN_?", "ahn_", filename)
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                count += 1

    print(f"Completed: {count} files renamed in {target_dir}\n")


def _save_three_view_preview(image_sitk, vis_output_dir, base_name, mask_sitk=None, alpha=0.4):
    """Save axial, coronal, and sagittal views with slice numbers in their titles."""
    img_np = sitk.GetArrayFromImage(image_sitk)
    mask_np = sitk.GetArrayFromImage(mask_sitk) if mask_sitk is not None else None
    spacing = image_sitk.GetSpacing()[::-1]  # NumPy axis order: z, y, x

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    try:
        for axis, (ax, view) in enumerate(zip(axes, ("Axial", "Coronal", "Sagittal"))):
            total_slices = img_np.shape[axis]
            slice_idx = total_slices // 2
            if mask_np is not None:
                other_axes = tuple(i for i in range(3) if i != axis)
                mask_counts = np.count_nonzero(mask_np > 0, axis=other_axes)
                if mask_counts.max() > 0:
                    slice_idx = int(np.argmax(mask_counts))

            img_slice = np.take(img_np, slice_idx, axis=axis)
            plane_axes = [i for i in range(3) if i != axis]
            aspect = spacing[plane_axes[0]] / spacing[plane_axes[1]]
            ax.imshow(img_slice, cmap="gray", aspect=aspect)
            if mask_np is not None:
                mask_slice = np.take(mask_np, slice_idx, axis=axis)
                mask_overlay = np.where(mask_slice > 0, 1.0, np.nan)
                ax.imshow(mask_overlay, cmap="Reds", alpha=alpha, vmin=0, vmax=1, aspect=aspect)
            ax.set_title(f"{view} — Slice {slice_idx + 1}/{total_slices}")
            ax.axis("off")

        fig.tight_layout()
        output_png_path = os.path.join(vis_output_dir, f"{base_name}.png")
        fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_image_only_slice(image_sitk, vis_output_dir, base_name):
    """Save three orthogonal center slices for image-only mode."""
    _save_three_view_preview(image_sitk, vis_output_dir, base_name)


def save_overlay_slice(image_sitk, mask_sitk, vis_output_dir, base_name, alpha=0.4):
    """Save three orthogonal overlays, selecting maximal mask area in each view."""
    _save_three_view_preview(image_sitk, vis_output_dir, base_name, mask_sitk, alpha)


def reorient_mask_to_image(
    mask_path,
    output_path,
    image_path,
    vis_output_dir=None,
    base_name=None,
    force_mask_geometry_from_image=True,
    alpha=0.4,
):
    """
    Resample one mask to exactly match its paired image geometry and optionally export an overlay plot.
    """
    mask = sitk.ReadImage(mask_path)
    image = sitk.ReadImage(image_path)

    if force_mask_geometry_from_image:
        mask = sitk.Image(mask)
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
        mask.SetDirection(image.GetDirection())

    reoriented_mask = sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )
    sitk.WriteImage(reoriented_mask, output_path)

    if vis_output_dir and base_name:
        save_overlay_slice(image, reoriented_mask, vis_output_dir, base_name, alpha=alpha)


def _process_one_case(task):
    (
        image_path,
        mask_input_path,
        mask_output_path,
        vis_output_dir,
        base_name,
        force_mask_geometry_from_image,
        alpha,
    ) = task

    reorient_mask_to_image(
        mask_input_path,
        mask_output_path,
        image_path,
        vis_output_dir=vis_output_dir,
        base_name=base_name,
        force_mask_geometry_from_image=force_mask_geometry_from_image,
        alpha=alpha,
    )
    return os.path.basename(image_path)


def _process_one_image_vis(task):
    image_path, vis_output_dir, base_name = task
    image = sitk.ReadImage(image_path)
    save_image_only_slice(image, vis_output_dir, base_name)
    return os.path.basename(image_path)


def batch_reorient_masks(
    image_dir,
    mask_dir,
    mask_output_dir,
    vis_output_dir=None,
    num_workers=8,
    force_mask_geometry_from_image=True,
    alpha=0.4,
):
    os.makedirs(mask_output_dir, exist_ok=True)
    if vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)

    tasks = []
    for fname in os.listdir(image_dir):
        if not fname.endswith(".nii.gz") and not fname.endswith(".nii"):
            continue

        image_path = os.path.join(image_dir, fname)
        candidate_mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(candidate_mask_path):
            print(f"Mask not found for {fname}, skipping.")
            continue

        mask_output_path = os.path.join(mask_output_dir, fname)
        base_name = fname.replace(".nii.gz", "").replace(".nii", "")

        tasks.append(
            (
                image_path,
                candidate_mask_path,
                mask_output_path,
                vis_output_dir,
                base_name,
                force_mask_geometry_from_image,
                alpha,
            )
        )

    if not tasks:
        print("No image/mask pairs found to process.")
        return

    print(f"Reorienting & plotting {len(tasks)} masks with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for fname in ex.map(_process_one_case, tasks):
            print(f"Processed: {fname}")

    print("Mask processing completed.")


def batch_visualize_images_only(image_dir, vis_output_dir, num_workers=8):
    os.makedirs(vis_output_dir, exist_ok=True)
    tasks = []
    for fname in os.listdir(image_dir):
        if not fname.endswith(".nii.gz") and not fname.endswith(".nii"):
            continue
        image_path = os.path.join(image_dir, fname)
        base_name = fname.replace(".nii.gz", "").replace(".nii", "")
        tasks.append((image_path, vis_output_dir, base_name))

    print(f"Generating visual previews for {len(tasks)} images with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for fname in ex.map(_process_one_image_vis, tasks):
            print(f"Visualized: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy image folder, reorient masks (if present), export slice previews, and rename."
    )
    parser.add_argument("-i", "--input", default=os.path.join(PROJECT_ROOT, "Cyst-X-input/segmentation"), help="Input folder with 'images' (and optional 'masks')")
    parser.add_argument("-o", "--output", default=os.path.join(PROJECT_ROOT, "Cyst-X/segmentation"), help="Output root directory")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--alpha", type=float, default=0.4, help="Visualization mask overlay alpha transparency")
    parser.add_argument("--no_vis", action="store_true", help="Disable PNG visualization export")
    parser.add_argument("-t", "--t", default=None, type = int, choices=[1, 2], help="Modality (1 or 2)")
    parser.add_argument(
        "--no_force_mask_geometry_from_image",
        action="store_true",
        help="Disable forcing mask geometry to match paired source image before resampling.",
    )

    args = parser.parse_args()
    if args.t == None:
        t_list = [1, 2]
    else:
        t_list = [args.t]
    for t in t_list:
        args.input_dir = os.path.join(args.input, f"t{t}")
        args.output_dir = os.path.join(args.output, f"t{t}")
        args.image_dir = os.path.join(args.input_dir, "images")
        args.mask_dir = os.path.join(args.input_dir, "masks")
        args.image_output_dir = os.path.join(args.output_dir, "images")
        args.mask_output_dir = os.path.join(args.output_dir, "masks")
        args.vis_output_dir = None if args.no_vis else os.path.join(args.output_dir, "visualizations")
    
        # 1. Copy entire image folder upfront outside of multiprocessing
        if os.path.exists(args.image_output_dir):
            shutil.rmtree(args.image_output_dir)
        shutil.copytree(args.image_dir, args.image_output_dir)
    
        # 2. Check if masks exist; toggle paired vs. image-only workflow
        has_masks = os.path.isdir(args.mask_dir) and len([
            f for f in os.listdir(args.mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
        ]) > 0
    
        if has_masks:
            print(f"Found masks in '{args.mask_dir}'. Running paired image + mask pipeline...")
            batch_reorient_masks(
                image_dir=args.image_dir,
                mask_dir=args.mask_dir,
                mask_output_dir=args.mask_output_dir,
                vis_output_dir=args.vis_output_dir,
                num_workers=args.num_workers,
                force_mask_geometry_from_image=not args.no_force_mask_geometry_from_image,
                alpha=args.alpha,
            )
            rename_ahn_files(args.mask_output_dir)
        else:
            print("No masks directory or mask files found. Running in image-only mode...")
            if args.vis_output_dir:
                batch_visualize_images_only(
                    image_dir=args.image_dir,
                    vis_output_dir=args.vis_output_dir,
                    num_workers=args.num_workers,
                )
    
        # 3. Rename prefixes in outputs (including visualization PNGs)
        rename_ahn_files(args.image_output_dir)
        if args.vis_output_dir:
            rename_ahn_files(args.vis_output_dir)