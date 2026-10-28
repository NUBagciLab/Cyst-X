# -*- coding: utf-8 -*-
"""
ProcessPoolExecutor-based batch reorientation to RAS and visualization pipeline using MONAI.
Supports both paired (image + mask) and image-only modes.
"""

import argparse
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    MapTransform,
)


class SynchronizeMaskAffined(MapTransform):
    """
    Synchronizes the mask's affine matrix and spatial metadata to strictly
    match the image's geometry before resampling/cropping.
    """
    def __init__(self, image_key: str = "image", mask_key: str = "mask"):
        super().__init__(keys=[image_key, mask_key])
        self.image_key = image_key
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        if hasattr(d[self.image_key], "meta") and hasattr(d[self.mask_key], "meta"):
            d[self.mask_key].meta["affine"] = d[self.image_key].meta["affine"].clone()
            d[self.mask_key].meta["spatial_shape"] = d[self.image_key].meta["spatial_shape"]
        if hasattr(d[self.image_key], "affine") and hasattr(d[self.mask_key], "affine"):
            d[self.mask_key].affine = d[self.image_key].affine.clone()
        return d


def build_pipeline():
    """Builds a MONAI transform pipeline to reorient both image and mask to RAS."""
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        #SynchronizeMaskAffined(image_key="image", mask_key="mask"),
        Orientationd(keys=["image", "mask"], axcodes="RAS", labels=None),
    ])


def build_image_only_pipeline():
    """Builds a MONAI transform pipeline to reorient image alone to RAS."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
    ])


def process_single_case(item_dict, alpha, no_vis, out_images_dir, out_masks_dir, out_vis_dir):
    try:
        has_mask = item_dict.get("mask") is not None
        name = item_dict["name"]

        if has_mask:
            pipeline = build_pipeline()
            transformed = pipeline(item_dict)
            img_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
            mask_np = (mask_tensor[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            pipeline = build_image_only_pipeline()
            transformed = pipeline(item_dict)
            img_tensor = transformed["image"]
            mask_np = None

        # Extract (H, W, D) numpy array
        img_np = img_tensor[0].detach().cpu().numpy()

        # Retrieve RAS affine from the transformed MONAI MetaTensor
        if hasattr(img_tensor, "affine"):
            out_affine = img_tensor.affine.detach().cpu().numpy()
        elif "affine" in img_tensor.meta:
            out_affine = np.array(img_tensor.meta["affine"])
        else:
            out_affine = np.eye(4)

        # 1. Save RAS NIfTI volumes
        nii_img = nib.Nifti1Image(img_np.astype(np.float32), out_affine)
        save_img_path = os.path.join(out_images_dir, f"{name}.nii.gz")
        nib.save(nii_img, save_img_path)

        if has_mask and mask_np is not None:
            nii_mask = nib.Nifti1Image(mask_np, out_affine)
            save_mask_path = os.path.join(out_masks_dir, f"{name}.nii.gz")
            nib.save(nii_mask, save_mask_path)

        # 2. Slice Visualization
        if not no_vis:
            total_slices = img_np.shape[-1]

            if has_mask and mask_np is not None:
                mask_counts = mask_np.sum(axis=(0, 1))
                if mask_counts.max() > 0:
                    slice_idx = int(np.argmax(mask_counts))
                else:
                    slice_idx = total_slices // 2
            else:
                slice_idx = total_slices // 2

            img_slice = img_np[:, :, slice_idx]

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(np.fliplr(np.rot90(img_slice)), cmap="gray")

            if has_mask and mask_np is not None:
                mask_slice = mask_np[:, :, slice_idx]
                mask_overlay = np.where(mask_slice > 0, 1.0, np.nan)
                plt.imshow(
                    np.fliplr(np.rot90(mask_overlay)),
                    cmap="Reds",
                    alpha=alpha,
                    vmin=0,
                    vmax=1,
                )

            plt.axis("off")

            vis_filename = f"{name}_slice_{slice_idx}_of_{total_slices}.png"
            vis_path = os.path.join(out_vis_dir, vis_filename)
            plt.savefig(vis_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return True, name, None
    except Exception as e:
        return False, item_dict.get("name", "unknown"), str(e)


def get_file_pairs(images_dir: str, masks_dir: str):
    patterns = [os.path.join(images_dir, "*.nii"), os.path.join(images_dir, "*.nii.gz")]
    img_files = []
    for p in patterns:
        img_files.extend(glob.glob(p))
    img_files = sorted(list(set(img_files)))

    data_list = []
    for img_path in img_files:
        name = Path(img_path).name.replace(".nii.gz", "").replace(".nii", "")
        
        mask_path = None
        if os.path.exists(masks_dir):
            candidate_gz = os.path.join(masks_dir, f"{name}.nii.gz")
            candidate_nii = os.path.join(masks_dir, f"{name}.nii")
            if os.path.exists(candidate_gz):
                mask_path = candidate_gz
            elif os.path.exists(candidate_nii):
                mask_path = candidate_nii

        if mask_path is None:
            print(f"[Info] Mask not found for {name}. Will process as image-only.")

        data_list.append({"image": img_path, "mask": mask_path, "name": name})

    return data_list


def parse_args():
    parser = argparse.ArgumentParser(description="MONAI 3D MRI Orientation Normalization and Visualization")
    parser.add_argument("-i", "--input", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X/segmentation"), help="Input folder")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-RAS/segmentation"), help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("-a", "--alpha", type=float, default=0.4, help="Overlay alpha transparency")
    parser.add_argument("-t", "--t", default=None, type = int, choices=[1, 2], help="Modality (1 or 2)")
    parser.add_argument("--no-vis", action="store_true", help="Disable PNG overlay export")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.t == None:
        t_list = [1, 2]
    else:
        t_list = [args.t]
    for t in t_list:
        args.input_dir = os.path.join(args.input, f"t{t}")
        args.output_dir = os.path.join(args.output, f"t{t}")
        args.image = os.path.join(args.input_dir, "images")
        args.mask = os.path.join(args.input_dir, "masks")
        out_images_dir = os.path.join(args.output_dir, "images")
        out_masks_dir = os.path.join(args.output_dir, "masks")
        out_vis_dir = os.path.join(args.output_dir, "visualizations")
    
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)
        if not args.no_vis:
            os.makedirs(out_vis_dir, exist_ok=True)
    
        data_list = get_file_pairs(args.image, args.mask)
        paired_count = sum(1 for item in data_list if item["mask"] is not None)
        image_only_count = len(data_list) - paired_count
    
        print(
            f"Found {len(data_list)} total scans:\n"
            f"  - Paired (image + mask): {paired_count}\n"
            f"  - Image-only: {image_only_count}\n"
            f"  Images dir: {args.image}\n"
            f"  Masks dir : {args.mask}"
        )
    
        if len(data_list) == 0:
            return
    
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_case,
                    item,
                    args.alpha,
                    args.no_vis,
                    out_images_dir,
                    out_masks_dir,
                    out_vis_dir,
                ): item["name"]
                for item in data_list
            }
    
            with tqdm(total=len(data_list), desc=f"Reorienting to RAS ({args.workers} workers)") as pbar:
                for future in as_completed(futures):
                    success, name, error_msg = future.result()
                    if not success:
                        tqdm.write(f"[Error] Failed to process {name}: {error_msg}")
                    pbar.update(1)


if __name__ == "__main__":
    main()