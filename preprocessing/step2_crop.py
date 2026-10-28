# -*- coding: utf-8 -*-
import argparse
import glob
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm


def get_resample_target_info(img: nib.Nifti1Image, target_spacing):
    data = img.get_fdata()
    affine = img.affine
    old_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    new_spacing = np.array(target_spacing)
    old_shape = np.array(data.shape)
    new_shape = np.round(old_shape * old_spacing / new_spacing).astype(int)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(new_spacing / old_spacing)

    new_shape = np.round(old_shape * old_spacing / new_spacing).astype(int)
    return new_shape, new_affine


def clean_image(image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    return image_data * mask_data


def z_score_standardization_image(image_data: np.ndarray) -> np.ndarray:
    non_zero_mask = image_data != 0
    non_zero_data = image_data[non_zero_mask]
    
    data_mean = non_zero_data.mean()
    data_std = non_zero_data.std()
    
    # standardize only non-zero values to [0, 1]
    standardized_data = image_data.copy()
    standardized_data[non_zero_mask] = (non_zero_data - data_mean) / data_std

    return standardized_data


def extract_roi(image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    non_zero_coords = np.array(np.nonzero(mask_data))  # Shape: (3, N) where N is the number of non-zero voxels
    min_coords = non_zero_coords.min(axis=1)  # (x_min, y_min, z_min)
    max_coords = non_zero_coords.max(axis=1) + 1  # (x_max, y_max, z_max), +1 for inclusive slicing

    # Extract the ROI from the CT image using the bounding box
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    roi_data = image_data[x_min:x_max, y_min:y_max, z_min:z_max]
    return roi_data


def preprocess_image(image_path: str, mask_path: str, output_path: str, target_spacing: Tuple[float, float, float]) -> None:
    image = nib.load(image_path)
    mask = nib.load(mask_path)

    # mask:0 nearest neighbor interpolation
    # image: 1 linear interpolation
    target_shape, img_affine_new = get_resample_target_info(image, target_spacing)
    _, mask_affine_new = get_resample_target_info(mask, target_spacing)
    # Only keep the target shape the same, while the affine is different(mask and image are different in the affine matrix)
    
    image_resampled = resample_from_to(image, (target_shape, img_affine_new), order=1) 
    mask_resampled = resample_from_to(mask, (target_shape, mask_affine_new), order=0)
    
    mask_resampled_array = mask_resampled.get_fdata()
    image_resampled_array = image_resampled.get_fdata()
    # Clean
    image_cleaned = clean_image(image_resampled_array, mask_resampled_array)

    # Extract ROI
    image_roi = extract_roi(image_cleaned, mask_resampled_array)

    # Normalize
    image_normalized = z_score_standardization_image(image_roi)

    # image_data = image.get_fdata()
    # mask_data = mask.get_fdata()
    # processed_image = clean_image(image_data, mask_data)
    # nib.save(processed_image, output_path)
    image_preprocessed = image_normalized
    image_preprocessed = nib.Nifti1Image(image_preprocessed, image_resampled.affine)
    nib.save(image_preprocessed, output_path)


def save_visualization(saved_nii_path: str, out_vis_dir: str, name: str):
    """
    Saves a 2D axial PNG preview of the slice with the largest mask area.
    """
    saved_img = nib.load(saved_nii_path)
    data = saved_img.get_fdata()

    total_slices = data.shape[2]
    # Count non-zero mask voxels across each 2D axial slice (axis 0 and 1)
    slice_counts = (data != 0).sum(axis=(0, 1))

    if slice_counts.max() > 0:
        slice_idx = int(np.argmax(slice_counts))
    else:
        slice_idx = total_slices // 2

    img_slice = data[:, :, slice_idx]

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(np.fliplr(np.rot90(img_slice)), cmap="gray")
    plt.axis("off")

    vis_filename = f"{name}_slice_{slice_idx}_of_{total_slices}.png"
    vis_path = os.path.join(out_vis_dir, vis_filename)
    plt.savefig(vis_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_single_case(item_dict: dict, target_spacing: Tuple[float, float, float], out_images_dir: str, out_vis_dir: str, no_vis: bool):
    try:
        name = item_dict["name"]
        clean_name = name.replace("AHN", "ahn_")
        output_path = os.path.join(out_images_dir, f"{clean_name}.nii.gz")

        # Calls your exact function
        preprocess_image(
            image_path=item_dict["image"],
            mask_path=item_dict["mask"],
            output_path=output_path,
            target_spacing=target_spacing,
        )

        # Visualization
        if not no_vis:
            save_visualization(output_path, out_vis_dir, clean_name)

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
        mask_path = os.path.join(masks_dir, f"{name}.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, f"{name}.nii")

        if os.path.exists(mask_path):
            data_list.append({"image": img_path, "mask": mask_path, "name": name})
        else:
            print(f"[Warning] Mask missing for image: {img_path}")

    return data_list


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Crop & Preprocess Pipeline")
    parser.add_argument("-i", "--input", type=str, default="./Cyst-X-RAS/segmentation", help="Input root directory")
    parser.add_argument("-o", "--output", type=str, default="./Cyst-X-RAS/classification", help="Output root directory")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("-t", "--t", default=None, type=int, choices=[1, 2], help="Modality (1 or 2). Omit for both.")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), help="Target spacing in mm")
    parser.add_argument("--no-vis", action="store_true", help="Disable PNG slice visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    t_list = [1, 2] if args.t is None else [args.t]
    target_spacing = tuple(args.spacing)

    for t in t_list:
        mod_input_dir = os.path.join(args.input, f"t{t}")
        mod_output_dir = os.path.join(args.output, f"t{t}")

        images_dir = os.path.join(mod_input_dir, "images")
        masks_dir = os.path.join(mod_input_dir, "masks")
        out_images_dir = os.path.join(mod_output_dir, "images")
        out_vis_dir = os.path.join(mod_output_dir, "visualizations")

        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"[Warning] Directory missing for t{t}. Skipping.")
            continue

        os.makedirs(out_images_dir, exist_ok=True)
        if not args.no_vis:
            os.makedirs(out_vis_dir, exist_ok=True)

        data_list = get_file_pairs(images_dir, masks_dir)
        print(f"\nProcessing t{t}: Found {len(data_list)} paired cases.")
        if len(data_list) == 0:
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_case,
                    item,
                    target_spacing,
                    out_images_dir,
                    out_vis_dir,
                    args.no_vis,
                ): item["name"]
                for item in data_list
            }

            with tqdm(total=len(data_list), desc=f"Preprocessing t{t} ({args.workers} workers)") as pbar:
                for future in as_completed(futures):
                    success, name, error_msg = future.result()
                    if not success:
                        tqdm.write(f"[Error] Failed to process {name}: {error_msg}")
                    pbar.update(1)


if __name__ == "__main__":
    main()