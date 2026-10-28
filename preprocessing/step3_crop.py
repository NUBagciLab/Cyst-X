"""
Prepare RAS-oriented pancreas crops for IPMN classification from MRI and masks.
Resample images and masks to a shared grid, crop to the foreground bounding box,
and standardize nonzero intensities, with axial, sagittal, and coronal previews.
"""
import argparse
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm


def get_resample_target_info(img: nib.Nifti1Image, target_spacing):
    new_spacing = np.asarray(target_spacing, dtype=float)
    if new_spacing.shape != (3,) or not np.all(np.isfinite(new_spacing)) or np.any(new_spacing <= 0):
        raise ValueError("Target spacing must contain three finite positive values.")
    old_spacing = nib.affines.voxel_sizes(img.affine)
    new_shape = np.maximum(1, np.round(np.array(img.shape) * old_spacing / new_spacing).astype(int))
    new_affine = img.affine.copy()
    new_affine[:3, :3] = img.affine[:3, :3] @ np.diag(new_spacing / old_spacing)
    return tuple(new_shape), new_affine


def clean_image(image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    return image_data * mask_data


def z_score_standardization_image(image_data: np.ndarray) -> np.ndarray:
    non_zero_mask = image_data != 0
    non_zero_data = image_data[non_zero_mask]
    standardized_data = image_data.copy()
    if non_zero_data.size == 0:
        return standardized_data
    data_mean = non_zero_data.mean()
    data_std = non_zero_data.std()
    # Z-scores have mean zero and unit variance; keep background zero.
    standardized_data[non_zero_mask] = (non_zero_data - data_mean) / data_std if data_std > 0 else 0
    return standardized_data


def roi_slices(mask_data: np.ndarray):
    coords = np.array(np.nonzero(mask_data))
    if coords.shape[1] == 0:
        raise ValueError("Mask has no foreground voxels on the image grid.")
    return tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(coords.min(axis=1), coords.max(axis=1)))


def extract_roi(image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    return image_data[roi_slices(mask_data)]


def require_ras(image: nib.Nifti1Image, source: str) -> None:
    """Check the orientation established by step 1 without changing voxel data."""
    orientation = nib.aff2axcodes(image.affine)
    if orientation != ("R", "A", "S"):
        raise ValueError(
            f"Expected RAS orientation for {source}, got {orientation}. "
            "Use step 1 RAS outputs (or step 2 images that preserve that orientation)."
        )


def preprocess_image(image_path: str, mask_path: str, output_path: str, target_spacing: Tuple[float, float, float]) -> None:
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    require_ras(image, image_path)
    require_ras(mask, mask_path)
    target_grid = get_resample_target_info(image, target_spacing)
    image_resampled = resample_from_to(image, target_grid, order=1)
    mask_resampled = resample_from_to(mask, target_grid, order=0)
    mask_array = mask_resampled.get_fdata() > 0
    image_cleaned = clean_image(image_resampled.get_fdata(), mask_array)
    slices = roi_slices(mask_array)
    image_normalized = z_score_standardization_image(image_cleaned[slices])

    # The crop's first voxel must retain its original physical location.
    crop_affine = image_resampled.affine.copy()
    crop_affine[:3, 3] = nib.affines.apply_affine(
        image_resampled.affine, [sl.start for sl in slices]
    )
    nib.save(nib.Nifti1Image(image_normalized.astype(np.float32), crop_affine), output_path)


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
    parser.add_argument("-i", "--input", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-n4/segmentation"), help="Input image root directory (step 2 output)")
    parser.add_argument("-m", "--mask-input", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-RAS/segmentation"), help="Mask root directory (step 1 output); expects t1/masks and t2/masks")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(PROJECT_ROOT, "Cyst-X-n4/classification"), help="Output root directory")
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
        masks_dir = os.path.join(args.mask_input, f"t{t}", "masks")
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
