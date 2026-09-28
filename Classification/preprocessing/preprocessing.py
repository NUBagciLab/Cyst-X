import nibabel as nib
import numpy as np
from typing import Tuple
from nibabel.processing import resample_from_to

def get_resample_target_info(img: nib.Nifti1Image,target_spacing):
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



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess a single image and mask pair.")
    parser.add_argument('-i', '--image', default="/dataset/IPMN_images_masks/t1/images/northwestern_0018.nii.gz", type=str, help="Input CT image (.nii.gz)")
    parser.add_argument('-m', '--mask', default="/dataset/IPMN_images_masks/t1/masks/northwestern_0018.nii.gz", type=str, help="Input segmentation mask (.nii.gz)")
    parser.add_argument('-o', '--output', default="./northwestern_0018.nii.gz", type=str, help="Output preprocessed image (.nii.gz)")
    parser.add_argument('--spacing', nargs=3, type=float, default=(1.0, 1.0, 1.0),
                        help="Target spacing in mm (default: 1.0 1.0 1.0)")
    
    args = parser.parse_args()
    
    preprocess_image(
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.output,
        target_spacing=tuple(args.spacing)
    )