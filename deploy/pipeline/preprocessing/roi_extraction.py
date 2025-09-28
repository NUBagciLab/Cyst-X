import nibabel as nib
import numpy as np
import argparse
import os

def process_and_extract_roi(image_path, segmentation_path, output_path):
    """
    Process image through normalization, masking, and ROI extraction.
    
    Args:
        image_path (str): Path to the input image (.nii.gz file).
        segmentation_path (str): Path to the segmentation mask (.nii.gz file).
        output_path (str): Path to save the ROI image (.nii.gz file).
    """
    try:
        # Load the image
        img = nib.load(image_path)
        img_data = img.get_fdata()
        
        # Step 1: Normalize the data to the range [0, 1]
        data_min = img_data.min()
        data_max = img_data.max()
        if data_max > data_min:
            normalized_data = (img_data - data_min) / (data_max - data_min)
        else:
            normalized_data = img_data
        
        # Load the segmentation mask
        seg_img = nib.load(segmentation_path)
        seg_data = seg_img.get_fdata()
        
        # Ensure the dimensions match
        if normalized_data.shape != seg_data.shape:
            raise ValueError("Image and segmentation mask must have the same dimensions.")
        
        # Step 2: Apply mask (clean)
        masked_data = normalized_data * seg_data
        
        # Step 3: Extract ROI based on the non-zero region of the mask
        non_zero_coords = np.array(np.nonzero(seg_data))
        if non_zero_coords.size == 0:
            print(f"Warning: No non-zero voxels found in mask for {image_path}")
            return
        
        min_coords = non_zero_coords.min(axis=1)
        max_coords = non_zero_coords.max(axis=1) + 1
        
        # Extract the ROI from the masked data using the bounding box
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords
        roi_data = masked_data[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Save the ROI as a new NIfTI image
        roi_affine = img.affine
        roi_header = img.header.copy()
        roi_img = nib.Nifti1Image(roi_data, roi_affine, roi_header)
        
        nib.save(roi_img, output_path)
        print(f"ROI saved: {output_path}")
        print(f"ROI bounds: x[{x_min}:{x_max}], y[{y_min}:{y_max}], z[{z_min}:{z_max}]")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def get_segmentation_path(image_filename, mask_dir):
    """
    Convert image filename to segmentation filename.
    Example: CAD_28_T1_0000.nii.gz -> CAD_28_T1.nii.gz
    """
    if image_filename.endswith('_0000.nii.gz'):
        seg_filename = image_filename.replace('_0000.nii.gz', '.nii.gz')
    else:
        seg_filename = image_filename
    
    return os.path.join(mask_dir, seg_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROI Extraction Pipeline")
    parser.add_argument('-i', '--input-dir', required=True, type=str, help="Input images directory")
    parser.add_argument('-m', '--mask-dir', required=True, type=str, help="Segmentation masks directory")
    parser.add_argument('-o', '--output-dir', required=True, type=str, help="Output directory for ROI images")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all .nii.gz files in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(args.input_dir, filename)
            segmentation_path = get_segmentation_path(filename, args.mask_dir)
            output_path = os.path.join(args.output_dir, filename)
            
            # Check if segmentation file exists
            if os.path.exists(segmentation_path):
                process_and_extract_roi(image_path, segmentation_path, output_path)
            else:
                print(f"Warning: Segmentation file not found: {segmentation_path}")
