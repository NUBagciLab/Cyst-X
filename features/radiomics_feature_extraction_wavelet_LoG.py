import os
import csv
import tqdm
from radiomics import featureextractor
import argparse
import SimpleITK as sitk

def extract_features_from_folder(input_folder: str, mask_folder: str, output_path: str, geometry_tolerance: float = 1e-6, label_values: list = None):
    """
    Extracts radiomic features for all image files in the input folder and saves them to a CSV file.
    If label_values is provided, extracts features separately for each label value and combines them into one row per image.
    Otherwise, extracts features using default PyRadiomics behavior (all non-zero voxels).

    Parameters:
    - input_folder (str): Path to the folder containing image files.
    - mask_folder (str): Path to the folder containing mask files (corresponding to each image).
    - output_path (str): Path where the CSV file will be saved.
    - geometry_tolerance (float): Tolerance for image/mask geometry mismatch (default: 1e-6).
    - label_values (list, optional): List of label values to extract features for. If None, uses default behavior.
    """
    
    # Initialize PyRadiomics feature extractor with advanced settings and geometry tolerance
    params = {'geometryTolerance': geometry_tolerance}
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    
    # Enable more advanced features
    extractor.enableAllImageTypes()  # Enables all default image types
    extractor.enableImageTypeByName('Wavelet')  # Enable wavelet features
    extractor.enableImageTypeByName('LoG')  # Enable Laplacian of Gaussian (LoG) features
    
    # Enable all available feature classes
    extractor.enableAllFeatures()
    
    if label_values:
        print(f"Feature extractor initialized with Wavelet and LoG transformations.")
        print(f"Extracting features for label values: {label_values}")
    else:
        print("Feature extractor initialized with Wavelet and LoG transformations.")
        print("Using default feature extraction (all non-zero voxels).")

    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as csv_file:
        writer = None
        
        # Loop through each file in the input folder
        for filename in tqdm.tqdm(os.listdir(input_folder)):
            # Process only image files
            if filename.endswith(('.nii', '.nii.gz', '.mha', '.mhd', '.dcm')):
                image_path = os.path.join(input_folder, filename)
                mask_path = os.path.join(mask_folder, filename)  # Assuming mask names match image names

                # Ensure mask file exists
                if not os.path.isfile(mask_path):
                    print(f"Mask file not found for {filename}, skipping...")
                    continue
                
                # If label_values is provided, extract features for each label separately and combine
                if label_values:
                    # Collect all features from all labels for this image
                    combined_features = {'Image': filename}
                    
                    for label_value in label_values:
                        try:
                            # Set the label parameter for this extraction
                            extractor.settings['label'] = label_value
                            features = extractor.execute(image_path, mask_path)
                            
                            # Rename features with label suffix
                            for feature_name, feature_value in features.items():
                                combined_features[f"{feature_name}_label{label_value}"] = feature_value
                            
                        except Exception as e:
                            print(f"Error extracting features from {filename} for label {label_value}: {e}")
                            # Continue with other labels even if one fails
                            continue
                    
                    # Write headers and feature values to CSV
                    if writer is None:
                        # Write the header with all combined feature names
                        writer = csv.DictWriter(csv_file, fieldnames=list(combined_features.keys()))
                        writer.writeheader()
                    
                    # Write the row for the current image (one row with all label features)
                    writer.writerow(combined_features)
                    
                else:
                    # Default behavior: extract features without specifying label
                    try:
                        # Remove label setting if it exists
                        if 'label' in extractor.settings:
                            del extractor.settings['label']
                        features = extractor.execute(image_path, mask_path)
                        
                        # Write headers and feature values to CSV
                        if writer is None:
                            # Write the header without Label column
                            writer = csv.DictWriter(csv_file, fieldnames=['Image'] + list(features.keys()))
                            writer.writeheader()
                        
                        # Write the row for the current image
                        writer.writerow({'Image': filename, **features})
                        
                    except Exception as e:
                        print(f"Error extracting features from {filename}: {e}")
                        continue
    
    print(f"Radiomics features saved to {output_path}")

def extract_voxel_based_features_from_folder(input_folder: str, mask_folder: str, output_path: str, geometry_tolerance: float = 1e-6, label_values: list = None):
    """
    Extracts voxel-based radiomic features for all image files in the input folder and saves them to nifti files in the output directory.
    If label_values is provided, extracts features separately for each label value.
    Otherwise, extracts features using default PyRadiomics behavior.

    Parameters:
    - input_folder (str): Path to the folder containing image files.
    - mask_folder (str): Path to the folder containing mask files (corresponding to each image).
    - output_path (str): Path to the directory where the nifti files will be saved.
    - geometry_tolerance (float): Tolerance for image/mask geometry mismatch (default: 1e-6).
    - label_values (list, optional): List of label values to extract features for. If None, uses default behavior.
    """
    
    # Initialize PyRadiomics feature extractor with advanced settings
    settings = {
        'padDistance': 0,   # no padding
        'geometryTolerance': geometry_tolerance  # Increase tolerance for geometry mismatches
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # Enable all features first
    extractor.enableAllFeatures()

    if label_values:
        print("Feature extractor initialized for voxel-based extraction.")
        print(f"Extracting features for label values: {label_values}")
    else:
        print("Feature extractor initialized for voxel-based extraction.")
        print("Using default feature extraction (all non-zero voxels).")

    # Loop through each file in the input folder
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        # Process only image files
        if filename.endswith(('.nii', '.nii.gz', '.mha', '.mhd', '.dcm')):
            image_path = os.path.join(input_folder, filename)
            mask_path = os.path.join(mask_folder, filename)  # Assuming mask names match image names
            
            # Ensure mask file exists
            if not os.path.isfile(mask_path):
                print(f"Mask file not found for {filename}, skipping...")
                continue
            
            # If label_values is provided, extract features for each label separately
            if label_values:
                for label_value in label_values:
                    output_dir = os.path.join(output_path, filename, f"label_{label_value}")
                    
                    # Create output directory if it doesn't exist
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        print(f"Created directory {output_dir}")
                    else:
                        print(f"Directory {output_dir} already exists")
                        continue
                    
                    # Extract features
                    try:
                        print(f"Processing {filename} for label {label_value}...")
                        # Set the label parameter for this extraction
                        extractor.settings['label'] = label_value
                        results = extractor.execute(image_path, mask_path, voxelBased=True)
                        feature_count = 0
                        for name, val in results.items():
                            if isinstance(val, sitk.Image):
                                out_path = os.path.join(output_dir, f"{name}.nii.gz")
                                sitk.WriteImage(val, out_path)
                                feature_count += 1
                        print(f"Saved {feature_count} voxel-based features for {filename}, label {label_value}")
                    except Exception as e:
                        print(f"Error extracting features from {filename} for label {label_value}: {e}")
                        continue
            else:
                # Default behavior: extract features without specifying label
                output_dir = os.path.join(output_path, filename)
                
                # Create output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory {output_dir}")
                else:
                    print(f"Directory {output_dir} already exists")
                    continue
                
                # Extract features
                try:
                    print(f"Processing {filename}...")
                    # Remove label setting if it exists
                    if 'label' in extractor.settings:
                        del extractor.settings['label']
                    results = extractor.execute(image_path, mask_path, voxelBased=True)
                    feature_count = 0
                    for name, val in results.items():
                        if isinstance(val, sitk.Image):
                            out_path = os.path.join(output_dir, f"{name}.nii.gz")
                            sitk.WriteImage(val, out_path)
                            feature_count += 1
                    print(f"Saved {feature_count} voxel-based features for {filename}")
                except Exception as e:
                    print(f"Error extracting features from {filename}: {e}")
                    continue
    
    print(f"Radiomics features saved to {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract radiomic features from all images in a folder and save them to a CSV.")
    parser.add_argument('-i', '--input_folder', required=True, help="Folder containing image files.")
    parser.add_argument('-m', '--mask_folder', required=True, help="Folder containing mask files corresponding to images.")
    parser.add_argument('-o', '--output_path', default='radiomics_features.csv', help="Output path for the CSV file (default: radiomics_features.csv).")
    parser.add_argument('-v', '--voxelBased', action='store_true', help="Enable voxel-based features.")
    parser.add_argument('-t', '--geometry_tolerance', type=float, default=1e-6, help="Tolerance for image/mask geometry mismatch (default: 1e-6). Increase this value if you encounter geometry mismatch errors.")
    parser.add_argument('-l', '--label_values', nargs='+', type=int, default=None, help="Label values to extract features for (e.g., -l 1 2 3). If not provided, uses default behavior.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.voxelBased:
        extract_voxel_based_features_from_folder(input_folder=args.input_folder, mask_folder=args.mask_folder, output_path=args.output_path, geometry_tolerance=args.geometry_tolerance, label_values=args.label_values)
    else:
        extract_features_from_folder(input_folder=args.input_folder, mask_folder=args.mask_folder, output_path=args.output_path, geometry_tolerance=args.geometry_tolerance, label_values=args.label_values)