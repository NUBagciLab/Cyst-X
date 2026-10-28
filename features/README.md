# Radiomics features and optional extraction

The precomputed feature tables are provided for the current Cyst-X radiomics experiments. **Feature extraction is optional: skip it when using these CSVs.**

| File | Cases | Contents |
|---|---:|---|
| `radiomics_features_t1.csv` | 723 | T1W radiomics features |
| `radiomics_features_t2.csv` | 738 | T2W radiomics features |
| `radiomics_feature_extraction_wavelet_LoG.py` | — | Optional extraction from image/mask pairs |

Each table has an `Image` column identifying the image filename and radiomics/diagnostic columns. The classification loaders select the eligible labeled cases for each task and use the current patient splits. These full feature-table counts therefore differ from the task-specific evaluation cohort sizes.

## Use the provided features

The radiomics scripts under `Classification/internal/` and `Classification/external/` look for these tables under the repository's root `features/` directory by default. To use a different table, pass `--features-path /path/to/radiomics_features_t2.csv` to the corresponding radiomics training or evaluation Python script. See the [classification README](../Classification/README.md) and each radiomics `USAGE.txt` for the current five-fold internal or seven-center external workflow.

## Feature extraction (optional)

This script and the extraction instructions are carried over from [v1's radiomics implementation](https://github.com/NUBagciLab/Cyst-X/tree/v1/Classification/Radiomics). The Python script is copied unchanged from v1; the former four-fold training and inference scripts are not needed for the current workflow.

Extract features only when processing new data or intentionally regenerating the tables. Install the extraction dependencies in your Python environment:

```bash
pip install pyradiomics SimpleITK numpy pandas scikit-learn tqdm
```

Run from the repository root:

```bash
python features/radiomics_feature_extraction_wavelet_LoG.py \
  -i /path/to/images \
  -m /path/to/masks \
  -o features/recomputed_radiomics_features_t2.csv \
  -l 1
```

Use a new output filename to retain the supplied table. Substitute T1W paths and an appropriate output filename to extract T1W features.

| Flag | Description |
|---|---|
| `-i`, `--input_folder` | Image folder; supports `.nii`, `.nii.gz`, `.mha`, `.mhd`, and `.dcm` |
| `-m`, `--mask_folder` | Mask folder with filenames matching the images |
| `-o`, `--output_path` | Output CSV path; with `-v`, the output directory for feature maps |
| `-l`, `--label_values` | One or more mask labels, for example `-l 1`; label-specific feature names receive a suffix such as `_label1` |
| `-v`, `--voxelBased` | Extract voxel-based feature maps as NIfTI files instead of a case-level CSV |
| `-t`, `--geometry_tolerance` | Image/mask geometry tolerance; default `1e-6` |

Images and masks must be aligned in physical space. For the current preprocessing pipeline, N4-corrected full-volume images are under `Cyst-X-n4/segmentation/t{1,2}/images`, with matching masks under `Cyst-X-RAS/segmentation/t{1,2}/masks`. See the [preprocessing README](../preprocessing/README.md). Missing masks and extraction failures are reported by the script, so check its messages and output case IDs before using a regenerated table.
