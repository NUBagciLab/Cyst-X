# Radiomics IPMN Classification

Radiomics pipeline for binary IPMN classification (None+LDG vs HDG) from T1/T2 MRI. The workflow is: **extract features → train with mRMR + Random Forest → infer on test sets → report metrics by center**.

## Requirements

```bash
pip install pyradiomics SimpleITK numpy pandas scikit-learn tqdm
```

Python 3.8+ recommended.

## Project Structure

```
Radiomics/
├── radiomics_feature_extraction_wavelet_LoG.py   # Step 1: feature extraction
├── radiomics_classification_mRMR.py              # Step 2: training pipeline
├── radiomics_inference.py                        # Step 3: fold-wise inference
├── calculate_metrics_by_center.py                # Step 4: evaluation
├── radiomics_mRMR.sh                             # Batch training launcher
├── radiomics_inference.sh                        # Batch inference launcher
├── features/
│   ├── radiomics_features_t1.csv                 # Pre-extracted T1 features
│   └── radiomics_features_t2.csv                 # Pre-extracted T2 features
├── splits_internal_t1.json                       # Internal CV splits (T1, 4 folds)
├── splits_internal_t2.json                       # Internal CV splits (T2, 4 folds)
├── splits_external_t1.json                       # External validation splits (T1, 7 folds)
└── splits_external_t2.json                       # External validation splits (T2, 7 folds)
```

---

## File Descriptions

### Scripts

| File | Purpose |
|------|---------|
| `radiomics_feature_extraction_wavelet_LoG.py` | Extracts PyRadiomics features (Wavelet + LoG) from image/mask pairs and saves them to CSV or NIfTI. |
| `radiomics_classification_mRMR.py` | Full training pipeline per fold: hyperparameter search → mRMR feature stability (bootstrap) → final model → test evaluation. |
| `radiomics_inference.py` | Rebuilds each fold's model from saved configs and runs predictions on the test split. |
| `calculate_metrics_by_center.py` | Computes accuracy and AUC (with 95% CI) grouped by hospital center and globally. |
| `radiomics_mRMR.sh` | Runs `radiomics_classification_mRMR.py` for multiple folds in parallel. |
| `radiomics_inference.sh` | Runs `radiomics_inference.py` for folds 0–3 on the internal T2 split. |

### Data

| File | Purpose |
|------|---------|
| `features/radiomics_features_t1.csv` | One row per case; `Image` column + thousands of radiomics columns (suffix `_label1`). |
| `features/radiomics_features_t2.csv` | Same format for T2 modality (~722 / ~737 cases). |
| `splits_internal_t{1,2}.json` | 4-fold cross-validation splits for internal evaluation. |
| `splits_external_t{1,2}.json` | 7-fold splits for external / multi-center validation. |

### Split JSON format

Each split file contains:

- `label_mapping`: class names → 0/1 labels
- `folds`: list of `{fold, train, test}` where each sample is `{image, label}`

Image filenames must match the `Image` column in the feature CSV.

---

## How to Run

Run all commands from the project root.

### Step 1 — Feature extraction (optional)

Skip this if you use the pre-built CSVs in `features/`.

```bash
python radiomics_feature_extraction_wavelet_LoG.py \
  -i /path/to/images \
  -m /path/to/masks \
  -o features/radiomics_features_t2.csv \
  -l 1
```

| Flag | Description |
|------|-------------|
| `-i` | Folder with images (`.nii`, `.nii.gz`, `.mha`, `.mhd`, `.dcm`) |
| `-m` | Folder with matching mask filenames |
| `-o` | Output CSV path |
| `-l 1` | Mask label value to extract (optional) |
| `-v` | Voxel-based output as NIfTI instead of CSV |
| `-t` | Geometry tolerance (default `1e-6`) |

### Step 2 — Train models

Single fold:

```bash
python radiomics_classification_mRMR.py \
  -f 0 \
  --features-path ./features/radiomics_features_t2.csv \
  --split-path ./splits_internal_t2.json \
  --output-dir ./results/t2_mRMR_internal
```

| Flag | Description |
|------|-------------|
| `-f` | Fold index (required) |
| `--features-path` | Feature CSV |
| `--split-path` | Split JSON |
| `--output-dir` | Where fold results are saved |

**Output per fold** (`<output-dir>/fold_<N>/`):

- `results.json` — metrics and metadata
- `final_model.pkl`, `final_scaler.pkl`
- `final_selected_features.json`, `final_training_config.json`
- `feature_stability.csv`

Batch training (edit paths in the script first):

```bash
bash radiomics_mRMR.sh
```

### Step 3 — Inference

Single fold:

```bash
python radiomics_inference.py \
  -f 0 \
  --features ./features/radiomics_features_t2.csv \
  --split ./splits_internal_t2.json \
  --model-dir ./results/t2_mRMR_internal \
  -o ./results/t2_mRMR_internal/predictions_fold0.csv
```

Output columns: `ID`, `Label`, `Prediction`, `Probability`, `Fold`.

Batch inference:

```bash
bash radiomics_inference.sh
```

> Update hard-coded paths in `radiomics_mRMR.sh` and `radiomics_inference.sh` before running on your machine.

### Step 4 — Metrics by center

Merge per-fold prediction CSVs first, then:

```bash
python calculate_metrics_by_center.py \
  --input_csv ./results/t2_mRMR_internal/predictions_all_folds.csv \
  --output_prefix metrics_t2_internal
```

**Outputs:**

- `metrics_t2_internal_center_summary.csv` — per-center accuracy / AUC
- `metrics_t2_internal_global_summary.csv` — overall metrics
- `*_center_fold_metrics.csv`, `*_global_fold_metrics.csv` — per-fold breakdown (multi-fold only)

Center names are parsed from case IDs (e.g. `CAD_119` → `CAD_MCF`).

---

## Typical Workflow

```
Images + Masks
      │
      ▼
radiomics_feature_extraction_wavelet_LoG.py  →  features/*.csv
      │
      ▼
radiomics_classification_mRMR.py  →  results/fold_*/  (train + evaluate)
      │
      ▼
radiomics_inference.py  →  predictions_fold*.csv
      │
      ▼
calculate_metrics_by_center.py  →  center / global summary CSVs
```

---

## Notes

- **Modality**: Replace `t1` / `t2` in paths to switch MRI sequences.
- **Internal vs external**: Use `splits_internal_*.json` (4 folds) or `splits_external_*.json` (7 folds).
- **Training time**: Hyperparameter search and 1000 bootstrap iterations per fold are compute-heavy; use `radiomics_mRMR.sh` to parallelize folds.
- **Default paths** in Python scripts point to a remote server (`/data2/...`, `/home/pyq6817/...`). Override them via CLI flags when running locally.
