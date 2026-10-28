# Cyst-X classification — v2

This directory contains the classification code from Cyst-X-v2.3. The original GitHub release is preserved by the `v1` tag; current development is on `main`. Internal binary experiments use five folds in v2, replacing the four-fold binary setup in v1. External experiments use seven leave-one-center-out models.

The current preprocessing aligns image/mask orientation and applies N4 bias-field correction before generating the classification ROIs. See [preprocessing](../preprocessing/README.md), the [dataset](https://huggingface.co/datasets/phy710/Cyst-X), and the [model archives](https://huggingface.co/phy710/Cyst-X).

## Organization

```text
Classification/
├── internal/
│   ├── 3-class/centralized/
│   ├── 2-class/                 # Centralized, FedAvg, FedProx, radiomics, four fusion methods
│   ├── low_vs_high/             # Centralized, radiomics, four fusion methods
│   └── histology/               # Centralized, radiomics, four fusion methods
├── external/
│   ├── 2-class/
│   ├── low_vs_high/
│   └── histology/
├── tsne/
├── vs_radiologists/
├── run_radiomics.sh
└── requirements.txt
```

`2-class` compares high-risk with no-/low-risk cases. `low_vs_high` includes only low- and high-risk IPMN. `histology` further applies the existing `keep_histology` rule, which excludes cases whose follow-up information contains `no change`. Fusion methods are `early_fusion`, `late_fusion`, `siamese_fusion`, and `logit_fusion`.

## Environment and data

Install a CUDA-compatible PyTorch build for your machine, then install the Python dependencies in your environment:

```bash
pip install -r Classification/requirements.txt
```

Use the dataset repository's `classification/` directory, which contains the cropped images, label workbooks, and current `t1_split.csv` and `t2_split.csv`. The paired fusion loaders use the T1 patient fold assignments. Prefer these split files over the legacy `data splits/` files. For a download under this repository's `Cyst-X/` directory, set an absolute data path from the repository root:

```bash
export DATA_PATH="$(pwd)/Cyst-X/classification"
```

The released classification ROIs are already generated from N4-corrected images. To regenerate them from the full-volume segmentation data, follow the preprocessing README. If using regenerated crops, point `DATA_PATH` at `Cyst-X-n4/classification` instead. Pass `--data-path` explicitly; some experiment scripts retain historical dataset defaults.

## Training and test evaluation

Run experiment commands from that experiment's directory. For example:

```bash
cd Classification/internal/2-class/centralized
CUDA_VISIBLE_DEVICES=0 bash ./main.sh --data-path "$DATA_PATH" --model densenet121 -t 1
```

DenseNet-121 is the default. The internal centralized 2-class and 3-class experiments also support `resnet34`, `resnet50`, and `efficientnet_b0`. `-t 1` and `-t 2` select T1W and T2W for single-modality scripts. Fusion scripts consume paired inputs; consult their `USAGE.txt`. FedProx accepts `--mu` (0.1 by default; the other reported setting is 0.3).

`main.sh` trains the folds and evaluates them; do not use it when you only want to test downloaded checkpoints. To evaluate existing binary models, run:

```bash
python fold_test.py --data-path "$DATA_PATH" --model densenet121 --t 1
python fold_test.py --data-path "$DATA_PATH" --model densenet121 --t 1 -c
```

These save `result.txt/.xlsx` and `result_calibrated.txt/.xlsx`. The test scripts may also generate ROC PDFs. Thresholds are selected from validation data by `calibrate.py`, seeking specificity of at least 0.90 without a sensitivity floor. To reuse published thresholds, keep `thresholds.json` beside the corresponding saved folds; some test scripts otherwise invoke calibration automatically.

## Validation results

For binary experiments, evaluate each model's validation partition using:

```bash
python fold_test_val.py --data-path "$DATA_PATH" --model densenet121 --t 1
python fold_test_val.py --data-path "$DATA_PATH" --model densenet121 --t 1 -c
```

The outputs are `val_result.txt/.xlsx` and `val_result_calibrated.txt/.xlsx`, with no ROC PDFs. Here `-c` only loads existing `thresholds.json`; missing thresholds cause an error. The workbook has five `Fold_1`–`Fold_5` sheets internally or seven `Center_1`–`Center_7` sheets externally. An external sheet identifies the model's held-out center; its validation patients come from the other centers. Patients can appear in validation for more than one model, so retain the fold identity in analysis.

## Pretrained models

Each experiment has a `saved.tar.gz` archive on Hugging Face. Download the classification archives into the matching code directories, for example from the repository root:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="phy710/Cyst-X",
    repo_type="model",
    revision="v2",
    allow_patterns=["Classification/*/*/*/saved.tar.gz"],
    local_dir=".",
)
```

In each experiment directory, `tar -xzf saved.tar.gz` restores its `saved/` folder. Single-input weights are under `saved/<model>/t1/` or `t2/`; fusion weights are under `saved/<model>/`. FedProx model directories include the coefficient, for example `densenet121_0.3/t1/`.

## Radiomics, readers, and t-SNE

Radiomics uses precomputed feature CSVs named `radiomics_features_t1.csv` and `radiomics_features_t2.csv`. Supply them under the repository's `features/` directory or pass `--features-path` to the radiomics Python scripts. The precomputed tables are included in the root [features folder](../features/README.md), together with the optional v1 extraction script. Feature extraction is not required when using the provided CSVs. See each radiomics `USAGE.txt`; its model name is `random_forest` and its CLI uses `-t` rather than `--model`.

The [reader comparison](vs_radiologists/README.md) reads saved test predictions for the 629 paired, 474 IPMN-only, and 357 histology reader cohorts. Its included CSV and workbook are reference inputs, rather than model outputs. Run from the repository root with `bash Classification/vs_radiologists/run.sh`; outputs are written to `results/tables/`.

`tsne/tsne.py --data-path "$DATA_PATH"` visualizes image inputs. The centralized 2-class `tsne.py` visualizes learned features using saved models. Run each script from its own directory.

Checkpoints, prediction workbooks, generated plots, caches, server logs, and local experiment backups are excluded from this code checkout.
