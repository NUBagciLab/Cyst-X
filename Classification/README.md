# Cyst-X classification

This directory contains the Cyst-X classification code. Internal binary experiments use five folds. External experiments use seven leave-one-center-out models.

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
└── run_radiomics.sh
```

`2-class` compares high-risk with no-/low-risk cases. `low_vs_high` includes only low- and high-risk IPMN. `histology` further applies the existing `keep_histology` rule, which excludes cases whose follow-up information contains `no change`. Fusion methods are `early_fusion`, `late_fusion`, `siamese_fusion`, and `logit_fusion`.

## Environment and data

Use the shared environment described in the [root README](../README.md#environment-and-requirements). Install the root [requirements.txt](../requirements.txt) from the repository root:

```bash
python -m pip install -r requirements.txt
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

## Additional supplementary analyses

[additional_analyses/](additional_analyses/) contains the independent, CPU-only supplementary analyses of saved held-out predictions. It reproduces reader sensitivity/specificity confidence intervals, paired model-minus-reader differences, evaluation on the common 357-patient histopathology subset, and matched pooled/center-mean AUC summaries. No retraining or recalibration is performed.

The package includes aggregate example outputs, accepted reference results, validation notes, and a checksum manifest. From the repository root run:

```bash
python -m pip install -r requirements.txt
python Classification/additional_analyses/run_analysis.py \
  --classification-root ./Classification \
  --labels ./Classification/vs_radiologists/IPMN_labels_total.xlsx \
  --readers ./Classification/vs_radiologists/Cyst-X_bigdata_risk_assessment.csv \
  --output-dir ./results/additional_analyses
python Classification/additional_analyses/verify_results.py ./results/additional_analyses
```

Restore the required saved prediction workbooks before running. Use a new or empty output directory. The default calculation uses 1,000 bootstrap replicates with seed 20261024; the verifier checks agreement with the accepted prediction snapshot. See the [validation notes](additional_analyses/VALIDATION.md) for reference checks and the [root README](../README.md) for manuscript table mappings.

### Second-round revision analyses

These scripts add the analyses requested in the second review round. Like the package above, they read saved predictions and never retrain a model. They also need each experiment's `val_result.xlsx` (written by `fold_test_val.py`), because recalibration is fitted on validation predictions only.

| Script | Output | Manuscript item |
|---|---|---|
| [federated_auc_ci.py](additional_analyses/federated_auc_ci.py) | `federated_auc_ci.csv` | Section 2.4: pooled and center-averaged external AUC, federated versus centralized, paired bootstrap 95% CIs |
| [center_adjusted_or.py](additional_analyses/center_adjusted_or.py) | `center_adjusted_or.csv` | Section 2.2.1 and Limitations: DenseNet-121 score odds ratio per SD after adjustment for center |
| [calibration_dca.py](additional_analyses/calibration_dca.py) | `calibration_dca.csv`, `calibration_rows.tex`, `net_benefit_rows.tex` | Supplementary Tables A19 and A20 |
| [plot_calibration_dca.py](additional_analyses/plot_calibration_dca.py) | `figA2_calibration_dca.pdf/.png` | Supplementary Fig. A2 |
| [plot_roc_pr.py](additional_analyses/plot_roc_pr.py) | `figure4/*.pdf` | Figure 4, panels (a)–(h) |

Run all of them and check the results from the repository root:

```bash
python -m pip install -r requirements.txt
bash Classification/additional_analyses/run_round2.sh ./Classification ./results/round2
```

`run_round2.sh` ends with [verify_round2.py](additional_analyses/verify_round2.py), which compares the outputs with [reference_results/round2](additional_analyses/reference_results/round2) and checks the odds ratios printed in the manuscript. Bootstrap CIs use 1,000 replicates stratified as described in each script, with seed 20261031. Example outputs, including the Figure 4 panels and Fig. A2, are in [example_output/round2](additional_analyses/example_output/round2); the verification record is in [VALIDATION.md](additional_analyses/VALIDATION.md).

Method notes:
- **Recalibration.** Each fold-specific (internal) or held-out-center-specific (external) model is recalibrated by logistic regression on the logit of its own validation predictions, and the fitted transformation is applied to that model's test predictions. Test labels and held-out-center data are not used.
- **Center adjustment.** Scores are rank-normalized within each fold (internal) or across all held-out predictions (external), standardized, and entered with center as a fixed effect; Wald 95% CIs.
- **Federated CIs.** Patients are resampled within each center and true class, with the same resampled patients for every model, so the differences are paired.
