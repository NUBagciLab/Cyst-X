# Cyst-X: A Multi-Center MRI Benchmark and Federated Learning Framework for Malignancy-Risk Stratification of Pancreatic Cystic Neoplasm

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2507.22017-B31B1B.svg)](https://arxiv.org/abs/2507.22017)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/phy710/Cyst-X/) 
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/phy710/Cyst-X/) 
[![OSF](https://img.shields.io/badge/OSF-Dataset%20Mirror-007ec6.svg)](https://osf.io/74vfs/)

Official implementation of **Cyst-X**, an end-to-end multi-center pipeline integrating state-of-the-art pancreas segmentation, federated optimization, and classical radiomics pipelines for automated malignancy risk stratification of intraductal papillary mucinous neoplasms (IPMNs).

## Differences from v1

The original release is preserved as [GitHub code v1](https://github.com/NUBagciLab/Cyst-X/tree/v1), [dataset v1](https://huggingface.co/datasets/phy710/Cyst-X/tree/v1), and [models v1](https://huggingface.co/phy710/Cyst-X/tree/v1). Use the [current code on main](https://github.com/NUBagciLab/Cyst-X/tree/main), [dataset v2](https://huggingface.co/datasets/phy710/Cyst-X/tree/v2), and [models v2](https://huggingface.co/phy710/Cyst-X/tree/v2) for the updated workflow.

- **Image/mask orientation:** Some v1 masks did not share the orientation of their paired MRI images. These mismatches are corrected in the released v2 segmentation dataset, so preprocessing begins at step 1 (RAS conversion).
- **Bias-field correction:** The current classification preprocessing applies N4 bias-field correction before cropping. The downloadable v2 classification ROIs were generated from N4-corrected images; the full-volume segmentation images are the inputs for the RAS/N4 pipeline described in the [preprocessing README](preprocessing/README.md).
- **Classification folds:** Internal two-class models used four-fold cross-validation in v1 and are standardized to five-fold cross-validation in v2. Use the current `classification/t1_split.csv` and `classification/t2_split.csv` distributed with the v2 dataset.

The code changes can be viewed in the [v1-to-main comparison](https://github.com/NUBagciLab/Cyst-X/compare/v1...main).

---

# 📌 Overview

Pancreatic cancer is projected to be the second-deadliest cancer by 2030, making early detection critical. IPMNs are key cancer precursors, but current consensus guidelines struggle to stratify malignancy risk accurately. 

**Cyst-X** addresses this structural bottleneck by providing:
1. **The Largest Multi-Center Pancreas MRI Resource:** 1,461 abdominal MRI scans from 764 patients across seven international medical centers with expert annotations and pathology-anchored ground truth.
2. **Advanced Pancreas Segmentation:** Integrating [PanSegNet](https://github.com/NUBagciLab/PaNSegNet) (a linear self-attention transformer-based backbone) for high-fidelity region of interest (ROI) extraction.
3. **Privacy-Preserving Federated Learning:** Distributed risk classification training utilizing `FedAvg` and `FedProx` without exchanging raw clinical images.

<p align="center">
  <img src="figures/pipeline.png" alt="" width="80%" />
</p>

# 📂 Repository Structure
```filesystem
Cyst-X/
├── Classification/          # Pipelines and frameworks for deep risk modeling
├── Segmentation/            # Pipeline and engines for pancreas ROI localization
├── MRQy/                    # MRQy analysis
└── README.md                # Main repository documentation hub
```

# 📦 Dataset Access & Download
The Cyst-X dataset includes raw/preprocessed NIfTI volumes, expert pancreas segmentation masks, center-wise stratified splits, and clinical metadata. Please download the dataset via [HuggingFace](https://huggingface.co/datasets/phy710/Cyst-X).:

First, please make sure that `huggingface_hub` is installed:
```Bash
pip install huggingface_hub
```
    
Then, using Python and the huggingface_hub library:
```Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="phy710/Cyst-X",
    repo_type="dataset",
    local_dir="/dataset"
)
```

You can also access and download the dataset mirror directly from the Open Science Framework ([OSF](https://osf.io/74vfs/)). However, uploading massive amounts of files to OSF is difficult. So we compressed the dataset into several zip files. Please organize the files according to the expected directory layout.

## Expected Directory Layout
Once downloaded, ensure your dataset directory matches the following structure:

    /dataset/
    ├── data splits/
    │   ├── classification/
    │   │   ├── 2-class/
    │   │   │   ├── t1.csv                          # 4-fold CV splits for T1W (High-Risk vs. No/Low-Risk; stratified center-wise)
    │   │   │   ├── t2.csv                          # 4-fold CV splits for T2W (High-Risk vs. No/Low-Risk; stratified center-wise)
    │   │   │   └── two_inputs.csv                  # 4-fold CV splits for paired (T1W+T2W) multi-modal fusion; stratified center-wise
    │   │   └── 3-class/
    │   │       ├── t1.csv                          # 5-fold CV splits for T1W (No-Risk vs. Low-Risk vs. High-Risk; pooled across all centers)
    │   │       └── t2.csv                          # 5-fold CV splits for T2W (No-Risk vs. Low-Risk vs. High-Risk; pooled across all centers)
    │   └── pansegnet segmentation/
    │       ├── Task210_PancreasT1MRI_splits.csv    # 5-fold cross-validation splits for T1W pancreas segmentation
    │       └── Task211_PancreasT2MRI_splits.csv    # 5-fold cross-validation splits for T2W pancreas segmentation
    ├── IPMN_Classification/                        # IPMN classification dataset
    │   ├── t1/                                     # Cropped 3D pancreatic ROIs for single-input T1W (.nii.gz)
    │   ├── t2/                                     # Cropped 3D pancreatic ROIs for single-input T2W (.nii.gz)
    │   ├── IPMN_labels_t1_total.xlsx               # Comprehensive labels and metadata for all T1W scans
    │   ├── IPMN_labels_t2_total.xlsx               # Comprehensive labels and metadata for all T2W scans
    │   └── IPMN_labels_total.xlsx                  # Master cohort metadata and paired (T1W+T2W) ground truth labels
    └── IPMN_images_masks/                          # Pancreas segmentation dataset
        ├── t1/
        │   ├── images/                             # Full-volume T1W NIfTI scans (.nii.gz)
        │   └── masks/                              # Full-volume ground-truth pancreas masks (.nii.gz)
        └── t2/
            ├── images/                             # Full-volume T2W NIfTI scans (.nii.gz)
            └── masks/                              # Full-volume ground-truth pancreas masks (.nii.gz)

# 📊 Mapping Between Reported Table/Figure and the Corresponding Script
Table 1 and Supplementary Tables A2--A3: `/Segmentation/PanSegNet/Task210_PancreasT1MRI/get_results.sh` and `/Segmentation/PanSegNet/Task211_PancreasT2MRI/get_results.sh`.

Table 2: 

(1) Three-class classification: `Classification/Deep Learning/internal/3-class/centralized/fold_test.py`;

(2) Two-class classification: `Classification/Deep Learning/internal/2-class/xxx/fold_test.py` and `Classification/Radiomics`, where `xxx` is `centralized`, `fedavg`, or `fedprox`. Alternatively, you can find the results in Supplementary Tables A9 and A10 at row `Global`.

Figure 3: `Classification/tsne/tsne.py` and `Classification/Deep Learning/internal/2-class/centralized/tsne.py`.

Table 3: `Classification/vs_radiologists/run.sh` for calibrated results comparison, `Classification/vs_radiologists/run_uncalibrated.sh` for uncalibrated results comparison, `Classification/vs_radiologists/run_histology.sh` for calibrated histology results comparison, `Classification/vs_radiologists/run_histology_uncalibrated.sh` for uncalibrated histology results comparison.

Figure 4: `Classification/results_calibration/analysis_internal.sh`. 

Figure 5: `MRQy/MRQy.py`.

Supplementary Tables A9, A10: `Classification/results_calibration/analysis_internal.sh`. 

Supplementary Tables A11, A12: `Classification/results_calibration/analysis_external.sh`. 

Supplementary Table A13: `Classification/results_calibration/analysis_fusion.sh`. 

To download the pre-trained models:
```Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="phy710/Cyst-X",
    repo_type="model",
    local_dir="./"
)
```
Then put the models into the folders of the corresponding tasks.

# 📝 Citation
If you use this dataset in your research, please cite our paper:

    @article{pan2025cyst,
      title={Cyst-X: A Multi-Center MRI Benchmark and Federated Learning Framework for Malignancy-Risk Stratification of Pancreatic Cystic Neoplasm},
      author={Pan, Hongyi and Durak, Gorkem and Keles, Elif and Hong, Ziliang and Seyithanoglu, Deniz and Zhang, Zheyuan and Medetalibeyoglu, Alpay and Aktas, Halil Ertugrul and Bejar, Andrea Mia and Taktak, Yavuz and others},
      journal={arXiv preprint arXiv:2507.22017},
      year={2025}
    }
