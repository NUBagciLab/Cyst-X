# Cyst-X: A Multi-Center MRI Benchmark and Federated Learning Framework for Malignancy-Risk Stratification of Pancreatic Cystic Neoplasm

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2507.22017-B31B1B.svg)](https://arxiv.org/abs/2507.22017)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/phy710/Cyst-X/) 
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/phy710/Cyst-X/) 
[![OSF](https://img.shields.io/badge/OSF-Dataset%20Mirror-007ec6.svg)](https://osf.io/74vfs/)

Official implementation of **Cyst-X**, an end-to-end multi-center pipeline integrating state-of-the-art pancreas segmentation, federated optimization, and classical radiomics pipelines for automated malignancy risk stratification of intraductal papillary mucinous neoplasms (IPMNs).

---

# 📌 Overview

Pancreatic cancer is projected to be the second-deadliest cancer by 2030, making early detection critical. IPMNs are key cancer precursors, but current consensus guidelines struggle to stratify malignancy risk accurately. 

**Cyst-X** addresses this structural bottleneck by providing:
1. **The Largest Multi-Center Pancreas MRI Resource:** 1,461 abdominal MRI scans from 764 patients across seven international medical centers with expert annotations and pathology-anchored ground truth.
2. **Advanced Pancreas Segmentation:** Integrating `PanSegNet` (a linear self-attention transformer-based backbone) for high-fidelity region of interest (ROI) extraction.
3. **Privacy-Preserving Federated Learning:** Distributed risk classification training utilizing `FedAvg` and `FedProx` without exchanging raw clinical images.

<p align="center">
  <img src="figures/pipeline.png" alt="" width="100%" />
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
The Cyst-X dataset includes raw/preprocessed NIfTI volumes, expert pancreas segmentation masks, center-wise stratified splits, and clinical metadata. Please download the dataset via Hugging Face:
```Bash
# Make sure you have git-lfs installed
git lfs install

# Clone the dataset repository
git clone [https://huggingface.co/datasets/phy710/Cyst-X](https://huggingface.co/datasets/phy710/Cyst-X) data/Cyst-X
```
    
Alternatively, using Python and the huggingface_hub library:
```Python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="phy710/Cyst-X",
    repo_type="dataset",
    local_dir="data/Cyst-X"
)
```

You can access and download the dataset mirror directly from the Open Science Framework (OSF).

## Expected Directory Layout
Once downloaded, ensure your dataset/ directory matches the following structure:

    Cyst-X/dataset/
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
    ├── IPMN_Classification/
    │   ├── t1/                                     # Cropped 3D pancreatic ROIs for single-input T1W (.nii.gz)
    │   ├── t2/                                     # Cropped 3D pancreatic ROIs for single-input T2W (.nii.gz)
    │   ├── IPMN_labels_t1_total.xlsx               # Comprehensive labels and metadata for all T1W scans
    │   ├── IPMN_labels_t2_total.xlsx               # Comprehensive labels and metadata for all T2W scans
    │   └── IPMN_labels_total.xlsx                  # Master cohort metadata and paired (T1W+T2W) ground truth labels
    └── IPMN_images_masks/
        ├── t1/
        │   ├── images/                             # Full-volume T1W NIfTI scans (.nii.gz)
        │   └── masks/                              # Full-volume ground-truth pancreas masks (.nii.gz)
        └── t2/
            ├── images/                             # Full-volume T2W NIfTI scans (.nii.gz)
            └── masks/                              # Full-volume ground-truth pancreas masks (.nii.gz)

# 📝 Citation
If you use this dataset in your research, please cite our paper:

    @article{pan2025cyst,
      title={Cyst-X: A Multi-Center MRI Benchmark and Federated Learning Framework for Malignancy-Risk Stratification of Pancreatic Cystic Neoplasm},
      author={Pan, Hongyi and Durak, Gorkem and Keles, Elif and Hong, Ziliang and Seyithanoglu, Deniz and Zhang, Zheyuan and Medetalibeyoglu, Alpay and Aktas, Halil Ertugrul and Bejar, Andrea Mia and Taktak, Yavuz and others},
      journal={arXiv preprint arXiv:2507.22017},
      year={2025}
    }
