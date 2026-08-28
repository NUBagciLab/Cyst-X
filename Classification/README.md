# Cyst-X: Malignancy-Risk Stratification Classification Pipelines

This directory contains the core implementation files for the automated risk stratification of pancreatic cystic neoplasms (IPMNs). The workspace is divided into end-to-end deep learning networks, hand-crafted radiomics workflows, and a calibration post-processing framework.

---

## 📂 Directory Structure

* **`Deep Learning/`** – Contains 3D network architectures (including DenseNet-121, ResNet-34, ResNet-50, and EfficientNet-B0), multi-modality fusion mechanisms, and distributed federated orchestration engines (`FedAvg` and `FedProx`).
* **`Radiomics/`** – Houses the classical pipeline components, including isotropic resampling, N4 bias-field correction, maximum-relevance minimum-redundancy (mRMR) feature selection, and the Random Forest classification baseline.
* **`results_calibration/`** – Performs center-specific clinical calibration on classification thresholds to improve the classification results.
* **`results/`** – ``Its function is same as `results_calibration` but outputs latex tables in our paper.
* **`vs_radiologists/`** - Compare the models with radiologists on 629 patients with both T1W and T2W scans.

---

## ⚙️ Experimental and Calibration Workflow

The classification pipeline operates in a sequential, two-stage framework to handle the high data heterogeneity inherent to multi-vendor clinical data:

### Stage 1: Standard Prediction (Uncalibrated)
The source scripts executed inside the `Deep Learning/` and `Radiomics/` directories perform inference and training under standard, uncalibrated constraints:
* **Binary Classification:** Employs a default unconstrained decision threshold of $50\%$ ($0.5$).
* **Multi-Class Classification:** Predicts outcomes strictly using argmax selection over the raw output logits.

### Stage 2: Clinically Meaningful Threshold Calibration for Binary Classification
Because institutional scanning protocols and cohort distributions vary widely across international centers, a uniform 0.5 cut-off results in variable diagnostic profiles. 

To resolve this, the uncalibrated prediction arrays compiled across the validation folds are output to the `results/` directory. We then execute a localized optimization routine that searches a grid space to dynamically determine center-specific operational thresholds. This calibration explicitly optimizes overall accuracy under realistic medical boundary constraints (requiring sensitivity $> 35\%$ and specificity $> 85\%$), mapping the models to clinical environments safely.
