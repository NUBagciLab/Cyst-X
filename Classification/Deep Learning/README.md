# Cyst-X: Deep Learning Risk Stratification Hub

This directory hosts the core deep learning engines for multi-center IPMN malignancy-risk stratification. It separates experimental execution into internal validation (cross-validation) and external validation (leave-one-center-out testing), supporting single-modality pipelines and comprehensive multi-modality fusion frameworks. Model weights are available at [HuggingFace](https://huggingface.co/phy710/Cyst-X/tree/main/Classification).

---

## 📂 Directory Structure

```filesystem
Deep Learning/
├── internal/                     # Models evaluated via internal cross-validation partitions
│   ├── 2-class/                  # Focus task: Binary high-risk vs. no/low-risk mapping
│   └── 3-class/centralized/      # Baseline pooled reference for 3-class categorization
├── external/                     # Models evaluated via Leave-One-Center-Out validation
│   ├── 2-class/                  # Generalizability assessment for the binary task
│   └── 3-class/centralized/      # External baseline pooled reference 
└── README.md                     # This structural overview document
```

## ⚙️ Experimental Configurations (2-Class Folders)

Within the `2-class/` execution environments, scripts are organized around distinct multi-institutional data training frameworks and feature-fusion protocols.

### 1. Single-Modality and Federated Frameworks
The core evaluation structures support training across four state-of-the-art 3D convolutional neural networks: **DenseNet-121**, **ResNet-34**, **ResNet-50**, and **EfficientNet-B0**. They are executed under three primary optimization methodologies:
* **`centralized/`**: Models optimized over a pooled dataset approach where data is centralized across a single silo.
* **`fedavg/`**: Distributed training using standard Federated Averaging across decentralized institutional partitions.
* **`fedprox/`**: Distributed training utilizing Federated Proximal optimization to combat inter-site data heterogeneity across different choices of the proximal coefficient $\mu$.

### 2. Multi-Modality Modality-Fusion Frameworks
* **`early_fsuion/`**
* **`late_fsuion/`**
* **`siamese_fsuion/`**
* **`logit_fsuion/`**

### 3. Histology-confirmed Cases Evaluation:
* **`_histology/`**

### 🚀 Running the Pipeline

The `main.sh` shell script orchestrates the full training and testing sequence. 

#### Command Usage

You can execute the shell script from your terminal using the following interface for training and testing. 

For FedProx:
```bash
bash ./main.sh [-model model_name] [-data_path data_path] [-mu mu_value]
```
For others:
```bash
bash ./main.sh [-model model_name] [-data_path data_path]
```

For test only:

    python fold_test.py --model "$model" --t "$t" --data-path "$data_path"
    
    python fold_test.py --model "$model" --t "$t" --data-path "$data_path" --mu "$mu"

For ploting t-SNE (only availabe at `Classification/Deep Learning/internal/2-class/centralized`):

    python tsne.py --model "$model" --t "$t" --data-path "$data_path"
