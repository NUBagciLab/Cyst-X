# Cyst-X: Deep Learning Risk Stratification Hub

This directory hosts the core deep learning engines for multi-center IPMN malignancy-risk stratification. It separates experimental execution into internal validation (cross-validation) and external validation (leave-one-center-out testing), supporting single-modality pipelines and comprehensive multi-modality fusion frameworks.

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
To leverage complementary information from paired sequences, our architectures are split by encoder weight sharing, classification head depth, and fusion location:

* **Early vs. Late Fusion:** Folders appended with the `_shared` suffix denote **Early Fusion** (where encoders share identical weights across both modalities to map into a shared latent space). Folders without `_shared` represent **Late Fusion** (where independent, modality-specific encoder weights are utilized to learn separate features).
* **Head Architecture Depth (1 vs. 2 Linear Layers):** Folders with a **`2`** in their name utilize a deeper classification head containing **two linear layers**. Folders without a `2` utilize a simpler head with a **single linear layer**.

#### Fusion Directory Mapping:
* **`fusion` / `fusion_shared`**: Latent Feature Concatenation with a **1-layer** classification head.
* **`fusion2` / `fusion2_shared`**: Latent Feature Concatenation with a **2-layer** classification head.
* **`fusion_add` / `fusion_add_shared`**: Latent Feature Addition with a **1-layer** classification head.
* **`fusion_add2` / `fusion_add2_shared`**: Latent Feature Addition with a **2-layer** classification head.
* **`fusion_prob`**: Weighted probability-level fusion where networks dynamically capture and learn the optimal contribution scalar of each isolated modality output.

> 💡 **Key Performance Finding:** During our benchmarking, the deeper **2-layer classification heads** yielded superior performance on internal cross-validation (internal evaluation), whereas the simpler **1-layer classification heads** demonstrated greater generalizability and better results during leave-one-center-out testing (external evaluation).

### 🚀 Running the Pipeline

The `main.sh` shell script orchestrates the full training and testing sequence. 

#### Command Usage

You can execute the shell script from your terminal using the following interface for training and testing. 
```bash
chmod +x ./main.sh
```
For FedProx:
```bash
./main.sh [-model model_name] [-data_path data_path] [-mu mu_value]
```
For others:
```bash
./main.sh [-model model_name] [-data_path data_path]
```

For test only:

    python fold_test.py --model "$model" --t "$t" --data-path "$data_path"
    
    python fold_test.py --model "$model" --t "$t" --data-path "$data_path" --mu "$mu"
