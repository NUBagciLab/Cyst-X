# 🚀 Running the Pipeline
## Step 1: Preparation
Copy the result files from the corresponding 2-class classification folders (`Deep Learning` or `Radiomics`) to `Internal 2 Classes` and `External 2 Classes`. If you download this repo, this part is already done.
## Step 2: Calibration
For calibration on all internal results:

    chmod +x analysis_internal.sh
    ./analysis_internal.sh

The results will be saved in `Internal 2 Classes Calibrated`. 

Alternatively, you can run calibration on a specific file:

    python analysis_internal.py -i ./Internal 2 Classes/3D Radiomics/t1.xlsx -o ./Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx

For calibration on all external results:

    chmod +x analysis_external.sh
    ./analysis_external.sh
    
The results will be saved in `External 2 Classes Calibrated`. 

Alternatively, you can run calibration on a specific file:

    python analysis_internal.py -i ./External 2 Classes/3D Radiomics/t1.xlsx -o ./External 2 Classes Calibrated/3D Radiomics/t1.xlsx

The thresholds are saved in the output Excel files. 

# 📂 Directory Structure
Within the `Internal 2 Classes` and `External 2 Classes`:
* `3D Radiomics`: Radiomics models from `Classification/Radiomics`
* `DenseNet-121` / `ResNet-34` / `ResNet-50` / `EfficientNet-B0`: Centralized models from `Classification/Deep Learning/internal/2-class/centralized` or `Classification/Deep Learning/external/2-class/centralized`. Models optimized over a pooled dataset approach where data is centralized across a single silo.
* `+FedAvg`: FedAvg models from `Classification/Deep Learning/internal/2-class/FedAvg` or `Classification/Deep Learning/external/2-class/FedAvg`. Distributed training using standard Federated Averaging across decentralized institutional partitions.
* `+FedProx(0.1)` / `+FedProx(0.3)`: FedProx models from `Classification/Deep Learning/internal/2-class/FedProx` or `Classification/Deep Learning/external/2-class/FedProx`. Distributed training utilizing Federated Proximal optimization to combat inter-site data heterogeneity across different choices of the proximal coefficient $\mu$.
* Folders with `fusion` in the name: Multi-modality fusion models from the same folder name under `Classification/Deep Learning/internal/2-class/`.
  * Early vs. Late Fusion: Folders appended with the `_shared` suffix denote **Early Fusion** (where encoders share identical weights across both modalities to map into a shared latent space). Folders without `_shared` represent **Late Fusion** (where independent, modality-specific encoder weights are utilized to learn separate features).
  * Head Architecture Depth (1 vs. 2 Linear Layers): Folders with a **`2`** in their name utilize a deeper classification head containing **two linear layers**. Folders without a `2` utilize a simpler head with a **single linear layer**.
  * `fusion` / `fusion_shared`: Latent Feature Concatenation with a **1-layer** classification head.
  * `fusion2` / `fusion2_shared`: Latent Feature Concatenation with a **2-layer** classification head.
  * `fusion_add` / `fusion_add_shared`: Latent Feature Addition with a **1-layer** classification head.
  * `fusion_add2` / `fusion_add2_shared`: Latent Feature Addition with a **2-layer** classification head.
  * `fusion_prob`: Weighted probability-level fusion where networks dynamically capture and learn the optimal contribution scalar of each isolated modality output.
> 💡 **Key Performance Finding:** During our benchmarking, the deeper **2-layer classification heads** yielded superior performance on internal cross-validation (internal evaluation), whereas the simpler **1-layer classification heads** demonstrated greater generalizability and better results during leave-one-center-out testing (external evaluation).
Therefore, we present internal `fusion2` / `fusion2_shared` / `fusion_add2` / `fusion_add2_shared` and external `fusion` / `fusion_shared` / `fusion_add` / `fusion_add_shared` in our paper.
