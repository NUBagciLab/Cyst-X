# 🚀 Running the Pipeline
## Step 1: Preparation
Copy the result files from the corresponding 2-class classification folders (`Deep Learning` or `Radiomics`) to `Internal 2 Classes` and `External 2 Classes`. If you download this repo, this part is already done.
## Step 2: Calibration
For calibration on all internal results:

    bash ./analysis_internal.sh

The results will be saved in `Internal 2 Classes Calibrated`. ROCs and PR Curves are also saved with the output Excel files.

Alternatively, you can run calibration on a specific file:

    python analysis_internal.py -i './Internal 2 Classes/3D Radiomics/t1.xlsx' -o './Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx'

For calibration on all external results:

    bash ./analysis_external.sh
    
The results will be saved in `External 2 Classes Calibrated`. 

For calibration on all fusion strategy results:

    bash ./analysis_fusion.sh

Alternatively, you can run calibration on a specific file:

    python analysis_internal.py -i './Internal 2 Classes/3D Radiomics/t1.xlsx' -o './Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx'

    python analysis_external.py -i './External 2 Classes/3D Radiomics/t1.xlsx' -o './External 2 Classes Calibrated/3D Radiomics/t1.xlsx'

    python analysis_fusion_internal.py -i './Internal 2 Classes/early_fusion/result.xlsx' -o ''./Internal 2 Classes Calibrated/early_fusion/result.xlsx'

    python analysis_fusion_external.py -i './External 2 Classes/early_fusion/result.xlsx' -o ''./External 2 Classes Calibrated/early_fusion/result.xlsx'

The thresholds are saved in the output Excel files. 

To get results on the histology-confirmed cases, please run `analysis_internal_histology.sh` and `analysis_external_histology`.sh. Alternatively, you can run `analysis_internal_all.sh` and `analysis_external_all`.sh directly. These will do calibration on all Excel files under the folders `Internal 2 Classes` and `External 2 Classes`

# 📂 Directory Structure
Within the `Internal 2 Classes` and `External 2 Classes`:
* Folders with `_histology` mean the model was trained and tested on the histology-confirmed cases instead of all cases.
* `3D Radiomics`: Radiomics models from `Classification/Radiomics`
* `DenseNet-121` / `ResNet-34` / `ResNet-50` / `EfficientNet-B0`: Centralized models from `Classification/Deep Learning/internal/2-class/centralized` or `Classification/Deep Learning/external/2-class/centralized`. Models optimized over a pooled dataset approach where data is centralized across a single silo.
* `+FedAvg`: FedAvg `DenseNet-121` models from `Classification/Deep Learning/internal/2-class/FedAvg` or `Classification/Deep Learning/external/2-class/FedAvg`. Distributed training using standard Federated Averaging across decentralized institutional partitions.
* `+FedProx(0.1)` / `+FedProx(0.3)`: FedProx `DenseNet-121` models from `Classification/Deep Learning/internal/2-class/FedProx` or `Classification/Deep Learning/external/2-class/FedProx`. Distributed training utilizing Federated Proximal optimization to combat inter-site data heterogeneity across different choices of the proximal coefficient $\mu$.
* Folders with `fusion` in the name: Multimodality fusion `DenseNet-121 `models from the same folder name under `Classification/Deep Learning/internal/2-class/`.
  * `early_fusion` / `late_fusion` / `siamese_fusion` / `logit_fusion`: Fusion strategy with both T1W and T2W inputs.
