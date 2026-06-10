# Cyst-X: Prediction Metrics and Threshold Calibration Hub

This directory serves as the centralized engine for storing raw classification outputs and executing center-specific calibration routines. It compiles predictive probabilities across all cross-validation rounds and applies clinically viable optimization constraints to maximize multi-institutional performance.

---

## 📂 Directory Structure

* **`Internal 2 Classes/`** – Contains uncalibrated prediction outputs generated during internal cross-validation trials across individual centers.
* **`External 2 Classes/`** – Houses prediction logs and inference scores generated across the leave-one-center-out generalizability experiments.
* **`fold_test.py`** – **The Fusion Evaluation Engine.** Located here at the root of the `results/` folder, this script compiles, evaluates, and applies threshold calibration specifically for the multi-modality **fused** models (e.g., fusion, fusion_add, fusion_prob).
* **`seed.py`** – Sets global environment random states and deterministic behavior flags to guarantee structural reproducibility across execution blocks.

> ⚠️ **Important Script Distinction:** There are three distinct versions of `fold_test.py` across the classification repository. The versions located inside the `internal/` and `external/` code directories are dedicated exclusively to evaluating single-modality inputs (**T1W or T2W only**). The script located here at the root of the `results/` folder is designed specifically to aggregate and calibrate **fused multi-modality inputs**.

---

## ⚖️ Operational Calibration Protocol

Because this dataset is derived from diverse international institutions, the raw imaging outputs exhibit marked inter-center acquisition heterogeneity. To safeguard patient classification metrics and resolve site-level data sparsity, we do not rely on a standard 50% cutoff but on a clinically meaningful threshold (sensitivity > 35% and specificity > 85%):

#### Command Usage
```bash
python fold_test.py
```
