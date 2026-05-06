#!/usr/bin/env python3
"""
Radiomics-based IPMN Classification Pipeline (T2)

This script implements the workflow described in task2.txt:
1. Hyperparameter optimization with CV (no bootstrapping)
2. Training-set bootstrapping for mRMR feature stability analysis
3. Final single-model training on full training set
4. Held-out test evaluation + test bootstrapping for uncertainty
"""

import argparse
import itertools
import json
import os
import pickle
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_data_split(json_path: str, fold: int) -> Tuple[List[Dict], List[Dict]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    fold_data = None
    for fold_item in data["folds"]:
        if fold_item["fold"] == fold:
            fold_data = fold_item
            break

    if fold_data is None:
        raise ValueError(f"Fold {fold} not found in split file: {json_path}")

    return fold_data["train"], fold_data["test"]


def load_features(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def match_features_to_data(
    features_df: pd.DataFrame, data_list: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    image_to_label = {item["image"]: item["label"] for item in data_list}
    matched_indices = []
    labels = []

    for idx, row in features_df.iterrows():
        image_name = row["Image"]
        if image_name in image_to_label:
            matched_indices.append(idx)
            labels.append(image_to_label[image_name])

    if len(matched_indices) == 0:
        raise ValueError("No matching images found between features CSV and split JSON.")

    feature_cols = [col for col in features_df.columns if col != "Image"]
    X = features_df.iloc[matched_indices][feature_cols].values
    y = np.array(labels, dtype=int)

    print(f"Matched {len(matched_indices)} / {len(data_list)} samples")
    return X, y, feature_cols


def _discretize_for_mrmr(X: np.ndarray, bins: int = 5) -> np.ndarray:
    X_disc = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        col = X[:, j]
        edges = np.quantile(col, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            X_disc[:, j] = 0
            continue
        X_disc[:, j] = np.digitize(col, edges[1:-1], right=False)
    return X_disc


def select_mrmr_features(X_scaled: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy mRMR-like selection:
    max relevance (mutual info to label) - redundancy (mean abs correlation to selected).
    """
    from sklearn.feature_selection import mutual_info_classif

    n_features = X_scaled.shape[1]
    k = min(k, n_features)
    if k <= 0:
        raise ValueError("k must be >= 1")

    X_disc = _discretize_for_mrmr(X_scaled, bins=5)
    relevance = mutual_info_classif(X_disc, y, discrete_features=True, random_state=42)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)

    corr = np.corrcoef(X_scaled, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)

    selected = []
    candidates = set(range(n_features))

    first_idx = int(np.argmax(relevance))
    selected.append(first_idx)
    candidates.remove(first_idx)

    while len(selected) < k and len(candidates) > 0:
        best_idx = None
        best_score = -np.inf
        for idx in candidates:
            redundancy = float(np.mean(corr[idx, selected])) if len(selected) > 0 else 0.0
            score = relevance[idx] - redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidates.remove(best_idx)

    return np.array(selected, dtype=int)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray) -> Dict[str, float]:
    # ROC-AUC is undefined when only one class is present.
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_prob_pos)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob_pos)

    return {
        "roc_auc": float(roc_auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "brier_score": float(brier),
        "f1_score": float(f1),
    }


def hyperparameter_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], float]:
    print("\n" + "=" * 60)
    print("Step 1: Hyperparameter Optimization (CV, no bootstrap)")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "k_features": [10, 20, 30, 50],
    }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combinations = list(itertools.product(*values))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    best_score = -np.inf
    best_params = None

    print(f"Total hyperparameter combinations: {len(all_combinations)}")
    for combo_idx, combo in enumerate(all_combinations, 1):
        params = dict(zip(keys, combo))
        fold_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train = X_train[train_idx]
            y_cv_train = y_train[train_idx]
            X_cv_val = X_train[val_idx]
            y_cv_val = y_train[val_idx]

            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)

            selected_idx = select_mrmr_features(X_cv_train_scaled, y_cv_train, params["k_features"])
            X_cv_train_sel = X_cv_train_scaled[:, selected_idx]
            X_cv_val_sel = X_cv_val_scaled[:, selected_idx]

            rf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X_cv_train_sel, y_cv_train)
            y_val_prob_pos = rf.predict_proba(X_cv_val_sel)[:, 1]
            fold_scores.append(roc_auc_score(y_cv_val, y_val_prob_pos))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        if combo_idx % 20 == 0 or combo_idx == len(all_combinations):
            print(f"Checked {combo_idx}/{len(all_combinations)} combinations")

    print(f"Best CV ROC-AUC: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    return best_params, best_score


def training_bootstrap_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    best_params: Dict[str, Any],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Step 2: Training-Set Bootstrapping for Stability Analysis")
    print("=" * 60)

    rng = np.random.RandomState(random_state)
    n_samples = len(y_train)
    n_features = X_train.shape[1]
    k = int(best_params["k_features"])

    select_count = np.zeros(n_features, dtype=int)
    rank_sum = np.zeros(n_features, dtype=float)
    rank_count = np.zeros(n_features, dtype=int)
    perm_importance_sum = np.zeros(n_features, dtype=float)
    selected_feature_lists = []

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            print(f"Bootstrap iteration {b + 1}/{n_bootstrap}")

        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]

        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)

        selected_idx = select_mrmr_features(X_boot_scaled, y_boot, k)
        selected_feature_lists.append([feature_names[i] for i in selected_idx])

        select_count[selected_idx] += 1

        selected_relevance = []
        for i in selected_idx:
            # Rank proxy: mutual information relevance in current bootstrap sample.
            from sklearn.feature_selection import mutual_info_classif

            rel = mutual_info_classif(
                _discretize_for_mrmr(X_boot_scaled[:, [i]], bins=5),
                y_boot,
                discrete_features=True,
                random_state=42,
            )[0]
            selected_relevance.append(rel)
        order = np.argsort(-np.array(selected_relevance))
        for rank_pos, order_idx in enumerate(order, 1):
            fidx = selected_idx[order_idx]
            rank_sum[fidx] += rank_pos
            rank_count[fidx] += 1

        rf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            random_state=random_state + b,
            n_jobs=-1,
        )
        X_boot_sel = X_boot_scaled[:, selected_idx]
        rf.fit(X_boot_sel, y_boot)

        perm = permutation_importance(
            rf,
            X_boot_sel,
            y_boot,
            n_repeats=5,
            random_state=random_state + b,
            n_jobs=-1,
            scoring="roc_auc",
        ).importances_mean
        perm_importance_sum[selected_idx] += perm

    selection_frequency = select_count / float(n_bootstrap)
    mean_rank = np.full(n_features, np.nan)
    mean_rank[rank_count > 0] = rank_sum[rank_count > 0] / rank_count[rank_count > 0]
    mean_perm_importance = perm_importance_sum / max(n_bootstrap, 1)

    top_idx = np.argsort(-selection_frequency)[: min(20, n_features)]
    print("\nTop features by selection frequency:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {feature_names[idx]}: {selection_frequency[idx]:.4f}")

    return {
        "n_bootstrap": n_bootstrap,
        "selected_feature_lists": selected_feature_lists,
        "selection_frequency": selection_frequency,
        "mean_rank_when_selected": mean_rank,
        "mean_permutation_importance": mean_perm_importance,
        "top_feature_names_by_frequency": [feature_names[i] for i in top_idx],
    }


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    best_params: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Step 3: Final Model Training (Single Model)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selected_idx = select_mrmr_features(X_train_scaled, y_train, int(best_params["k_features"]))
    selected_feature_names = [feature_names[i] for i in selected_idx]
    X_train_sel = X_train_scaled[:, selected_idx]

    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_sel, y_train)

    print(f"Selected features (K={len(selected_idx)}):")
    for name in selected_feature_names[:20]:
        print(f"  - {name}")
    if len(selected_feature_names) > 20:
        print(f"  ... ({len(selected_feature_names) - 20} more)")

    return {
        "model": model,
        "scaler": scaler,
        "selected_idx": selected_idx,
        "selected_feature_names": selected_feature_names,
    }


def test_evaluation_and_bootstrap(
    X_test: np.ndarray,
    y_test: np.ndarray,
    final_bundle: Dict[str, Any],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Step 4: Test Set Evaluation and Test Bootstrapping")
    print("=" * 60)

    scaler = final_bundle["scaler"]
    model = final_bundle["model"]
    selected_idx = final_bundle["selected_idx"]

    X_test_scaled = scaler.transform(X_test)
    X_test_sel = X_test_scaled[:, selected_idx]

    y_prob_pos = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_prob_pos >= 0.5).astype(int)
    single_metrics = compute_binary_metrics(y_test, y_pred, y_prob_pos)

    print("Single held-out test evaluation:")
    for k, v in single_metrics.items():
        print(f"  {k}: {v:.4f}")

    rng = np.random.RandomState(random_state)
    n_test = len(y_test)
    metric_names = list(single_metrics.keys())
    boot_collect = {m: [] for m in metric_names}

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"Test bootstrap iteration {b + 1}/{n_bootstrap}")

        idx = rng.choice(n_test, size=n_test, replace=True)
        X_boot = X_test_sel[idx]
        y_boot = y_test[idx]

        y_prob_boot = model.predict_proba(X_boot)[:, 1]
        y_pred_boot = (y_prob_boot >= 0.5).astype(int)
        m = compute_binary_metrics(y_boot, y_pred_boot, y_prob_boot)
        for metric in metric_names:
            boot_collect[metric].append(m[metric])

    ci = {}
    for metric in metric_names:
        values = np.array(boot_collect[metric], dtype=float)
        valid_values = values[~np.isnan(values)]
        if valid_values.size == 0:
            ci[metric] = {
                "mean": float("nan"),
                "ci_2.5": float("nan"),
                "ci_97.5": float("nan"),
            }
            continue
        ci[metric] = {
            "mean": float(np.mean(valid_values)),
            "ci_2.5": float(np.percentile(valid_values, 2.5)),
            "ci_97.5": float(np.percentile(valid_values, 97.5)),
        }

    print("\nBootstrapped test metrics (mean [95% CI]):")
    for metric, stats in ci.items():
        print(f"  {metric}: {stats['mean']:.4f} [{stats['ci_2.5']:.4f}, {stats['ci_97.5']:.4f}]")

    return {
        "single_evaluation": single_metrics,
        "bootstrap_statistics": ci,
        "n_bootstrap": n_bootstrap,
    }


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def save_results(results: Dict[str, Any], output_dir: str, fold: int):
    os.makedirs(output_dir, exist_ok=True)
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    saved_files = []

    results_for_json = {k: v for k, v in results.items() if k not in ["final_bundle"]}
    results_json_path = os.path.join(fold_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(results_for_json), f, indent=2)
    saved_files.append(results_json_path)

    bundle = results["final_bundle"]
    model_path = os.path.join(fold_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle["model"], f)
    saved_files.append(model_path)

    scaler_path = os.path.join(fold_dir, "final_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(bundle["scaler"], f)
    saved_files.append(scaler_path)

    selected_features_path = os.path.join(fold_dir, "final_selected_features.json")
    with open(selected_features_path, "w", encoding="utf-8") as f:
        json.dump(bundle["selected_feature_names"], f, indent=2)
    saved_files.append(selected_features_path)

    training_config_path = os.path.join(fold_dir, "final_training_config.json")
    with open(training_config_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(results["best_hyperparameters"]), f, indent=2)
    saved_files.append(training_config_path)

    stability = results["bootstrap_analysis"]
    feature_stability_df = pd.DataFrame(
        {
            "feature_name": results["feature_names"],
            "selection_frequency": stability["selection_frequency"],
            "mean_rank_when_selected": stability["mean_rank_when_selected"],
            "mean_permutation_importance": stability["mean_permutation_importance"],
        }
    ).sort_values("selection_frequency", ascending=False)
    feature_stability_path = os.path.join(fold_dir, "feature_stability.csv")
    feature_stability_df.to_csv(feature_stability_path, index=False)
    saved_files.append(feature_stability_path)

    missing_files = [p for p in saved_files if not os.path.exists(p)]
    if missing_files:
        raise RuntimeError(f"Some result files were not written successfully: {missing_files}")

    print(f"\nResults saved to {fold_dir}")
    print("Saved files:")
    for path in saved_files:
        print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(description="Radiomics-based IPMN Classification (Task2)")
    parser.add_argument("-f", "--fold", type=int, required=True, help="Fold number")
    parser.add_argument(
        "--features-path",
        type=str,
        default="/home/pyq6817/New_IPMN_Classification/Radiomics/features/radiomics_features_t2.csv",
        help="Path to radiomics feature CSV",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="/home/pyq6817/New_IPMN_Classification/Radiomics/splits_final_t2.json",
        help="Path to train/test split JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data2/pyq6817/CystX/radiomics_results/t2_mRMR",
        help="Directory where fold results will be saved",
    )
    args = parser.parse_args()

    features_path = args.features_path
    split_path = args.split_path
    output_dir = args.output_dir

    print("=" * 60)
    print(f"Radiomics IPMN Classification (Task2) - Fold {args.fold}")
    print("=" * 60)
    print(f"Features path: {features_path}")
    print(f"Split path: {split_path}")
    print(f"Output dir: {output_dir}")

    print("\nStep 0: Initial Data Split")
    print("-" * 60)
    train_data, test_data = load_data_split(split_path, args.fold)
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    print("\nLoading features...")
    features_df = load_features(features_path)
    print(f"Total radiomics features: {features_df.shape[1] - 1}")

    X_train, y_train, feature_names = match_features_to_data(features_df, train_data)
    X_test, y_test, _ = match_features_to_data(features_df, test_data)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    best_params, best_cv_score = hyperparameter_optimization(X_train, y_train)

    bootstrap_results = training_bootstrap_analysis(
        X_train,
        y_train,
        feature_names,
        best_params,
        n_bootstrap=1000,
    )

    final_bundle = train_final_model(X_train, y_train, feature_names, best_params)

    test_results = test_evaluation_and_bootstrap(
        X_test,
        y_test,
        final_bundle,
        n_bootstrap=1000,
    )

    all_results = {
        "fold": args.fold,
        "best_hyperparameters": best_params,
        "best_cv_score": float(best_cv_score),
        "bootstrap_analysis": bootstrap_results,
        "test_results": test_results,
        "final_bundle": final_bundle,
        "feature_names": feature_names,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    save_results(all_results, output_dir, args.fold)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

