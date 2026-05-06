#!/usr/bin/env python3
"""
Radiomics inference script for fold-specific prediction.

This script reconstructs each fold's final scaler and model using:
- final_selected_features.json
- final_training_config.json
- split JSON train data

Then it performs inference on the specified fold test set.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_DIRS = {
    "final": "/data2/pyq6817/CystX/radiomics_results/t1_mRMR",
    "external": "/data2/pyq6817/CystX/radiomics_results/t1_mRMR_external",
    "internal": "/data2/pyq6817/CystX/radiomics_results/t1_mRMR_internal",
}


def infer_default_model_dir(split_path: Path) -> Path:
    split_name = split_path.name.lower()
    if "external" in split_name:
        return Path(DEFAULT_MODEL_DIRS["external"])
    if "internal" in split_name:
        return Path(DEFAULT_MODEL_DIRS["internal"])
    return Path(DEFAULT_MODEL_DIRS["final"])


def load_split_fold(split_path: Path, fold: int) -> Tuple[List[Dict], List[Dict]]:
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    for fold_item in split_data["folds"]:
        if int(fold_item["fold"]) == fold:
            return fold_item["train"], fold_item["test"]

    raise ValueError(f"Fold {fold} not found in split file: {split_path}")


def build_feature_label_df(features_df: pd.DataFrame, split_list: List[Dict]) -> pd.DataFrame:
    split_df = pd.DataFrame(
        {
            "Image": [item["image"] for item in split_list],
            "Label": [int(item["label"]) for item in split_list],
        }
    )
    merged = split_df.merge(features_df, on="Image", how="inner")
    if merged.empty:
        raise ValueError("No matching rows found between split JSON and feature CSV by Image.")
    return merged


def get_case_id(image_name: str) -> str:
    if image_name.endswith(".nii.gz"):
        return image_name[: -len(".nii.gz")]
    return Path(image_name).stem


def train_fold_model(
    train_df: pd.DataFrame,
    feature_names: List[str],
    selected_feature_names: List[str],
    train_cfg: Dict,
) -> Tuple[StandardScaler, RandomForestClassifier]:
    X_train = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df["Label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selected_indices = [feature_names.index(name) for name in selected_feature_names]
    X_train_selected = X_train_scaled[:, selected_indices]

    model = RandomForestClassifier(
        n_estimators=int(train_cfg["n_estimators"]),
        max_depth=train_cfg["max_depth"],
        min_samples_leaf=int(train_cfg["min_samples_leaf"]),
        max_features=train_cfg["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_selected, y_train)
    return scaler, model


def run_inference(
    test_df: pd.DataFrame,
    feature_names: List[str],
    selected_feature_names: List[str],
    scaler: StandardScaler,
    model: RandomForestClassifier,
    fold: int,
) -> pd.DataFrame:
    X_test = test_df[feature_names].to_numpy(dtype=float)
    X_test_scaled = scaler.transform(X_test)

    selected_indices = [feature_names.index(name) for name in selected_feature_names]
    X_test_selected = X_test_scaled[:, selected_indices]

    y_pred = model.predict(X_test_selected).astype(int)
    proba = model.predict_proba(X_test_selected)
    if 1 in model.classes_:
        class1_idx = int(np.where(model.classes_ == 1)[0][0])
    else:
        class1_idx = int(np.argmax(model.classes_))
    y_prob_class1 = proba[:, class1_idx]

    output = pd.DataFrame(
        {
            "ID": [get_case_id(x) for x in test_df["Image"].tolist()],
            "Label": test_df["Label"].astype(int).tolist(),
            "Prediction": y_pred.tolist(),
            "Probability": y_prob_class1.tolist(),
            "Fold": [fold] * len(test_df),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Radiomics fold inference")
    parser.add_argument("-f", "--fold", type=int, required=True, help="Fold number")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Feature CSV path (e.g., radiomics_features_t1.csv or radiomics_features_t2.csv)",
    )
    parser.add_argument("--split", type=str, required=True, help="Split JSON path")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model result directory; auto-inferred from split when omitted",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path; default: <model_dir>/fold_<fold>/inference_predictions.csv",
    )
    args = parser.parse_args()

    split_path = Path(args.split).expanduser().resolve()
    features_path = Path(args.features).expanduser().resolve()
    model_dir = (
        Path(args.model_dir).expanduser().resolve()
        if args.model_dir
        else infer_default_model_dir(split_path)
    )
    fold_dir = model_dir / f"fold_{args.fold}"

    selected_features_path = fold_dir / "final_selected_features.json"
    training_config_path = fold_dir / "final_training_config.json"
    results_path = fold_dir / "results.json"

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (fold_dir / "inference_predictions.csv")
    )

    if not features_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")
    if not selected_features_path.exists():
        raise FileNotFoundError(f"Missing selected feature file: {selected_features_path}")
    if not training_config_path.exists():
        raise FileNotFoundError(f"Missing training config file: {training_config_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    print(f"Fold: {args.fold}")
    print(f"Features: {features_path}")
    print(f"Split: {split_path}")
    print(f"Model dir: {model_dir}")

    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_feature_names = json.load(f)
    with open(training_config_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        result_meta = json.load(f)

    feature_names = result_meta.get("feature_names", [])
    if not feature_names:
        raise ValueError(f"feature_names not found in results: {results_path}")

    features_df = pd.read_csv(features_path)
    train_list, test_list = load_split_fold(split_path, args.fold)

    train_df = build_feature_label_df(features_df, train_list)
    test_df = build_feature_label_df(features_df, test_list)

    missing_full = [f for f in feature_names if f not in features_df.columns]
    if missing_full:
        raise ValueError(f"Feature CSV missing {len(missing_full)} full features required by model.")
    missing_selected = [f for f in selected_feature_names if f not in feature_names]
    if missing_selected:
        raise ValueError(f"Selected feature list has {len(missing_selected)} unknown feature names.")

    scaler, model = train_fold_model(train_df, feature_names, selected_feature_names, train_cfg)
    pred_df = run_inference(test_df, feature_names, selected_feature_names, scaler, model, args.fold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    print(f"Rows: {len(pred_df)}")


if __name__ == "__main__":
    main()
