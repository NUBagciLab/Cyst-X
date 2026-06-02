#!/usr/bin/env python3
import argparse
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


MERGED_CENTER_MAP = {
    "cad": "CAD_MCF",
    "mcf": "CAD_MCF",
    "northwestern": "NU_NORTHWESTERN",
    "nu": "NU_NORTHWESTERN",
}


def extract_center(center_id: str) -> str:
    text = str(center_id).strip()
    match = re.match(r"([A-Za-z]+)", text)
    if not match:
        return "UNKNOWN"
    raw_center = match.group(1).lower()
    return MERGED_CENTER_MAP.get(raw_center, raw_center.upper())


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        y_true_sample = y_true[indices]
        if len(np.unique(y_true_sample)) < 2:
            continue
        auc_values.append(roc_auc_score(y_true_sample, y_score[indices]))

    if not auc_values:
        return np.nan, np.nan

    alpha = 1.0 - confidence
    lower = np.percentile(auc_values, 100 * (alpha / 2))
    upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_metrics(df: pd.DataFrame, bootstrap_iters: int, seed: int) -> Dict[str, float]:
    y_true = df["Label"].to_numpy()
    y_pred = df["Prediction"].to_numpy()
    y_score = df["Probability"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        auc = np.nan
        ci_lower, ci_upper = np.nan, np.nan
    else:
        auc = roc_auc_score(y_true, y_score)
        ci_lower, ci_upper = bootstrap_auc_ci(
            y_true=y_true,
            y_score=y_score,
            n_bootstrap=bootstrap_iters,
            seed=seed,
        )

    return {
        "n_cases": len(df),
        "accuracy": float(acc),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "auc_ci95_lower": ci_lower,
        "auc_ci95_upper": ci_upper,
    }


def aggregate_fold_metrics(fold_metrics_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = (
        fold_metrics_df.groupby(group_col, dropna=False)
        .agg(
            n_folds=("Fold", "nunique"),
            n_cases_total=("n_cases", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_ci95_lower_mean=("auc_ci95_lower", "mean"),
            auc_ci95_upper_mean=("auc_ci95_upper", "mean"),
        )
        .reset_index()
    )
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    return agg


def convert_metric_columns_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_cols = [col for col in out.columns if col not in {"Center", "Scope", "Fold", "n_cases", "n_folds", "n_cases_total"}]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
        out[col] = out[col].round(2)
    return out


def merge_auc_ci_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"auc_ci95_lower", "auc_ci95_upper"}.issubset(out.columns):
        out["auc_ci95"] = out.apply(
            lambda row: f"({row['auc_ci95_lower']:.2f}, {row['auc_ci95_upper']:.2f})"
            if pd.notna(row["auc_ci95_lower"]) and pd.notna(row["auc_ci95_upper"])
            else np.nan,
            axis=1,
        )
        out = out.drop(columns=["auc_ci95_lower", "auc_ci95_upper"])

    if {"auc_ci95_lower_mean", "auc_ci95_upper_mean"}.issubset(out.columns):
        out["auc_ci95_mean"] = out.apply(
            lambda row: f"({row['auc_ci95_lower_mean']:.2f}, {row['auc_ci95_upper_mean']:.2f})"
            if pd.notna(row["auc_ci95_lower_mean"]) and pd.notna(row["auc_ci95_upper_mean"])
            else np.nan,
            axis=1,
        )
        out = out.drop(columns=["auc_ci95_lower_mean", "auc_ci95_upper_mean"])

    return out


def merge_mean_std_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric_base in ["accuracy", "auc"]:
        mean_col = f"{metric_base}_mean"
        std_col = f"{metric_base}_std"
        if {mean_col, std_col}.issubset(out.columns):
            out[metric_base] = out.apply(
                lambda row: f"{row[mean_col]:.2f}$\\pm${row[std_col]:.2f}"
                if pd.notna(row[mean_col]) and pd.notna(row[std_col])
                else np.nan,
                axis=1,
            )
            out = out.drop(columns=[mean_col, std_col])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate center-wise and global metrics from prediction CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/data2/pyq6817/CystX/radiomics_results/t2_mRMR_internal/predictions_all_folds.csv",
        help="Path to predictions CSV.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="metrics_results",
        help="Prefix for output CSV files.",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for AUC 95%% CI.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required_cols = {"ID", "Label", "Prediction", "Probability", "Fold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Center"] = df["ID"].apply(extract_center)
    df["Fold"] = df["Fold"].astype(str).str.strip().str.lower()

    unique_folds = sorted(df["Fold"].dropna().unique().tolist())
    center_fold_counts = df.groupby("Center", dropna=False)["Fold"].nunique(dropna=False)
    each_center_single_fold = bool((center_fold_counts <= 1).all())
    use_direct_center_and_global = (len(unique_folds) <= 1) or each_center_single_fold

    if use_direct_center_and_global:
        center_rows = []
        for center, center_df in df.groupby("Center", dropna=False):
            metrics = compute_metrics(center_df, args.bootstrap_iters, args.seed)
            center_rows.append({"Center": center, **metrics})
        center_summary = pd.DataFrame(center_rows).sort_values("Center")

        global_summary = pd.DataFrame(
            [{"Scope": "Global", **compute_metrics(df, args.bootstrap_iters, args.seed)}]
        )
    else:
        center_fold_rows = []
        for (center, fold), sub_df in df.groupby(["Center", "Fold"], dropna=False):
            metrics = compute_metrics(sub_df, args.bootstrap_iters, args.seed)
            center_fold_rows.append({"Center": center, "Fold": fold, **metrics})
        center_fold_df = pd.DataFrame(center_fold_rows).sort_values(["Center", "Fold"])
        center_summary = aggregate_fold_metrics(center_fold_df, "Center").sort_values("Center")

        global_fold_rows = []
        for fold, fold_df in df.groupby("Fold", dropna=False):
            metrics = compute_metrics(fold_df, args.bootstrap_iters, args.seed)
            global_fold_rows.append({"Scope": "Global", "Fold": fold, **metrics})
        global_fold_df = pd.DataFrame(global_fold_rows).sort_values("Fold")
        global_summary = aggregate_fold_metrics(global_fold_df, "Scope")

        center_fold_df = convert_metric_columns_to_percent(center_fold_df)
        global_fold_df = convert_metric_columns_to_percent(global_fold_df)
        center_fold_df = merge_auc_ci_columns(center_fold_df)
        global_fold_df = merge_auc_ci_columns(global_fold_df)
        center_fold_df.to_csv(f"{args.output_prefix}_center_fold_metrics.csv", index=False)
        global_fold_df.to_csv(f"{args.output_prefix}_global_fold_metrics.csv", index=False)

    center_summary = convert_metric_columns_to_percent(center_summary)
    global_summary = convert_metric_columns_to_percent(global_summary)
    center_summary = merge_mean_std_columns(center_summary)
    global_summary = merge_mean_std_columns(global_summary)
    center_summary = merge_auc_ci_columns(center_summary)
    global_summary = merge_auc_ci_columns(global_summary)
    center_summary.to_csv(f"{args.output_prefix}_center_summary.csv", index=False)
    global_summary.to_csv(f"{args.output_prefix}_global_summary.csv", index=False)

    print(f"Input file: {args.input_csv}")
    print(f"Detected folds: {unique_folds}")
    print("Center summary:")
    print(center_summary.to_string(index=False))
    print("\nGlobal summary:")
    print(global_summary.to_string(index=False))
    print("\nSaved:")
    print(f" - {args.output_prefix}_center_summary.csv")
    print(f" - {args.output_prefix}_global_summary.csv")
    if not use_direct_center_and_global:
        print(f" - {args.output_prefix}_center_fold_metrics.csv")
        print(f" - {args.output_prefix}_global_fold_metrics.csv")


if __name__ == "__main__":
    main()
