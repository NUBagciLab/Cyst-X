"""Evaluate radiomics on saved validation partitions; save Excel and text results."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from auc_ci import calculate_auc_ci, calculate_auc_ci_cv, highlight_errors
from common import DEFAULT_DATA, DEFAULT_FEATURES, case_id, load_features, modality_dir, resolve_feature_path
from train import predict


def write_validation_excel(frame, path):
    """Write one worksheet per model fold, preserving predictions and highlighting."""
    fold_column = 'Fold'
    with pd.ExcelWriter(path) as writer:
        for fold in range(5):
            sheet = frame.loc[frame[fold_column] == fold].reset_index(drop=True)
            if sheet["ID"].duplicated().any():
                raise ValueError(f"Duplicate validation case within fold {fold}.")
            sheet.style.apply(highlight_errors, axis=1).to_excel(
                writer, sheet_name=f"Fold_{fold + 1}", index=False)


def metrics(labels, probabilities, threshold):
    predicted = probabilities>=threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auc":float(roc_auc_score(labels, probabilities)), "acc":float(accuracy_score(labels, predicted)),
            "sens":float(tp/(tp+fn)) if tp+fn else 0.0, "spec":float(tn/(tn+fp)) if tn+fp else 0.0}


def evaluate(args):
    output = modality_dir(args.output_dir, args.t)
    if args.calibrate:
        path = output/"thresholds.json"
        thresholds = json.loads(path.read_text())
        if len(thresholds)!=5 or not np.isfinite(np.asarray(thresholds, dtype=float)).all():
            raise ValueError("Expected five validation-selected thresholds.")
    else:
        thresholds = [0.5]*5
    predictions, fold_metrics, folds_data, lines = [], [], [], []
    def report(message):
        print(message)
        lines.append(message)
    for fold in range(5):
        bundle = joblib.load(output/f"fold{fold}"/"model.joblib")
        features = load_features(resolve_feature_path(args.features_path or bundle["feature_source"], args.t))
        partitions = pd.DataFrame(bundle["partitions"])
        validation = partitions[partitions.Split=="val"].copy()
        probabilities = predict(bundle, features, validation)
        labels = validation.Label.to_numpy()
        threshold = thresholds[fold]
        values = metrics(labels, probabilities, threshold)
        fold_metrics.append(values)
        folds_data.append((labels, probabilities))
        predictions.append(pd.DataFrame({"ID":validation.ID.to_numpy(), "Label":labels,
            "Prediction":(probabilities>=threshold).astype(int), "Probability":probabilities,
            "Fold":fold, "Threshold":threshold}))
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(["ID", "Fold"]).any():
        raise ValueError("Duplicate case within a validation fold.")
    for fold, values in enumerate(fold_metrics):
        report(f"Fold {fold} validation acc {values['acc']:.4f} "
               f"sens {values['sens']:.4f} spec {values['spec']:.4f} auc {values['auc']:.4f}")
    means = {metric:float(np.mean([entry[metric] for entry in fold_metrics]))
             for metric in ("acc", "sens", "spec", "auc")}
    stds = {metric:float(np.std([entry[metric] for entry in fold_metrics]))
            for metric in means}
    report("Validation " + " ".join(f"{metric} {means[metric]:.4f}±{stds[metric]:.4f}"
                             for metric in ("acc", "sens", "spec", "auc")))
    ci = calculate_auc_ci_cv(folds_data, n_bootstraps=args.n_bootstraps)
    report(f"95%CI: [{ci[0]*100:.2f}, {ci[1]*100:.2f}]")
    tn, fp, fn, tp = confusion_matrix(result.Label, result.Prediction, labels=[0, 1]).ravel()
    pooled_auc = roc_auc_score(result.Label, result.Probability)
    pooled_ci = calculate_auc_ci(result.Label.to_numpy(), result.Probability.to_numpy(), n_bootstraps=args.n_bootstraps)
    report(f"Pooled AUC {pooled_auc:.4f} [95% CI {pooled_ci[0]:.4f}, {pooled_ci[1]:.4f}] "
           f"acc {accuracy_score(result.Label,result.Prediction):.4f} sens {tp/(tp+fn):.4f} spec {tn/(tn+fp):.4f}")
    plus_minus = r"$\pm$"
    report(f"{means['auc']*100:.2f}{plus_minus}{stds['auc']*100:.2f} & "
           f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}] & "
           f"{means['acc']*100:.2f}{plus_minus}{stds['acc']*100:.2f} & "
           f"{means['sens']*100:.2f}{plus_minus}{stds['sens']*100:.2f} & "
           f"{means['spec']*100:.2f}{plus_minus}{stds['spec']*100:.2f}")
    metadata = pd.read_excel(Path(args.data_path)/f"IPMN_labels_t{args.t}_total.xlsx", usecols=[0, 5])
    metadata = metadata.dropna(subset=[metadata.columns[1]])
    metadata.columns = ["ID", "Risk Assessment"]
    metadata.ID = metadata.ID.map(case_id)
    if not set(result.ID).issubset(set(metadata.ID)) or metadata.ID.duplicated().any():
        raise ValueError("Current label spreadsheet IDs differ from the saved evaluation cohort.")
    result = result.merge(metadata, on="ID", how="left", validate="many_to_one")
    if not np.array_equal((result["Risk Assessment"].to_numpy()//2).astype(int), result.Label):
        raise ValueError("Current risk labels differ from the training labels.")
    stem = "val_result_calibrated" if args.calibrate else "val_result"
    write_validation_excel(result, output / f'{stem}.xlsx')
    (output/f"{stem}.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES), help="Feature folder or a specific CSV")
    parser.add_argument("--output-dir", default="./saved")
    parser.add_argument("-t", "--t", type=int, choices=[1, 2], default=1)
    parser.add_argument("-c", "--calibrate", action="store_true", help="Use existing thresholds.json; never recalibrate")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.n_bootstraps<1:
        parser.error("n-bootstraps must be positive.")
    evaluate(args)
