"""Select validation-only thresholds for radiomics experiments."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from common import N_FOLDS, DEFAULT_FEATURES, load_features, modality_dir, resolve_feature_path
from train import predict


def calibrate(probabilities, labels):
    probabilities = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    for threshold in sorted([0]+list(set(probabilities))+[1]):
        tn, fp, fn, tp = confusion_matrix(labels, probabilities>=threshold, labels=[0, 1]).ravel()
        spec = tn/(tn+fp) if tn+fp else 0.0
        if spec>=0.9:
            return float(threshold)
    print("No valid threshold found. Defaulting to 0.5.")
    return 0.5


def calibrate_and_save_thresholds(args):
    output = modality_dir(args.output_dir, args.t)
    thresholds = []
    for fold in range(N_FOLDS):
        bundle = joblib.load(output/f"fold{fold}"/"model.joblib")
        features = load_features(resolve_feature_path(args.features_path or bundle["feature_source"], args.t))
        partitions = pd.DataFrame(bundle["partitions"])
        val = partitions[partitions.Split=="val"]
        threshold = float(calibrate(predict(bundle, features, val), val.Label.to_numpy()))
        thresholds.append(threshold)
        print(f"Fold {fold} validation threshold: {threshold:.6f}")
    (output/"thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return thresholds


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./saved")
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES), help="Feature folder or a specific CSV")
    parser.add_argument("-t", "--t", type=int, choices=[1, 2], default=1)
    calibrate_and_save_thresholds(parser.parse_args())
