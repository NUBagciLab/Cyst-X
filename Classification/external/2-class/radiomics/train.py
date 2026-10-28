"""Train external 2-class radiomics: train-only preprocessing/selection, validation RF tuning."""
import argparse
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from common import N_FOLDS, DEFAULT_DATA, DEFAULT_FEATURES, get_partitions, load_features, matrix, modality_dir, resolve_feature_path
from selection import select_mrmr_features


def train_fold(args):
    feature_path = resolve_feature_path(args.features_path or DEFAULT_FEATURES, args.t)
    features = load_features(feature_path)
    partitions = get_partitions(args.data_path, args.t, args.fold)
    missing = partitions.loc[~partitions.ID.isin(features.index), "ID"]
    if len(missing):
        raise ValueError(f"Missing features for {len(missing)} labeled cases, e.g. {missing.iloc[0]}")
    train = partitions[partitions.Split=="train"]
    val = partitions[partitions.Split=="val"]
    y_train, y_val = train.Label.to_numpy(), val.Label.to_numpy()
    if len(np.unique(y_train))!=2 or len(np.unique(y_val))!=2:
        raise ValueError("Pooled train and validation sets must each contain both classes.")
    preprocessing = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  VarianceThreshold(), StandardScaler())
    X_train = preprocessing.fit_transform(matrix(features, train))
    X_val = preprocessing.transform(matrix(features, val))
    kept_names = features.columns[preprocessing.named_steps["variancethreshold"].get_support()]
    ranking = select_mrmr_features(X_train, y_train, min(max(args.k_features), X_train.shape[1]))
    grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 0.5],
            "k_features": args.k_features}
    candidates = list(ParameterSampler(grid, n_iter=args.search_iterations, random_state=args.seed))
    best_key, best_bundle, records = None, None, []
    for params in tqdm(candidates, desc=f"RF validation search t{args.t} fold{args.fold}"):
        indices = ranking[:min(params["k_features"], len(ranking))]
        model = RandomForestClassifier(**{k:v for k,v in params.items() if k!="k_features"},
                                       random_state=args.seed, n_jobs=args.jobs)
        model.fit(X_train[:, indices], y_train)
        probabilities = model.predict_proba(X_val[:, indices])[:, 1]
        auc = float(roc_auc_score(y_val, probabilities))
        acc = float(accuracy_score(y_val, probabilities>=0.5))
        records.append({**params, "actual_k": len(indices), "val_auc": auc, "val_acc": acc})
        if best_key is None or (auc, acc)>best_key:
            best_key = (auc, acc)
            best_bundle = {"preprocessing": preprocessing, "model": model, "selected_indices": indices,
                           "feature_columns": list(features.columns), "selected_features": list(kept_names[indices]),
                           "params": params, "val_auc": auc, "val_acc": acc,
                           "fold": args.fold, "t": args.t, "seed": args.seed,
                           "selection": "MI relevance minus mean absolute correlation (original Cyst-X mRMR-like)",
                           "feature_source": str(feature_path.resolve())}
    output = modality_dir(args.output_dir, args.t) / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    best_bundle["partitions"] = partitions.to_dict("records")
    joblib.dump(best_bundle, output/"model.joblib")
    partitions.to_csv(output/"split.csv", index=False)
    pd.DataFrame(records).to_csv(output/"search_results.csv", index=False)
    config = {k:best_bundle[k] for k in ("params", "val_auc", "val_acc", "fold", "t", "seed",
                                        "selection", "feature_source", "selected_features")}
    config["versions"] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__}
    (output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {output}; validation AUC {best_key[0]:.4f}, accuracy {best_key[1]:.4f}")


def predict(bundle, features, partition):
    X = bundle["preprocessing"].transform(matrix(features, partition, bundle["feature_columns"]))
    return bundle["model"].predict_proba(X[:, bundle["selected_indices"]])[:, 1]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES), help="Feature folder or a specific CSV")
    parser.add_argument("--output-dir", default="./saved")
    parser.add_argument("-t", "--t", type=int, choices=[1, 2], default=1)
    parser.add_argument("-f", "--fold", type=int, choices=range(N_FOLDS), default=0)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--search-iterations", type=int, default=20)
    parser.add_argument("--k-features", nargs="+", type=int, default=[10, 20, 30, 50])
    args = parser.parse_args()
    if args.search_iterations<1 or min(args.k_features)<1:
        parser.error("Search iterations and feature counts must be positive.")
    train_fold(args)
