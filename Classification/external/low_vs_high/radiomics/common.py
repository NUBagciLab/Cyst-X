"""Feature loading and centralized-compatible external train/validation/test partitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import get_data_list

PROJECT_ROOT = Path(__file__).resolve().parents[4]
N_FOLDS = 7
EXTERNAL = True
DEFAULT_DATA = "/dataset/Cyst-X-resampled/classification"
DEFAULT_FEATURES = PROJECT_ROOT / "features"


def resolve_feature_path(path, t):
    path = Path(path)
    return path if path.suffix.lower() == ".csv" else path / f"radiomics_features_t{t}.csv"


def case_id(name):
    return Path(str(name)).name.removesuffix(".gz").removesuffix(".nii")


def load_features(path):
    frame = pd.read_csv(path)
    if "Image" not in frame and "ID" not in frame:
        raise ValueError("Feature CSV needs an Image or ID column.")
    ids = frame.pop("Image" if "Image" in frame else "ID").map(case_id)
    if ids.duplicated().any():
        raise ValueError("Feature CSV contains duplicate case IDs.")
    frame = frame.loc[:, ~frame.columns.str.startswith("diagnostics_")]
    frame = frame.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    frame.index = ids
    if frame.empty:
        raise ValueError("Feature table is empty.")
    return frame


def get_partitions(data_path, t, fold):
    if fold not in range(N_FOLDS):
        raise ValueError("Held-out center must be between 0 and 6.")
    rows = []
    for center in range(7):
        images, labels = get_data_list(root=data_path, t=t, center=center)
        if center == fold:
            rows.extend({"ID":case_id(name), "Label":int(np.asarray(label).item()),
                         "Center":center, "Split":"test"} for name,label in zip(images,labels))
            print(f"Center {center+1}: {len(images)} external test cases (held out)")
            continue
        if len(images)<2:
            train, y_train, val, y_val = images, labels, [], []
        else:
            classes, counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
            n_val = int(np.ceil(len(images)*0.2))
            can_stratify = len(classes)>1 and counts.min()>=2 and min(n_val,len(images)-n_val)>=len(classes)
            train, val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=42,
                stratify=labels if can_stratify else None, shuffle=True)
        for split, names, targets in (("train",train,y_train),("val",val,y_val)):
            rows.extend({"ID":case_id(name), "Label":int(np.asarray(label).item()),
                         "Center":center, "Split":split} for name,label in zip(names,targets))
        print(f"Center {center+1}: {len(train)} train | {len(val)} val")
    partitions = pd.DataFrame(rows)
    if partitions.ID.duplicated().any():
        raise ValueError("Duplicate IDs found across centers or partitions.")
    if set(partitions.Split)!={"train","val","test"}:
        raise ValueError("Training, validation and external test must all contain cases.")
    return partitions


def get_cohort_metadata(data_path, t):
    """Apply the same cohort membership as data_loader when exporting risk metadata."""
    table = pd.read_excel(Path(data_path)/f"IPMN_labels_t{t}_total.xlsx", usecols=[0,5])
    table = table.dropna(subset=[table.columns[1]]).copy()
    table.columns = ["ID", "Risk Assessment"]
    table.ID = table.ID.map(case_id)
    images, _ = get_data_list(root=data_path, t=t)
    eligible = {case_id(name) for name in images}
    return table[table.ID.isin(eligible)].copy()


def matrix(features, partition, columns=None):
    missing = partition.loc[~partition.ID.isin(features.index), "ID"]
    if len(missing):
        raise ValueError(f"Missing features for {len(missing)} cases, e.g. {missing.iloc[0]}")
    return features.loc[partition.ID, columns if columns is not None else features.columns]


def modality_dir(output_dir, t):
    return Path(output_dir) / "random_forest" / f"t{t}"
