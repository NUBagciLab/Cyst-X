"""Cyst-X mRMR-like ranking: MI relevance minus absolute correlation redundancy."""
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def _discretize_for_mrmr(X, bins=5):
    discrete = np.zeros_like(X, dtype=int)
    for column in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, column], np.linspace(0, 1, bins+1)))
        if len(edges)>2:
            discrete[:, column] = np.digitize(X[:, column], edges[1:-1], right=False)
    return discrete


def select_mrmr_features(X_scaled, y, k):
    k = min(k, X_scaled.shape[1])
    if k<=0:
        raise ValueError("k must be >= 1")
    relevance = mutual_info_classif(_discretize_for_mrmr(X_scaled), y, discrete_features=True, random_state=42)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.nan_to_num(np.abs(np.atleast_2d(np.corrcoef(X_scaled, rowvar=False))),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    selected = [int(np.argmax(relevance))]
    candidates = set(range(X_scaled.shape[1])) - set(selected)
    while len(selected)<k and candidates:
        best = max(sorted(candidates), key=lambda i: relevance[i]-float(np.mean(correlation[i, selected])))
        selected.append(best)
        candidates.remove(best)
    return np.array(selected, dtype=int)
