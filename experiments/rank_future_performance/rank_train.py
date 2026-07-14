import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_by_group(preds: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> float:
    correlations = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 2:
            continue
        corr, _ = spearmanr(preds[mask], targets[mask])
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else float('nan')


def relevance_grades(y: np.ndarray, groups: np.ndarray, n_grades: int = 5) -> np.ndarray:
    grades = np.zeros(len(y), dtype=np.int32)
    for group in np.unique(groups):
        mask = groups == group
        percentile = pd.Series(y[mask]).rank(pct=True).to_numpy()  # 0..1, higher = better within the group
        grades[mask] = np.clip((percentile * n_grades).astype(int), 0, n_grades - 1)
    return grades


def precision_at_k(preds: np.ndarray, targets: np.ndarray, groups: np.ndarray, k_frac: float) -> float:
    precisions = []
    for group in np.unique(groups):
        mask = groups == group
        n = int(mask.sum())
        if n < 2:
            continue
        k = max(1, round(n * k_frac))
        pred_top = set(np.argsort(-preds[mask])[:k])
        true_top = set(np.argsort(-targets[mask])[:k])
        precisions.append(len(pred_top & true_top) / k)
    return float(np.mean(precisions)) if precisions else float('nan')
