import numpy as np
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
