import numpy as np


def macro_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        total[c] = mask.sum()
        correct[c] = (preds[mask] == c).sum()
    has_support = total > 0
    return (correct[has_support] / total[has_support]).mean()
