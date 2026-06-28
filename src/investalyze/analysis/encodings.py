"""Pure vector encodings for price/return windows: turn raw rows into comparable shapes.

Each function maps a 2-D array (rows = windows) to an encoded array, independently per row.
These are the reusable building blocks downstream code (segment clustering, plotting, transition
analysis) composes after pulling raw windows from `segments.build_segments`.
"""

import numpy as np


def rebase_to_100(windows: np.ndarray) -> np.ndarray:
    """Rebase each row to start at 100 (`row / row[0] * 100`). Level-invariant, amplitude-preserving."""
    return windows / windows[:, [0]] * 100.0


def log_returns(windows: np.ndarray) -> np.ndarray:
    """Per-row log returns `diff(log(row))`, shape `(n, m - 1)`. Additive, amplitude-aware."""
    return np.diff(np.log(windows), axis=1)


def demean(windows: np.ndarray) -> np.ndarray:
    """Center each row on its own mean (`row - row.mean()`). Removes level, keeps absolute amplitude."""
    return windows - windows.mean(axis=1, keepdims=True)


def zscore(windows: np.ndarray) -> np.ndarray:
    """Standardize each row to mean 0, std 1 (`(row - mean) / std`). Removes level and scale -> pure shape."""
    return (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)


def minmax(windows: np.ndarray) -> np.ndarray:
    """Rescale each row to [0, 1] (`(row - min) / (max - min)`). Removes level and scale via the range."""
    lo = windows.min(axis=1, keepdims=True)
    hi = windows.max(axis=1, keepdims=True)
    return (windows - lo) / (hi - lo)
