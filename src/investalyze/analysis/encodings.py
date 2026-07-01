"""Pure vector encodings for price/return windows: turn raw rows into comparable shapes.

Each function encodes a 2-D `history` (rows = windows) independently per row. The normalisation
parameters are learned from `history` only; pass an optional `future` and it is transformed with
those same parameters, so nothing from the future leaks back into the encoded history (the
`scaler.fit(history).transform(future)` discipline, made structural by keeping the two arrays apart).

With `future=None` each function returns the encoded history. With a future it returns the pair
`(encoded_history, encoded_future)`.
"""

import numpy as np


def rebase_to_100(
    history: np.ndarray,
    future: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Rebase each row to start at 100 (`row / row[0] * 100`). Level-invariant, amplitude-preserving.

    The base is `history[0]`; the future is rebased by the same base, staying continuous with it.
    """
    base = history[:, [0]]
    hist = history / base * 100.0
    if future is None:
        return hist
    return hist, future / base * 100.0


def log_returns(
    history: np.ndarray,
    future: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-row log returns `diff(log(row))`. Additive, amplitude-aware.

    The history yields `m - 1` returns. The future's returns are measured against the history's last
    price first, so its leading return is the history→future step (causal); it keeps `future` width.
    """
    hist = np.diff(np.log(history), axis=1)
    if future is None:
        return hist
    bridged = np.concatenate([history[:, [-1]], future], axis=1)
    return hist, np.diff(np.log(bridged), axis=1)


def demean(
    history: np.ndarray,
    future: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Center each row on the history's mean (`row - mean`). Removes level, keeps absolute amplitude."""
    mean = history.mean(axis=1, keepdims=True)
    hist = history - mean
    if future is None:
        return hist
    return hist, future - mean


def zscore(
    history: np.ndarray,
    future: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Standardize each row by the history's mean and std (`(row - mean) / std`). Removes level and scale."""
    mean = history.mean(axis=1, keepdims=True)
    std = history.std(axis=1, keepdims=True)
    hist = (history - mean) / std
    if future is None:
        return hist
    return hist, (future - mean) / std


def minmax(
    history: np.ndarray,
    future: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Rescale each row to [0, 1] using the history's min and max (`(row - min) / (max - min)`).

    The future is scaled by the same range, so a future that exceeds the history's range lands outside
    [0, 1] — the honest result of refusing to peek at it when setting the bounds.
    """
    lo = history.min(axis=1, keepdims=True)
    hi = history.max(axis=1, keepdims=True)
    rng = hi - lo
    hist = (history - lo) / rng
    if future is None:
        return hist
    return hist, (future - lo) / rng
