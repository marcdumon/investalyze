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
