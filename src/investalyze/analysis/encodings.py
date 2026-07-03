"""Column-oriented encoders: turn windows-in-columns into comparable shapes.

Each input is a 2-D array with one window per column and time down the rows. Parameters are
learned per column from `history` only; the fitted transformer applied to a `future` slice uses
those same parameters, so nothing from the future leaks into the encoded history. The sklearn
scalers (`zscore`, `minmax`, `demean`) additionally guard zero-variance columns, so a flat window
encodes to zeros rather than NaN.
"""

from typing import Self

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def zscore() -> StandardScaler:
    """Per-window standardisation to mean 0, std 1 (learned per column from history)."""
    return StandardScaler()


def minmax() -> MinMaxScaler:
    """Per-window rescale to [0, 1] (learned per column from history)."""
    return MinMaxScaler()


def demean() -> StandardScaler:
    """Per-window centring to mean 0, keeping absolute amplitude."""
    return StandardScaler(with_std=False)


class RebaseTo100:
    """Rebase each window to start at 100 (`column / column[0] * 100`)."""

    def fit(self, history: np.ndarray) -> Self:
        """Learn each window's base: its first-row (earliest) price."""
        self.base_ = history[[0], :]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Rebase `x` by the fitted base; a future stays continuous with its history."""
        return x / self.base_ * 100.0


class LogReturns:
    """Per-window log returns (`diff(log(column))` down the rows)."""

    def fit(self, history: np.ndarray) -> Self:
        """Store each window's last observed price as the bridge anchor for the future."""
        self.anchor_ = history[[-1], :]
        return self

    def transform(self, history: np.ndarray) -> np.ndarray:
        """Log returns within `history`; yields `time_steps - 1` rows."""
        return np.diff(np.log(history), axis=0)

    def transform_future(self, future: np.ndarray) -> np.ndarray:
        """Log returns for `future`, its leading return bridged from the history's last price."""
        bridged = np.vstack([self.anchor_, future])
        return np.diff(np.log(bridged), axis=0)
