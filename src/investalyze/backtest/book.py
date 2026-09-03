"""Portfolio state the engine mutates as it walks the calendar.

Share counts are in the split-adjusted units the price matrices use, so a position's count is
constant across a split and the engine needs no corporate-action handling. Counts are fractional and
signed: negative is a short. The per-session arrays are filled in as the loop runs and become the
equity curve the report is built from.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Book:
    """Cash, share counts, and the per-session history a run accumulates."""

    cash: float
    shares: np.ndarray                        # (N,) float64, split-adjusted shares, negative for shorts
    cost_basis: np.ndarray                    # (N,) float64, average fill price behind each position
    equity: np.ndarray                        # (T,) float64, cash plus positions marked at the close
    gross: np.ndarray                         # (T,) float64, sum of absolute position values
    net: np.ndarray                           # (T,) float64, signed sum of position values
    fills: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, cash: float, n_assets: int, n_sessions: int) -> 'Book':
        """A book holding only cash, sized for a run of `n_sessions` over `n_assets` names."""
        return cls(
            cash=float(cash),
            shares=np.zeros(n_assets, dtype=np.float64),
            cost_basis=np.zeros(n_assets, dtype=np.float64),
            equity=np.full(n_sessions, np.nan, dtype=np.float64),
            gross=np.zeros(n_sessions, dtype=np.float64),
            net=np.zeros(n_sessions, dtype=np.float64),
        )

    def position_value(self, close: np.ndarray) -> np.ndarray:
        """Signed market value of every position at the given close prices."""
        return self.shares * np.nan_to_num(close, nan=0.0)

    def blotter(self) -> pd.DataFrame:
        """Every fill of the run, one row each, in the order they happened."""
        return pd.DataFrame(self.fills)
