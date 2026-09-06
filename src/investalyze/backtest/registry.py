"""Registry of trading signals.

A signal is a two-stage callable whose contract lives in `engine.py`, next to the loop that consumes
it: the factory runs once before the loop and does the heavy vectorized work over the whole panel,
reading whatever it needs from the database, and returns a step function the loop calls per session.
Cross-sectional factor signals and per-ticker timing rules both take this shape, so one engine runs
both and `kind` decides which evaluation report they earn.

Adding a signal is one factory function plus a `SIGNALS` line.
"""

import duckdb

from investalyze.backtest.engine import (
    CROSS_SECTIONAL,
    HOLD,
    KINDS,
    TIMING,
    Decision,
    SignalFactory,
    SignalStep,
)
from investalyze.backtest.market import MarketData
from investalyze.backtest.signals import cross_sectional, timing

__all__ = ['CROSS_SECTIONAL', 'HOLD', 'KINDS', 'SIGNALS', 'SIGNAL_DESCRIPTIONS', 'TIMING',
           'Decision', 'SignalFactory', 'SignalStep', 'hold_cash']


def hold_cash(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Signal that never trades, giving every run a flat baseline to compare against.

    `con` and `params` are accepted to match the signal factory shape and are unused.
    """

    def step(session: int, book) -> Decision:
        """Hold whatever the book already holds."""
        return HOLD

    return step


# name -> (kind, signal factory); kind decides which evaluation report the run earns
SIGNALS: dict[str, tuple[str, SignalFactory]] = {
    'hold_cash': (TIMING, hold_cash),
    'factor_composite': (CROSS_SECTIONAL, cross_sectional.factor_composite),
    'single_factor': (CROSS_SECTIONAL, cross_sectional.single_factor),
    'sma_cross': (TIMING, timing.sma_cross),
    'breakout_52w': (TIMING, timing.breakout_52w),
    'ibs_band_reversion': (TIMING, timing.ibs_band_reversion),
}

# name -> one-line description, shown in `--help`; keys must match SIGNALS exactly
SIGNAL_DESCRIPTIONS: dict[str, str] = {
    'hold_cash': 'never trades; the flat baseline every run is compared against',
    'factor_composite': 'top bucket of an equal-weighted composite of the selected factors',
    'single_factor': 'top bucket of one factor, ranked across the universe',
    'sma_cross': 'hold a name while its fast moving average is above its slow one',
    'breakout_52w': 'buy a name at a new high over the lookback, exit on a trailing stop',
    'ibs_band_reversion': 'buy a close under a volatility band on weak intraday strength, exit on a rally',
}
