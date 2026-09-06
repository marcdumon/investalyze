"""Per-ticker timing rules: each name's own history decides whether it is held.

These take the same shape as the cross-sectional signals, so the same engine runs both. The factory
precomputes the rolling quantities over the whole panel as wide matrices, and the step function reads
one row and keeps its own per-name state. Nothing here reads a session later than the one it is on.

A rule emits target weights only on the sessions its membership changes, so the engine is asked to
trade when the rule changes its mind rather than every session.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.backtest.book import Book
from investalyze.backtest.engine import HOLD, Decision, SignalStep
from investalyze.backtest.market import MarketData
from investalyze.backtest.orders import STOP, OrderBatch

DEFAULT_FAST = 50
DEFAULT_SLOW = 200
DEFAULT_LOOKBACK = 252
DEFAULT_STOP_FRACTION = 0.15
DEFAULT_MAX_POSITIONS = 20
DEFAULT_BAND_WINDOW = 25
DEFAULT_BAND_MULT = 2.25
DEFAULT_IBS_THRESHOLD = 0.6


def _equal_weights(held: np.ndarray, max_positions: int) -> np.ndarray:
    """Equal weight across the held names, sized so a full book holds `max_positions` of them."""
    targets = np.zeros(len(held), dtype=np.float64)
    count = int(held.sum())
    if count:
        targets[held] = 1.0 / max(count, max_positions)
    return targets


def sma_cross(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Hold a name while its fast moving average sits above its slow one.

    `con` is accepted to match the signal factory shape and is unused.
    """
    fast_window = int(params.get('fast', DEFAULT_FAST))
    slow_window = int(params.get('slow', DEFAULT_SLOW))
    max_positions = int(params.get('max_positions', DEFAULT_MAX_POSITIONS))
    if fast_window >= slow_window:
        raise ValueError(f'fast window {fast_window} must be shorter than slow window {slow_window}')

    closes = pd.DataFrame(market.adjusted_close)
    fast = closes.rolling(fast_window, min_periods=fast_window).mean()
    slow = closes.rolling(slow_window, min_periods=slow_window).mean()
    above = (fast > slow).to_numpy()
    previous = {'held': None}

    def step(session: int, book: Book) -> Decision:
        """Rebalance to the names above their slow average, on the sessions that set changes."""
        held = above[session]
        if previous['held'] is not None and np.array_equal(held, previous['held']):
            return HOLD
        previous['held'] = held.copy()
        return Decision(targets=_equal_weights(held, max_positions), orders=None)

    return step


def breakout_52w(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Buy a name making a new high over the lookback, and rest a trailing stop under the position.

    The stop is restated every session from the running high, so it ratchets up and never down. The
    entry is expressed as a target weight and the exit as a stop order, which is the pairing a target
    weight alone cannot express: a target says where to be at a close, a stop says what to do inside
    a session.

    `con` is accepted to match the signal factory shape and is unused.
    """
    lookback = int(params.get('lookback', DEFAULT_LOOKBACK))
    stop_fraction = float(params.get('stop_fraction', DEFAULT_STOP_FRACTION))
    max_positions = int(params.get('max_positions', DEFAULT_MAX_POSITIONS))
    if not 0 < stop_fraction < 1:
        raise ValueError(f'stop_fraction must be in (0, 1), got {stop_fraction}')

    closes = pd.DataFrame(market.adjusted_close)
    running_high = closes.rolling(lookback, min_periods=lookback).max()
    at_high = (closes >= running_high).to_numpy()
    highs = running_high.to_numpy()
    raw_high = pd.DataFrame(market.close).rolling(lookback, min_periods=lookback).max().to_numpy()
    previous = {'held': np.zeros(len(market.tickers), dtype=bool)}

    def step(session: int, book: Book) -> Decision:
        """Enter the names at a new high, and rest a stop under everything currently held."""
        breaking = at_high[session] & np.isfinite(highs[session])
        held = breaking | (book.shares > 0)
        targets = None
        if not np.array_equal(held, previous['held']):
            previous['held'] = held.copy()
            targets = _equal_weights(held, max_positions)

        protected = np.flatnonzero((book.shares > 0) & np.isfinite(raw_high[session]))
        orders = None
        if protected.size:
            orders = OrderBatch(
                asset=protected.astype(np.int32),
                qty=-book.shares[protected],
                kind=np.full(protected.size, STOP, dtype=np.int8),
                trigger=raw_high[session, protected] * (1 - stop_fraction),
                age=np.zeros(protected.size, dtype=np.int16),
            )
        if targets is None and orders is None:
            return HOLD
        return Decision(targets=targets, orders=orders)

    return step


def ibs_band_reversion(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Buy a close that falls through a volatility band on weak intraday strength, sell it on a rally.

    The band sits `band_mult` average ranges below the rolling high; a close under it, paired with a
    low internal bar strength (IBS, the close's place between the low and the high), is the entry.
    Position stays open until the close beats the prior session's high.

    `con` is accepted to match the signal factory shape and is unused.
    """
    window = int(params.get('window', DEFAULT_BAND_WINDOW))
    band_mult = float(params.get('band_mult', DEFAULT_BAND_MULT))
    ibs_threshold = float(params.get('ibs_threshold', DEFAULT_IBS_THRESHOLD))
    max_positions = int(params.get('max_positions', DEFAULT_MAX_POSITIONS))
    if not 0 < ibs_threshold < 1:
        raise ValueError(f'ibs_threshold must be in (0, 1), got {ibs_threshold}')

    high = pd.DataFrame(market.high)
    low = pd.DataFrame(market.low)
    close = pd.DataFrame(market.close)
    day_range = high - low
    avg_range = day_range.rolling(window, min_periods=window).mean()
    band = high.rolling(window, min_periods=window).max() - band_mult * avg_range
    ibs = (close - low) / day_range.replace(0.0, np.nan)

    entry = ((close < band) & (ibs < ibs_threshold)).to_numpy()
    beats_prior_high = (close.to_numpy() > high.shift(1).to_numpy())
    previous = {'held': np.zeros(len(market.tickers), dtype=bool)}

    def step(session: int, book: Book) -> Decision:
        """Rebalance to the names either newly entered or still held and not yet exited."""
        held = (entry[session] | (book.shares > 0)) & ~beats_prior_high[session]
        if np.array_equal(held, previous['held']):
            return HOLD
        previous['held'] = held.copy()
        return Decision(targets=_equal_weights(held, max_positions), orders=None)

    return step
