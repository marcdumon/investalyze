"""Dense per-run market matrices: everything the engine reads, materialized once.

Every matrix is (session, asset) on a shared trading calendar, so the engine indexes a whole
cross-section with one integer. `open_`, `high`, `low`, `close` and `volume` are the split-adjusted
vendor bars and are mutually consistent, so fills and marks come from them and dividends arrive as
cash. `adjusted_close` carries the dividend adjustment on top and serves the signal layer, where a
total-return series is what momentum, volatility and forward returns are defined on.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import panels
from investalyze.ingest import storage

ADV_WINDOW = 21          # sessions in the median dollar-volume estimate the fill cap uses
ERROR_CHECKS = ('ohlc_inconsistent', 'nonpositive_price')  # default anomalies to treat as untradable


@dataclass(frozen=True)
class MarketData:
    """Dense (session, asset) matrices for one run, plus the calendar and ticker axes."""

    dates: pd.DatetimeIndex     # (T,) the trading calendar
    tickers: np.ndarray         # (N,) sorted ticker symbols
    open_: np.ndarray           # (T, N) float64, split-adjusted
    high: np.ndarray            # (T, N) float64, split-adjusted
    low: np.ndarray             # (T, N) float64, split-adjusted
    close: np.ndarray           # (T, N) float64, split-adjusted
    volume: np.ndarray          # (T, N) float32, shares traded
    dividend: np.ndarray        # (T, N) float64, cash per share on the ex-date, 0 elsewhere
    adv: np.ndarray             # (T, N) float32, 21-session median of close * volume
    tradable: np.ndarray        # (T, N) bool, whether the engine may fill in this name today
    adjusted_close: np.ndarray  # (T, N) float64, dividend-adjusted close, for the signal layer

    @property
    def shape(self) -> tuple[int, int]:
        """Sessions by assets."""
        return len(self.dates), len(self.tickers)

    def index_of(self, ticker: str) -> int:
        """Column index of one ticker."""
        found = np.searchsorted(self.tickers, ticker)
        if found >= len(self.tickers) or self.tickers[found] != ticker:
            raise KeyError(f'{ticker} is not in this run universe')
        return int(found)


def _wide(long: pd.DataFrame, column: str, calendar: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """One long column widened to the calendar by ticker grid."""
    frame = long[column].unstack('Ticker')
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.reindex(index=calendar, columns=tickers)


def _anomaly_mask(con: duckdb.DuckDBPyConnection, tickers: list[str], calendar: pd.DatetimeIndex,
                  checks: list[str]) -> pd.DataFrame:
    """True where a quality check flagged that ticker and session as an error."""
    blank = pd.DataFrame(False, index=calendar, columns=tickers)
    if not checks or not storage.table_exists(con, 'anomalies'):
        return blank
    flagged = con.execute("""
        SELECT DISTINCT Ticker, Date
        FROM anomalies
        WHERE Severity = 'error' AND Date IS NOT NULL
          AND CheckName IN (SELECT unnest(?::VARCHAR[]))
          AND Ticker IN (SELECT unnest(?::VARCHAR[]))
    """, [checks, tickers]).df()
    for ticker, date in zip(flagged['Ticker'], pd.DatetimeIndex(flagged['Date'])):
        if date in blank.index and ticker in blank.columns:
            blank.loc[date, ticker] = True
    return blank


def build_market(con: duckdb.DuckDBPyConnection, tickers: list[str], *, start: str, end: str,
                 exclude_checks: list[str] | None = None) -> MarketData:
    """Load one run's price, volume and dividend matrices onto a shared trading calendar.

    The calendar comes from `panels.trading_calendar`, so holiday stubs that carry a handful of rows
    stay out of the session grid the engine steps through.
    """
    if not tickers:
        raise ValueError('tickers is empty')
    calendar = panels.trading_calendar(con, start=start, end=end)
    if calendar.empty:
        raise ValueError(f'no trading sessions between {start} and {end}')
    ordered = sorted(tickers)

    long = con.execute("""
        SELECT Ticker, Date, O, H, L, C, V, AC
        FROM prices
        WHERE Ticker IN (SELECT unnest(?::VARCHAR[])) AND Date IN (SELECT unnest(?::DATE[]))
    """, [ordered, list(calendar.date)]).df()
    if long.empty:
        raise ValueError('no priced rows for the requested tickers on this calendar')
    indexed = long.set_index(['Date', 'Ticker'])

    open_ = _wide(indexed, 'O', calendar, ordered)
    high = _wide(indexed, 'H', calendar, ordered)
    low = _wide(indexed, 'L', calendar, ordered)
    close = _wide(indexed, 'C', calendar, ordered)
    volume = _wide(indexed, 'V', calendar, ordered)
    adjusted = _wide(indexed, 'AC', calendar, ordered)

    dividends = con.execute("""
        SELECT Ticker, Date, sum(Dividend) AS Dividend
        FROM dividends
        WHERE Ticker IN (SELECT unnest(?::VARCHAR[])) AND Date IN (SELECT unnest(?::DATE[]))
        GROUP BY Ticker, Date
    """, [ordered, list(calendar.date)]).df()
    dividend = pd.DataFrame(0.0, index=calendar, columns=ordered)
    if not dividends.empty:
        paid = _wide(dividends.set_index(['Date', 'Ticker']), 'Dividend', calendar, ordered)
        dividend = paid.fillna(0.0)

    turnover = close * volume
    adv = turnover.rolling(ADV_WINDOW, min_periods=1).median()

    checks = list(ERROR_CHECKS) if exclude_checks is None else list(exclude_checks)
    masked = _anomaly_mask(con, ordered, calendar, checks)
    tradable = ((close > 0) & (open_ > 0) & (volume > 0) & ~masked).fillna(False)

    return MarketData(
        dates=calendar,
        tickers=np.array(ordered),
        open_=open_.to_numpy(dtype=np.float64),
        high=high.to_numpy(dtype=np.float64),
        low=low.to_numpy(dtype=np.float64),
        close=close.to_numpy(dtype=np.float64),
        volume=volume.to_numpy(dtype=np.float32),
        dividend=dividend.to_numpy(dtype=np.float64),
        adv=adv.to_numpy(dtype=np.float32),
        tradable=tradable.to_numpy(dtype=bool),
        adjusted_close=adjusted.to_numpy(dtype=np.float64),
    )


def benchmark_caveat(name: str, benchmark_end: pd.Timestamp | None, run_end: pd.Timestamp) -> str | None:
    """A note when the benchmark stops before the run does, so a ragged comparison is never silent."""
    if benchmark_end is None:
        return f'benchmark {name} has no data over this range, so no relative statistics are reported'
    if benchmark_end < run_end:
        short = (run_end - benchmark_end).days
        return (f'benchmark {name} covers only to {benchmark_end.date()}, {short} days short of the '
                'backtest end; relative statistics use the overlap')
    return None
