"""Chop price series into fixed-width raw windows for motif and transition analysis.

`build_segments` emits raw price windows of a single `window_length`. Encoding (rebase-to-100,
log-returns, ...) and the segment/successor split are downstream choices: apply a function from
`investalyze.analysis.encodings`, then slice each column at the segment length you pick.
"""

import duckdb
import numpy as np
import pandas as pd

_STOCK_CLASS = 'stocks'
_MARKET_CLASSES = ('indices', 'bonds', 'currencies')
_ALL_CLASSES = frozenset({_STOCK_CLASS, *_MARKET_CLASSES})
_META_COLS = ['segment_id', 'Ticker', 'AssetClass', 'start_date', 'end_date', 'start_idx']


def list_tickers(con: duckdb.DuckDBPyConnection, *, classes: list[str]) -> list[str]:
    """Distinct tickers belonging to the given classes, sorted.

    `stocks` comes from `prices.AC` (split/dividend adjusted); the market classes come from
    `market_data.AssetClass`. Raises ValueError on an unrecognized class name.
    """
    unknown = [c for c in classes if c not in _ALL_CLASSES]
    if unknown:
        raise ValueError(f'unknown class(es) {unknown}, known: {sorted(_ALL_CLASSES)}')

    tickers: set[str] = set()
    if _STOCK_CLASS in classes:
        rows = con.execute('SELECT DISTINCT Ticker FROM prices WHERE AC IS NOT NULL').fetchall()
        tickers.update(t for (t,) in rows)

    market = [c for c in classes if c in _MARKET_CLASSES]
    if market:
        rows = con.execute(
            'SELECT DISTINCT Ticker FROM market_data WHERE AssetClass = ANY(?) AND C IS NOT NULL', [market]
        ).fetchall()
        tickers.update(t for (t,) in rows)

    return sorted(tickers)


def get_series(
    con: duckdb.DuckDBPyConnection, tickers: list[str], *, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Long frame `[Ticker, Date, AssetClass, Price]` for the given tickers.

    `stocks` tickers are matched against `prices.AC` (split/dividend adjusted); everything else
    against `market_data.C` (that table has no adjusted close), tagged with its own `AssetClass`.
    A ticker string resolves to at most one of the two tables. Sorted by Ticker then Date.
    """
    clauses = ['Ticker = ANY(?)']
    values: list = [tickers]
    if start is not None:
        clauses.append('Date >= ?')
        values.append(start)
    if end is not None:
        clauses.append('Date <= ?')
        values.append(end)
    where = ' AND '.join(clauses)

    stock_sql = f"SELECT Ticker, Date, '{_STOCK_CLASS}' AS AssetClass, AC AS Price FROM prices WHERE AC IS NOT NULL AND {where}"
    market_sql = f'SELECT Ticker, Date, AssetClass, C AS Price FROM market_data WHERE C IS NOT NULL AND {where}'
    sql = f'{stock_sql} UNION ALL {market_sql} ORDER BY Ticker, Date'
    return con.execute(sql, values + values).df()


def build_segments(series: pd.DataFrame, *, window_length: int, stride: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Chop each instrument's `Price` series into fixed-width raw windows.

    Per instrument: sort by Date, index rows 0..k-1, take windows starting at offsets
    0, stride, 2*stride, ... while the window fits (`offset + window_length <= k`). Each column is the
    raw `Price` series over the window; encode via `investalyze.analysis.encodings` and slice into
    segment / successor downstream at whatever segment length you pick. Windows holding a NaN or
    non-positive value are dropped (cannot rebase / log). Segmenting is by row position, calendar
    gaps are ignored.

    Returns:
      W    float ndarray (window_length, n_segments): each column is one window's raw prices.
      meta DataFrame aligned to W's columns: meta.iloc[i] describes W[:, i]. Columns:
           segment_id, Ticker, AssetClass, start_date, end_date, start_idx.
    """
    if window_length <= 0:
        raise ValueError('window_length must be > 0')
    if stride <= 0:
        raise ValueError('stride must be > 0')

    rows: list[np.ndarray] = []
    meta_records: list[dict] = []

    for ticker, group in series.groupby('Ticker', sort=True):
        group = group.sort_values('Date')
        prices = group['Price'].to_numpy(dtype=float)
        dates = group['Date'].to_numpy()
        asset_class = group['AssetClass'].iloc[0]
        for offset in range(0, len(prices) - window_length + 1, stride):
            window = prices[offset : offset + window_length]
            if not (window > 0).all():  # rejects non-positive and NaN (NaN > 0 is False)
                continue
            rows.append(window)
            meta_records.append({
                'segment_id': len(rows) - 1,
                'Ticker': ticker,
                'AssetClass': asset_class,
                'start_date': dates[offset],
                'end_date': dates[offset + window_length - 1],
                'start_idx': offset,
            })

    if not meta_records:
        return np.empty((window_length, 0), dtype=float), pd.DataFrame(columns=_META_COLS)

    return np.column_stack(rows), pd.DataFrame(meta_records)
