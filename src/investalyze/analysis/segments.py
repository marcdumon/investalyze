"""Chop price series into fixed-width raw windows for motif and transition analysis.

`build_segments` emits raw price windows of a single `window_length`. Encoding (rebase-to-100,
log-returns, ...) and the segment/successor split are downstream choices: apply a function from
`investalyze.analysis.encodings`, then slice each row at the segment length you pick.
"""

import duckdb
import numpy as np
import pandas as pd

_STOCK_CLASS = 'stocks'
_MARKET_CLASSES = ('indices', 'bonds', 'currencies')
_META_COLS = ['segment_id', 'Ticker', 'AssetClass', 'start_date', 'end_date', 'start_idx']


def load_series(
    con: duckdb.DuckDBPyConnection, *, classes: list[str], tickers: list[str] | None = None, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Long frame `[Ticker, Date, AssetClass, Value]` for the selected universe.

    `stocks` reads `prices.AC` (split/dividend adjusted); the market classes read
    `market_data.C` filtered by `AssetClass` (that table has no adjusted close). `tickers`,
    if given, further restricts within the chosen classes. Sorted by Ticker then Date.
    """
    def _filters() -> tuple[str, list]:
        """Shared WHERE tail (ticker list + date range): the SQL fragment and its bind params."""
        clauses: list[str] = []
        values: list = []
        if tickers is not None:
            clauses.append(f"Ticker IN ({', '.join('?' for _ in tickers)})")
            values.extend(tickers)
        if start is not None:
            clauses.append('Date >= ?')
            values.append(start)
        if end is not None:
            clauses.append('Date <= ?')
            values.append(end)
        where = (' AND ' + ' AND '.join(clauses)) if clauses else ''
        return where, values

    selects: list[str] = []
    params: list = []

    if _STOCK_CLASS in classes:
        where, values = _filters()
        selects.append(f"SELECT Ticker, Date, '{_STOCK_CLASS}' AS AssetClass, AC AS Value "
                       f'FROM prices WHERE AC IS NOT NULL{where}')
        params += values

    market = [c for c in classes if c in _MARKET_CLASSES]
    if market:
        where, values = _filters()
        class_placeholders = ', '.join('?' for _ in market)
        selects.append(f'SELECT Ticker, Date, AssetClass, C AS Value '
                       f'FROM market_data WHERE AssetClass IN ({class_placeholders})'
                       f' AND C IS NOT NULL{where}')
        params += market + values

    if not selects:
        return pd.DataFrame({'Ticker': [], 'Date': [], 'AssetClass': [], 'Value': []})

    sql = ' UNION ALL '.join(selects) + ' ORDER BY Ticker, Date'
    return con.execute(sql, params).df()


def build_segments(series: pd.DataFrame, *, window_length: int, stride: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Chop each instrument's `Value` series into fixed-width raw windows.

    Per instrument: sort by Date, index rows 0..k-1, take windows starting at offsets
    0, stride, 2*stride, ... while the window fits (`offset + window_length <= k`). Each row is the
    raw `Value` series over the window; encode via `investalyze.analysis.encodings` and slice into
    segment / successor downstream at whatever segment length you pick. Windows holding a NaN or
    non-positive value are dropped (cannot rebase / log). Segmenting is by row position — calendar
    gaps are ignored.

    Returns:
      W    float ndarray (n_segments, window_length) — raw prices.
      meta DataFrame: segment_id, Ticker, AssetClass, start_date, end_date, start_idx.
    """
    if window_length <= 0:
        raise ValueError('window_length must be > 0')
    if stride <= 0:
        raise ValueError('stride must be > 0')

    rows: list[np.ndarray] = []
    meta_records: list[dict] = []

    for ticker, group in series.groupby('Ticker', sort=True):
        group = group.sort_values('Date')
        values = group['Value'].to_numpy(dtype=float)
        dates = group['Date'].to_numpy()
        asset_class = group['AssetClass'].iloc[0]
        for offset in range(0, len(values) - window_length + 1, stride):
            window = values[offset : offset + window_length]
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
        return np.empty((0, window_length), dtype=float), pd.DataFrame(columns=_META_COLS)

    return np.vstack(rows), pd.DataFrame(meta_records)
