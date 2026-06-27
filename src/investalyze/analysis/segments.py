"""Chop price series into fixed-length, rebased-to-100 segment vectors + a successor map.

Each segment is rebased to start at 100 (`v / v[0] * 100`) — level-invariant across
instruments while preserving amplitude, so plain Euclidean distance means "same shape,
same size of move". The output feeds motif clustering and next-segment transition analysis.
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


def build_segments(series: pd.DataFrame, *, length: int, stride: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Chop each instrument's `Value` series into rebased-to-100 segments.

    Per instrument: sort by Date, index rows 0..k-1, take windows starting at offsets
    0, stride, 2*stride, ... while `offset + length <= k`. Each window is rebased to 100;
    windows holding a NaN or non-positive value are dropped (cannot rebase). Segmenting is
    by row position — calendar gaps are ignored.

    Returns:
      X         float ndarray (n_segments, length) — `window / window[0] * 100`.
      meta      DataFrame: segment_id, Ticker, AssetClass, start_date, end_date, start_idx.
      successor int ndarray (n_segments,) — segment_id of the block starting `start_idx + length`
                in the same instrument, else -1.
    """
    if length <= 0:
        raise ValueError('length must be > 0')
    if stride <= 0:
        raise ValueError('stride must be > 0')

    rows: list[np.ndarray] = []
    meta_records: list[dict] = []

    for ticker, group in series.groupby('Ticker', sort=True):
        group = group.sort_values('Date')
        values = group['Value'].to_numpy(dtype=float)
        dates = group['Date'].to_numpy()
        asset_class = group['AssetClass'].iloc[0]
        for offset in range(0, len(values) - length + 1, stride):
            window = values[offset : offset + length]
            if not (window > 0).all():  # rejects non-positive and NaN (NaN > 0 is False)
                continue
            rows.append(window / window[0] * 100.0)
            meta_records.append({
                'segment_id': len(rows) - 1,
                'Ticker': ticker,
                'AssetClass': asset_class,
                'start_date': dates[offset],
                'end_date': dates[offset + length - 1],
                'start_idx': offset,
            })

    if not meta_records:
        return np.empty((0, length), dtype=float), pd.DataFrame(columns=_META_COLS), np.empty(0, dtype=int)

    # successor = the segment starting `length` rows later in the same instrument, else -1.
    by_key = {(rec['Ticker'], rec['start_idx']): rec['segment_id'] for rec in meta_records}
    successor = np.array([by_key.get((rec['Ticker'], rec['start_idx'] + length), -1) for rec in meta_records], dtype=int)

    return np.vstack(rows), pd.DataFrame(meta_records), successor
