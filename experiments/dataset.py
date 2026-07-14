from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_CHANNELS = ('O', 'H', 'L', 'C', 'V')
_META_COLS = ['Ticker', 'start_date', 'end_date', 'start_idx']


def list_all_tickers(con: duckdb.DuckDBPyConnection, *, exclude: list[str] | None = None) -> list[str]:
    where = 'WHERE NOT (Ticker = ANY(?))' if exclude else ''
    values = [exclude] if exclude else []
    return sorted(t for (t,) in con.execute(f'SELECT DISTINCT Ticker FROM prices {where}', values).fetchall())


def sample_tickers(
    con: duckdb.DuckDBPyConnection, n: int, *, seed: int | None = None, exclude: list[str] | None = None
) -> list[str]:
    all_tickers = list_all_tickers(con, exclude=exclude)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_tickers), size=min(n, len(all_tickers)), replace=False)
    return sorted(all_tickers[i] for i in idx)


def load_universe(name: str, data_root: str | Path = '../../data') -> list[str]:
    """Load a ticker universe saved by apps/screener as a list of tickers.

    The default data_root matches running a notebook inside an experiment directory.
    """
    return pd.read_csv(Path(data_root) / 'universes' / f'{name}.csv')['Ticker'].tolist()


def get_ohlcv_series(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str] | None,
    *,
    exclude: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    clauses = []
    values: list = []
    if tickers is not None:
        clauses.append('Ticker = ANY(?)')
        values.append(tickers)
    if exclude:
        clauses.append('NOT (Ticker = ANY(?))')
        values.append(exclude)
    if start is not None:
        clauses.append('Date >= ?')
        values.append(start)
    if end is not None:
        clauses.append('Date <= ?')
        values.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
    sql = f'SELECT Ticker, Date, O, H, L, C, V, AC FROM prices {where} ORDER BY Ticker, Date'
    return con.execute(sql, values).df()


def get_ticker_labels(con: duckdb.DuckDBPyConnection, column: str) -> pd.DataFrame:
    """Ticker -> `column` ('Sector' or 'Industry') from the companies table, blank/missing labels dropped."""
    return con.execute(f"""
        SELECT Ticker, "{column}" AS label
        FROM companies
        WHERE "{column}" IS NOT NULL AND trim("{column}") != ''
    """).df()


def build_windows(series: pd.DataFrame, *, window_length: int, stride: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if window_length <= 0:
        raise ValueError('window_length must be > 0')
    if stride <= 0:
        raise ValueError('stride must be > 0')

    columns: dict[str, list[np.ndarray]] = {c: [] for c in _CHANNELS}
    meta_records: list[dict] = []

    for ticker, group in series.groupby('Ticker', sort=True):
        group = group.sort_values('Date')
        ohlc = group[['O', 'H', 'L', 'C']].to_numpy(dtype=float)
        volume = group['V'].to_numpy(dtype=float)
        dates = group['Date'].to_numpy()
        n = len(group)
        for offset in range(0, n - window_length + 1, stride):
            window_ohlc = ohlc[offset : offset + window_length]
            window_v = volume[offset : offset + window_length]
            if not (window_ohlc > 0).all():  # rejects non-positive and NaN (NaN > 0 is False)
                continue
            if not (window_v > 0).all():  # a zero (not just negative) breaks ratio/log-based encoders
                continue
            for i, channel in enumerate(('O', 'H', 'L', 'C')):
                columns[channel].append(window_ohlc[:, i])
            columns['V'].append(window_v)
            meta_records.append({
                'Ticker': ticker,
                'start_date': dates[offset],
                'end_date': dates[offset + window_length - 1],
                'start_idx': offset,
            })

    if not meta_records:
        empty = {c: np.empty((window_length, 0), dtype=float) for c in _CHANNELS}
        return empty, pd.DataFrame(columns=_META_COLS)

    channels = {c: np.column_stack(cols) for c, cols in columns.items()}
    return channels, pd.DataFrame(meta_records)


def attach_labels(
    channels: dict[str, np.ndarray], meta: pd.DataFrame, labels: pd.DataFrame
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Join a Ticker -> label frame onto meta/channels, dropping windows whose ticker has no label."""
    keep = meta['Ticker'].isin(labels['Ticker']).to_numpy()
    meta = meta.loc[keep].merge(labels, on='Ticker', how='left').reset_index(drop=True)
    channels = {c: arr[:, keep] for c, arr in channels.items()}
    return channels, meta


def _label_values(
    adj_close: np.ndarray, end_rows: np.ndarray, offsets: np.ndarray, *,
    window_length: int, horizon: int, label: str, exit_halfwidth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized future-performance label per candidate window, plus a keep mask for windows that can be labelled."""
    if label == 'point':
        values = adj_close[end_rows + horizon] / adj_close[end_rows] - 1.0
        return values, np.ones(len(end_rows), dtype=bool)

    if label in ('mean_exit', 'vol_scaled'):
        exit_mean = sliding_window_view(adj_close, 2 * exit_halfwidth + 1)[end_rows + horizon - exit_halfwidth].mean(axis=1)
        mean_exit_return = exit_mean / adj_close[end_rows] - 1.0
        if label == 'mean_exit':
            return mean_exit_return, np.ones(len(end_rows), dtype=bool)
        safe = np.where(adj_close > 0, adj_close, 1.0)  # placeholder rows never fall inside a window that passed the AC check
        log_returns = np.diff(np.log(safe))
        window_returns = sliding_window_view(log_returns, window_length - 1)[offsets]
        mean_return = window_returns.mean(axis=1)
        variance = np.einsum('ij,ij->i', window_returns, window_returns) / (window_length - 1) - mean_return**2
        trailing_vol = np.sqrt(np.clip(variance, 0.0, None))
        keep = trailing_vol > 0.0  # flat window: vol-scaling undefined
        values = np.divide(mean_exit_return, trailing_vol, out=np.zeros_like(mean_exit_return), where=keep)
        return values, keep

    # trend: slope of a line fitted through log(adj_close) over the horizon path, times the fit's R^2
    safe = np.where(adj_close > 0, adj_close, 1.0)  # placeholder rows never fall inside a path that passed the AC check
    path = sliding_window_view(np.log(safe), horizon)[end_rows + 1]
    centered = path - path.mean(axis=1, keepdims=True)
    x = np.arange(horizon, dtype=float) - (horizon - 1) / 2.0
    var_x = float(x @ x) / horizon
    covariance = centered @ x / horizon
    var_y = np.einsum('ij,ij->i', centered, centered) / horizon
    flat = var_y <= 0.0  # flat path: zero trend (also covers horizon == 1, where var_x is 0 as well)
    values = np.where(flat, 0.0, covariance**3 / np.where(flat, 1.0, var_x**2 * var_y))
    return values, np.ones(len(end_rows), dtype=bool)


def build_windows_with_future_return(
    series: pd.DataFrame, *, window_length: int, stride: int, horizon: int, label: str = 'point', exit_halfwidth: int = 2
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if window_length <= 0:
        raise ValueError('window_length must be > 0')
    if stride <= 0:
        raise ValueError('stride must be > 0')
    if horizon <= 0:
        raise ValueError('horizon must be > 0')
    if exit_halfwidth < 0 or exit_halfwidth >= horizon:
        raise ValueError('exit_halfwidth must be in [0, horizon)')
    if label not in ('point', 'mean_exit', 'vol_scaled', 'trend'):
        raise ValueError(f"label must be 'point', 'mean_exit', 'vol_scaled' or 'trend', got {label!r}")

    # one flat (ticker, date)-sorted copy of every column; everything below is vectorized over these arrays
    codes, tickers_sorted = pd.factorize(series['Ticker'], sort=True)
    order = np.lexsort((series['Date'].to_numpy(), codes))
    codes = codes[order]
    dates = series['Date'].to_numpy()[order]
    price_volume = {channel: series[channel].to_numpy(dtype=np.float32)[order] for channel in _CHANNELS}  # halves window memory
    adj_close = series['AC'].to_numpy(dtype=float)[order]  # full-precision split/dividend-adjusted close - labels use this, not raw C
    n = len(codes)

    # [row_start, row_end) = the row span of each row's own ticker
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    span_start = np.concatenate(([0], boundaries))
    span_end = np.concatenate((boundaries, [n]))
    row_start = np.repeat(span_start, span_end - span_start)
    row_end = np.repeat(span_end, span_end - span_start)

    # shared reference dates (not per-ticker offsets) so windows line up across tickers - required
    # to compute a cross-sectional (same-day) group per date later
    all_dates = np.unique(dates)
    reference_dates = all_dates[window_length - 1 : len(all_dates) - horizon : stride]

    # candidate window ends: rows on a reference date whose ticker has room for the window and the label
    end_rows = np.flatnonzero(np.isin(dates, reference_dates))
    label_reach = horizon + exit_halfwidth if label in ('mean_exit', 'vol_scaled') else horizon
    in_span = (end_rows - (window_length - 1) >= row_start[end_rows]) & (end_rows + label_reach < row_end[end_rows])
    end_rows = end_rows[in_span]
    offsets = end_rows - (window_length - 1)

    # validity by exact prefix-sum counts: a row range is usable iff it contains zero flagged rows
    bad_price = np.zeros(n, dtype=bool)
    for column in price_volume.values():
        bad_price |= ~(column > 0)  # flags non-positive and NaN (NaN > 0 is False)
    bad_price_prefix = np.concatenate(([0], np.cumsum(bad_price)))
    bad_ac_prefix = np.concatenate(([0], np.cumsum(~(adj_close > 0))))

    def clean(prefix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """True where the inclusive row range [lo, hi] contains no flagged rows."""
        return prefix[hi + 1] - prefix[lo] == 0

    valid = clean(bad_price_prefix, offsets, end_rows) & clean(bad_ac_prefix, end_rows, end_rows)
    if label == 'point':
        valid &= clean(bad_ac_prefix, end_rows + horizon, end_rows + horizon)
    elif label in ('mean_exit', 'vol_scaled'):
        valid &= clean(bad_ac_prefix, end_rows + horizon - exit_halfwidth, end_rows + horizon + exit_halfwidth)
        if label == 'vol_scaled':
            valid &= clean(bad_ac_prefix, offsets, end_rows)
    else:  # trend
        valid &= clean(bad_ac_prefix, end_rows + 1, end_rows + horizon)
    end_rows, offsets = end_rows[valid], offsets[valid]

    if len(end_rows) == 0:
        empty = {channel: np.empty((window_length, 0), dtype=np.float32) for channel in _CHANNELS}
        return empty, pd.DataFrame(columns=[*_META_COLS, 'future_return'])

    label_values, keep = _label_values(
        adj_close, end_rows, offsets, window_length=window_length, horizon=horizon, label=label, exit_halfwidth=exit_halfwidth
    )
    end_rows, offsets, label_values = end_rows[keep], offsets[keep], label_values[keep]

    channels = {channel: sliding_window_view(column, window_length)[offsets].T for channel, column in price_volume.items()}
    meta = pd.DataFrame({
        'Ticker': np.asarray(tickers_sorted)[codes[end_rows]],
        'start_date': dates[offsets],
        'end_date': dates[end_rows],
        'start_idx': offsets - row_start[end_rows],
        'future_return': label_values,
    })
    return channels, meta


def demean_by_date(meta: pd.DataFrame, value_col: str, date_col: str = 'end_date') -> np.ndarray:
    group_mean = meta.groupby(date_col)[value_col].transform('mean')
    return (meta[value_col] - group_mean).to_numpy()


def _carve(meta: pd.DataFrame, pool: np.ndarray, frac: float, method: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if method == 'recent':
        end_dates = pd.to_datetime(meta['end_date']).to_numpy()
        start_dates = pd.to_datetime(meta['start_date']).to_numpy()
        carved = np.zeros(len(meta), dtype=bool)
        kept = np.zeros(len(meta), dtype=bool)
        for ticker_idx in meta.groupby('Ticker', sort=False).indices.values():  # one grouping, not a per-ticker rescan
            ticker_pool = ticker_idx[pool[ticker_idx]]
            if len(ticker_pool) == 0:
                continue
            cutoff = pd.Series(end_dates[ticker_pool]).quantile(1 - frac)
            carved[ticker_pool[start_dates[ticker_pool] > cutoff]] = True
            kept[ticker_pool[end_dates[ticker_pool] <= cutoff]] = True
    elif method == 'random':
        draw = rng.random(len(meta))
        carved = pool & (draw < frac)
        kept = pool & ~carved
    else:
        raise ValueError(f"method must be 'recent' or 'random', got {method!r}")
    return carved, kept


def _carve_last_n(meta: pd.DataFrame, pool: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    start_dates = pd.to_datetime(meta['start_date']).to_numpy()
    end_dates = pd.to_datetime(meta['end_date']).to_numpy()
    carved = np.zeros(len(meta), dtype=bool)
    kept = np.zeros(len(meta), dtype=bool)
    for ticker_idx in meta.groupby('Ticker', sort=False).indices.values():  # one grouping, not a per-ticker rescan
        ticker_pool = ticker_idx[pool[ticker_idx]]
        if len(ticker_pool) == 0:
            continue
        ordered = ticker_pool[np.argsort(start_dates[ticker_pool])]
        last_n = ordered[-n:]
        carved[last_n] = True
        cutoff = start_dates[last_n].min()  # earliest start_date among the carved windows
        kept[ticker_pool[end_dates[ticker_pool] < cutoff]] = True  # windows overlapping the carved ones are dropped
    return carved, kept


def split_windows(
    meta: pd.DataFrame,
    *,
    valid_frac: float = 0.1,
    valid_method: str = 'recent',
    test_n: int = 5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = np.ones(len(meta), dtype=bool)

    test_mask, pool = _carve_last_n(meta, pool, test_n)  # test is always the last n windows per ticker
    valid_mask, pool = _carve(meta, pool, valid_frac, valid_method, rng)
    train_mask = pool

    # a ticker with zero train windows can never be learned - drop it from all three splits
    tickers_with_train = set(meta.loc[train_mask, 'Ticker'])
    has_train = meta['Ticker'].isin(tickers_with_train).to_numpy()
    return train_mask & has_train, valid_mask & has_train, test_mask & has_train


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def encode_windows(channels: dict[str, np.ndarray], encoder_factory: Callable[[], object]) -> np.ndarray:
    encoded = [encoder_factory().fit(arr).transform(arr) for arr in channels.values()]
    stacked = np.stack(encoded, axis=0)  # (n_channels, window_length, n_windows)
    return stacked.transpose(2, 0, 1)  # (n_windows, n_channels, window_length)


def encode_windows_flat(channels: dict[str, np.ndarray], encoder_factory: Callable[[], object]) -> np.ndarray:
    """Encode each channel straight into its slice of one flat float32 matrix, windows as rows.

    Rows equal `flatten_windows(encode_windows(...))` cast to float32 - the layout tree models
    (xgboost) consume directly, built with one full-size allocation.
    """
    window_length, n_windows = next(iter(channels.values())).shape
    flat = np.empty((n_windows, len(channels) * window_length), dtype=np.float32)
    for i, arr in enumerate(channels.values()):
        encoded = encoder_factory().fit(arr).transform(arr)  # (window_length, n_windows)
        flat[:, i * window_length : (i + 1) * window_length] = encoded.T  # cast to float32 during the write
    return flat


def encode_labels(meta: pd.DataFrame, train_mask: np.ndarray, label_col: str = 'Ticker') -> tuple[np.ndarray, list[str]]:
    """Map `meta[label_col]` to contiguous class indices, with the class set fit only on train_mask rows."""
    classes_sorted = sorted(meta.loc[train_mask, label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    labels = meta[label_col].map(class_to_idx).fillna(-1).astype(np.int64).to_numpy()
    return labels, classes_sorted


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return counts.sum() / (n_classes * counts)
