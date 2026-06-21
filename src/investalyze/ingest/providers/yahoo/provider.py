"""Yahoo provider — stock prices (raw OHLCV + adjusted close), dividends, splits.

Owns its whole flow: read the ticker universe -> fetch via yfinance -> transform ->
compute adjusted close -> sanity-check vs Yahoo -> save through storage.write. The
network fetch is the only side effect. Split into more files in this folder if it grows.
"""

import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

from investalyze.ingest import storage

log = logging.getLogger('investalyze.ingest.yahoo')

_PRICES, _DIVS, _SPLITS = 'prices', 'dividends', 'splits'
_KEY = ['Ticker', 'Date']


def _to_prices(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker yahoo frame -> canonical raw price rows (no adjusted close yet)."""
    return pd.DataFrame({
        'Ticker': ticker,
        'Date': pd.DatetimeIndex(frame.index).date,
        'O': frame['Open'].astype(float),
        'H': frame['High'].astype(float),
        'L': frame['Low'].astype(float),
        'C': frame['Close'].astype(float),
        'V': frame['Volume'].astype('int64'),
    }).reset_index(drop=True)


def _to_dividends(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Ex-date cash dividends (rows where Dividends > 0)."""
    events = frame[frame['Dividends'] > 0]
    return pd.DataFrame({
        'Ticker': ticker,
        'Date': pd.DatetimeIndex(events.index).date,
        'Dividend': events['Dividends'].astype(float),
    }).reset_index(drop=True)


def _to_splits(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Split events (rows where Stock Splits > 0)."""
    events = frame[frame['Stock Splits'] > 0]
    return pd.DataFrame({
        'Ticker': ticker,
        'Date': pd.DatetimeIndex(events.index).date,
        'Ratio': events['Stock Splits'].astype(float),
    }).reset_index(drop=True)


def _calc_adjusted_close(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """Back-adjusted close from raw close + dividends (total-return method).

    Dividends ONLY: yfinance's Close is already split-adjusted, so splits must not
    be re-applied here (doing so double-counts them — e.g. EZGO's three reverse
    splits gave a 150000x error). Both inputs share one ascending Date index;
    `dividends` is 0 where there is no ex-date event.
    """
    close = close.sort_index()
    dividends = dividends.reindex(close.index).fillna(0.0)

    prev_close = close.shift(1)
    factor = pd.Series(1.0, index=close.index)
    div_days = dividends > 0
    factor[div_days] = 1 - dividends[div_days] / prev_close[div_days]

    # product of factors strictly after each date
    incl = factor[::-1].cumprod()[::-1]  # product of factor[i:]
    after = incl.shift(-1).fillna(1.0)  # product of factor[i+1:]
    return close * after


def _calc_ac_max_diff(derived: pd.Series, yahoo: pd.Series) -> float:
    """Max relative difference between our AC and Yahoo's, over comparable dates."""
    pair = pd.DataFrame({'d': derived, 'y': yahoo}).dropna()
    pair = pair[pair['y'] > 0]
    if pair.empty:
        return 0.0
    return float(((pair['d'] - pair['y']).abs() / pair['y']).max())


def _chunk(items: list[str], size: int) -> list[list[str]]:
    """Split a list into chunks of at most `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fetch(symbols: list[str], *, start: str | None) -> dict[str, pd.DataFrame]:
    """Download one batch via yfinance -> per-ticker frames (empty frame if Yahoo returned nothing).

    `start` set -> incremental from that date (yfinance ignores period); otherwise full history.
    """
    raw = yf.download(symbols, auto_adjust=False, actions=True, group_by='ticker', progress=False, period='max', start=start)
    if raw is None or raw.empty:
        return {}
    return {sym: pd.DataFrame(raw[sym]).dropna(how='all') if sym in raw.columns.get_level_values(0) else pd.DataFrame()
            for sym in symbols}


def _load_existing_tickers(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Distinct tickers already loaded in `table` (empty set if the table is absent)."""
    tables = {t for (t,) in con.execute('SHOW TABLES').fetchall()}
    if table not in tables:
        return set()
    return {t for (t,) in con.execute(f'SELECT DISTINCT Ticker FROM {table}').fetchall()}


def _load_last_dates(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Each ticker's latest stored `prices` date (ISO string); empty if the table is absent."""
    if _PRICES not in {t for (t,) in con.execute('SHOW TABLES').fetchall()}:
        return {}
    return {t: str(d) for t, d in con.execute('SELECT Ticker, MAX(Date) FROM prices GROUP BY Ticker').fetchall()}


def _load_events(con: duckdb.DuckDBPyConnection, table: str, value_col: str, ticker: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Stored events for `ticker` from `table`, reindexed to `idx` with 0.0 where absent."""
    if table not in {t for (t,) in con.execute('SHOW TABLES').fetchall()}:
        return pd.Series(0.0, index=idx)
    rows = con.execute(f'SELECT Date, {value_col} FROM {table} WHERE Ticker = ?', [ticker]).df()
    return pd.Series(rows[value_col].to_numpy(), index=pd.DatetimeIndex(rows['Date'])).reindex(idx).fillna(0.0)


def _recompute_ac(con: duckdb.DuckDBPyConnection, ticker: str) -> None:
    """Re-derive AC for all stored rows of `ticker` from full stored close + events; upsert."""
    px = con.execute('SELECT Date, Ticker, O, H, L, C, V FROM prices WHERE Ticker = ? ORDER BY Date', [ticker]).df()
    if px.empty:
        return
    idx = pd.DatetimeIndex(px['Date'])
    div_s = _load_events(con, _DIVS, 'Dividend', ticker, idx)
    ac = _calc_adjusted_close(pd.Series(px['C'].to_numpy(), index=idx), div_s)
    px['AC'] = ac.to_numpy()
    storage.write(con, _PRICES, px[['Ticker', 'Date', 'O', 'H', 'L', 'C', 'V', 'AC']], key=_KEY)


def run(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict, *, update: bool = False) -> int:
    """Load Yahoo stock prices/dividends/splits into the DB. Returns the prices row count.

    `settings` is the provider's `[yahoo]` config (no fallback defaults — a missing key raises).
    """
    raw_dir = data_root / 'yahoo' / 'raw'
    state_dir = data_root / 'yahoo' / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)

    symbols = pd.read_csv(raw_dir / settings['ticker_file'])['ticker'].tolist()
    empty_file = state_dir / 'empty.csv'
    empty = set(pd.read_csv(empty_file)['ticker']) if empty_file.exists() else set()
    done = _load_existing_tickers(con, _PRICES) if not update else set()
    todo = [s for s in symbols if s not in empty and s not in done]

    newly_empty: list[str] = []
    flagged: list[dict] = []
    last_dates = _load_last_dates(con) if update else {}
    batches = _chunk(todo, settings['batch_size'])
    # fetch + save one batch at a time so progress commits as we go (an interrupted run resumes).
    for i, batch in enumerate(batches):
        if update:
            known = [last_dates[s] for s in batch if s in last_dates]
            # earliest start across the batch (full history if any ticker is new); overlap on tickers
            # with later starts re-fetches a few rows -> idempotent via merge upsert.
            start = None if len(known) < len(batch) else (pd.Timestamp(min(known)) + pd.Timedelta(days=1)).date().isoformat()
        else:
            start = None
        frames = _fetch(batch, start=start)
        empty_before = len(newly_empty)
        batch_prices: list[pd.DataFrame] = []
        batch_divs: list[pd.DataFrame] = []
        batch_splits: list[pd.DataFrame] = []
        recompute: list[str] = []
        for sym in batch:
            frame = frames.get(sym, pd.DataFrame())
            prepared = _prepare_ticker(sym, frame, ac_tolerance=settings['ac_tolerance'], newly_empty=newly_empty, flagged=flagged)
            if prepared is None:
                continue
            prices, divs, splits = prepared
            batch_prices.append(prices)
            if not divs.empty:
                batch_divs.append(divs)
            if not splits.empty:
                batch_splits.append(splits)
            if update and (frame['Dividends'] > 0).any():
                recompute.append(sym)
        # one merge per table per batch (vs one per ticker) — the merge scans the growing
        # target once instead of len(batch) times, which dominated wall time.
        if batch_prices:
            storage.write(con, _PRICES, pd.concat(batch_prices, ignore_index=True), key=_KEY)
        if batch_divs:
            storage.write(con, _DIVS, pd.concat(batch_divs, ignore_index=True), key=_KEY)
        if batch_splits:
            storage.write(con, _SPLITS, pd.concat(batch_splits, ignore_index=True), key=_KEY)
        for sym in recompute:   # after the batch write so the new rows are present
            _recompute_ac(con, sym)
        if newly_empty:
            pd.DataFrame({'ticker': sorted(empty | set(newly_empty))}).to_csv(empty_file, index=False)
        if flagged:
            pd.DataFrame(flagged).to_csv(state_dir / 'ac_discrepancies.csv', index=False)
        batch_empty = newly_empty[empty_before:]
        log.info(f'batch {i + 1}/{len(batches)} saved {len(batch) - len(batch_empty)} empty {len(batch_empty)} (n={len(todo)})')
        if batch_empty:
            log.warning(f'no data for {", ".join(batch_empty)} — marked empty')
        if settings['sleep'] and i < len(batches) - 1:
            time.sleep(settings['sleep'])

    log.info(f'done — {len(todo) - len(newly_empty)} saved, {len(newly_empty)} empty, {len(flagged)} AC-flagged')
    tables = {t for (t,) in con.execute('SHOW TABLES').fetchall()}
    row = con.execute(f'SELECT COUNT(*) FROM {_PRICES}').fetchone() if _PRICES in tables else None
    return int(row[0]) if row is not None else 0


def _prepare_ticker(sym: str, frame: pd.DataFrame, *, ac_tolerance: float, newly_empty: list[str], flagged: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Transform one ticker's fetched frame into (prices, dividends, splits) rows.

    Returns None (and marks the ticker empty) when Yahoo gave no data. The caller
    accumulates the frames and writes them per batch. Pure apart from appending to
    `newly_empty` / `flagged`.
    """
    if frame.empty:
        newly_empty.append(sym)
        log.debug(f'{sym} no data — empty')
        return None
    prices = _to_prices(sym, frame)
    divs = _to_dividends(sym, frame)
    splits = _to_splits(sym, frame)
    ac = _calc_adjusted_close(frame['Close'], frame['Dividends'])
    prices['AC'] = ac.to_numpy()
    diff = _calc_ac_max_diff(ac, frame['Adj Close'])
    if diff > ac_tolerance:
        flagged.append({'ticker': sym, 'max_rel_diff': diff})
    log.debug(f'{sym} prepared ({len(prices)} rows)')
    return prices, divs, splits
