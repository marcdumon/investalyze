"""Pool assembly for the screener: per-ticker metadata metrics merged with the factor columns.

Factor columns come from two places: the built-in snapshot factors in `analysis.factors`, and the
factor library under `factors/`, computed at the latest trading session. A library factor whose name
matches a built-in one is not given a column, since the pool is shared with the Stock page, which reads
the built-in values under those names; `shadowed_factors` says which ones are left out.

Built once through a short-lived read-only connection and cached at module level, alongside the
per-factor display scales derived from it; the page's Refresh control calls clear_cache() so the next
access rebuilds both.
"""

import logging

import duckdb
import pandas as pd

from investalyze import paths
from investalyze.analysis import factors as builtin
from investalyze.analysis import panels
from investalyze.apps.screener import metrics
from investalyze.apps.screener.logic import FactorScale, factor_scale
from investalyze.ingest import storage
from investalyze.research import compute
from investalyze.research import factors as factor_library

log = logging.getLogger(__name__)

LIBRARY_FAMILY = 'factor library'

_POOL: pd.DataFrame | None = None
_SCALES: dict[str, FactorScale] | None = None
_LIBRARY: dict | None = None


def library() -> dict:
    """The factor library factors the Screener shows: every stored factor not named like a built-in one."""
    global _LIBRARY
    if _LIBRARY is None:
        stored = factor_library.load(paths.FACTORS_DIR)
        _LIBRARY = {name: factor for name, factor in stored.items() if name not in builtin.FACTORS}
    return _LIBRARY


def shadowed_factors() -> list[str]:
    """Factor library names left out of the Screener because a built-in factor has the same name."""
    return sorted(name for name in factor_library.load(paths.FACTORS_DIR) if name in builtin.FACTORS)


def factor_names() -> list[str]:
    """Every factor column in the pool: built-in first, then the factor library."""
    return list(builtin.FACTORS) + sorted(library())


def families() -> dict[str, list[str]]:
    """Factor columns grouped for the filter panel, with the factor library as its own group."""
    grouped = dict(builtin.FAMILIES)
    if library():
        grouped[LIBRARY_FAMILY] = sorted(library())
    return grouped


def higher_is_better() -> dict[str, bool]:
    """Direction of every factor column."""
    return dict(builtin.HIGHER_IS_BETTER) | {name: factor.higher_is_better for name, factor in library().items()}


def factor_docs() -> dict[str, builtin.FactorDoc]:
    """Definition and formula of every factor column."""
    docs = dict(builtin.FACTOR_DOCS)
    for name, factor in library().items():
        docs[name] = builtin.FactorDoc(what=factor.description or 'Defined in the factor library.',
                                       formula=factor.expression)
    return docs


def _library_values(con: duckdb.DuckDBPyConnection, stored: dict) -> pd.DataFrame:
    """Factor library values for every ticker on the latest trading session."""
    names = sorted(stored)
    if not names:
        return pd.DataFrame(columns=['Ticker'])
    last = storage.scalar(con, 'SELECT max(Date) FROM prices')
    calendar = panels.trading_calendar(con, start=(pd.Timestamp(last) - pd.Timedelta(days=20)).date().isoformat())
    try:
        frame = compute.compute(con, names, factor_library.load(paths.FACTORS_DIR), as_of=pd.DatetimeIndex([calendar.max()]))
    except ValueError as error:
        log.warning(f'factor library left out of the screener: {error}')
        return pd.DataFrame(columns=['Ticker'])
    present = [name for name in names if name in frame.columns]
    return frame[['Ticker'] + present]


def _merge_pool(metrics_df: pd.DataFrame, factors_df: pd.DataFrame, library_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the built-in and factor library columns onto the metrics frame."""
    pool = metrics_df.merge(factors_df[['Ticker'] + list(builtin.FACTORS)], on='Ticker', how='left')
    pool = pool.merge(library_df, on='Ticker', how='left')
    for name in library():
        if name not in pool.columns:
            pool[name] = float('nan')
    return pool


def get_pool() -> pd.DataFrame:
    """Return the cached pool frame, building it once via a short-lived read-only connection."""
    global _POOL
    if _POOL is None:
        with storage.connect(read_only=True) as con:
            _POOL = _merge_pool(metrics.build_metrics(con), builtin.build_factors(con), _library_values(con, library()))
    return _POOL


def get_scales() -> dict[str, FactorScale]:
    """Return the cached per-factor spinner step and display precision, derived once from the pool.

    Every quantile pair behind them scans the whole pool, which is wasted work on each page visit
    and each grid rebuild when the pool itself has not changed.
    """
    global _SCALES
    if _SCALES is None:
        pool = get_pool()
        _SCALES = {factor: factor_scale(pool[factor]) for factor in factor_names()}
    return _SCALES


def clear_cache() -> None:
    """Drop the cached pool, factor scales and factor library so the next access rebuilds them."""
    global _POOL, _SCALES, _LIBRARY
    _POOL = None
    _SCALES = None
    _LIBRARY = None
