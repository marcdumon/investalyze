"""Pool assembly for the screener: per-ticker metadata metrics merged with the factor columns.

Built once through a short-lived read-only connection and cached at module level; the page's
Refresh control calls clear_cache() so the next access rebuilds.
"""

from pathlib import Path

import pandas as pd

from investalyze.analysis import factors
from investalyze.apps.screener import metrics
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'

_POOL: pd.DataFrame | None = None


def _merge_pool(metrics_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the factor columns onto the metrics frame; identity columns come from metrics."""
    return metrics_df.merge(factors_df[['Ticker'] + factors.FACTORS], on='Ticker', how='left')


def get_pool() -> pd.DataFrame:
    """Return the cached pool frame, building it once via a short-lived read-only connection."""
    global _POOL
    if _POOL is None:
        con = storage.connect(DATA_ROOT, read_only=True)
        try:
            _POOL = _merge_pool(metrics.build_metrics(con), factors.build_factors(con))
        finally:
            con.close()
    return _POOL


def clear_cache() -> None:
    """Drop the cached pool so the next access rebuilds it."""
    global _POOL
    _POOL = None
