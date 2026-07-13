"""Pure screening logic: cross-sectional percentile ranks, filters and the composite score.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import numpy as np
import pandas as pd

from investalyze.analysis.factors import FACTORS, HIGHER_IS_BETTER


def compute_ranks(df: pd.DataFrame, factor_cols: list[str] = FACTORS) -> pd.DataFrame:
    """Add a rank_<factor> column (0 to 100) per factor, oriented so 100 is always best.

    Ranks are percentile ranks over the rows of `df` (the currently selected pool); NaN factor
    values get NaN ranks.
    """
    ranked = df.copy()
    for factor in factor_cols:
        values = ranked[factor] if HIGHER_IS_BETTER[factor] else -ranked[factor]
        ranked[f'rank_{factor}'] = values.rank(pct=True) * 100
    return ranked


def apply_filters(df: pd.DataFrame, filters: dict[str, tuple[float | None, float | None]]) -> pd.DataFrame:
    """Keep rows whose ranks fall inside every active (min, max) percentile bound.

    A factor with both bounds None is inactive; an active bound excludes NaN ranks.
    """
    mask = pd.Series(True, index=df.index)
    for factor, (lo, hi) in filters.items():
        if lo is None and hi is None:
            continue
        rank = df[f'rank_{factor}']
        mask &= rank.notna()
        if lo is not None:
            mask &= rank >= lo
        if hi is not None:
            mask &= rank <= hi
    return df[mask]


def composite_score(df: pd.DataFrame, selected: list[str]) -> pd.Series:
    """Equal-weight mean of the selected factors' ranks; a ticker's NaN ranks are skipped."""
    if not selected:
        return pd.Series(np.nan, index=df.index)
    return df[[f'rank_{factor}' for factor in selected]].mean(axis=1)


def apply_metadata_filters(
    df: pd.DataFrame, search: str | None, sectors: list[str] | None, industries: list[str] | None,
    buckets: list[str] | None, min_dvol_mn: float | None, min_years: float | None,
    active: str, max_anomalies: int | None
) -> pd.DataFrame:
    """Apply every metadata sidebar filter to the pool; empty/None controls leave their angle unfiltered."""
    mask = pd.Series(True, index=df.index)
    if search:
        needle = search.strip().upper()
        mask &= df['Ticker'].str.upper().str.contains(needle, regex=False) | df['name'].str.upper().str.contains(needle, regex=False)
    if sectors:
        mask &= df['sector'].isin(sectors)
    if industries:
        mask &= df['industry'].isin(industries)
    if buckets:
        mask &= df['mcap_bucket'].isin(buckets)
    if min_dvol_mn is not None:
        mask &= df['dollar_vol'] >= min_dvol_mn * 1e6
    if min_years is not None:
        mask &= df['years'] >= min_years
    if active == 'active':
        mask &= df['active']
    elif active == 'delisted':
        mask &= ~df['active']
    if max_anomalies is not None:
        mask &= df['n_anomalies'] <= max_anomalies
    return df[mask]
