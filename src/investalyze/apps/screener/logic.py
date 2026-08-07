"""Pure screening logic: cross-sectional percentile ranks, raw-value filters and the composite score.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import math

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
    """Keep rows whose raw factor values fall inside every active (min, max) bound.

    Bounds are in each factor's own units, so they read the same as the Raw display mode.
    A factor with both bounds None is inactive; an active bound excludes NaN values.
    """
    mask = pd.Series(True, index=df.index)
    for factor, (lo, hi) in filters.items():
        if lo is None and hi is None:
            continue
        values = df[factor]
        mask &= values.notna()
        if lo is not None:
            mask &= values >= lo
        if hi is not None:
            mask &= values <= hi
    return df[mask]


def spin_step(values: pd.Series) -> float:
    """Step for a factor's min/max spinner: the power of ten nearest a twentieth of its IQR.

    Factors differ in scale by orders of magnitude, so a shared constant is either uselessly fine
    for one end of the list or uselessly coarse for the other. Falls back to 0.01 when the spread
    is unusable (all-NaN, constant, or a single value).
    """
    spread = values.quantile(0.75) - values.quantile(0.25)
    if pd.isna(spread) or spread <= 0:
        return 0.01
    return float(10 ** math.floor(math.log10(spread / 20)))


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
