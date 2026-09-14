"""Pure screening logic: cross-sectional percentile ranks, raw-value filters and the composite score.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from investalyze.analysis.factors import FACTORS, HIGHER_IS_BETTER


def compute_ranks(df: pd.DataFrame, factor_cols: list[str] = FACTORS,
                  higher_is_better: dict[str, bool] | None = None) -> pd.DataFrame:
    """Add a rank_<factor> column (0 to 100) per factor, oriented so 100 is always best.

    Ranks are percentile ranks over the rows of `df` (the currently selected pool); NaN factor
    values get NaN ranks. `higher_is_better` gives each factor's direction, defaulting to the built-in
    factors'.
    """
    orientation = HIGHER_IS_BETTER if higher_is_better is None else higher_is_better
    ranked = df.copy()
    for factor in factor_cols:
        values = ranked[factor] if orientation[factor] else -ranked[factor]
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


class FactorScale(NamedTuple):
    """How one factor is presented: the increment of its bound spinners and the decimals it displays."""

    step: float
    decimals: int


def factor_scale(values: pd.Series) -> FactorScale:
    """Spinner step and display precision for a factor, both sized to a twentieth of its IQR.

    Factors differ in scale by orders of magnitude, so shared constants are either uselessly fine at
    one end of the list or uselessly coarse at the other. The step is the power of ten at or below
    that twentieth; the precision carries one digit further, so two adjacent steps still read apart
    and a small non-zero value such as a 0.002 momentum keeps its magnitude. Falls back to a step of
    0.01 shown at 2 decimals when the spread is unusable (all-NaN, constant, or a single value).
    """
    spread = values.quantile(0.75) - values.quantile(0.25)
    if pd.isna(spread) or spread <= 0:
        return FactorScale(0.01, 2)
    exponent = math.floor(math.log10(spread / 20))
    return FactorScale(float(10 ** exponent), min(6, max(0, 1 - exponent)))


def collect_bounds(inputs_list: list) -> dict[str, tuple[float | None, float | None]]:
    """Pair each factor's min and max bound out of a callback's pattern-matching input groups.

    Reads the factor off each input's own id dict rather than trusting argument order, so inserting
    an Input above the factor filters leaves their binding intact. A blank input reads as no bound.

    Args:
        inputs_list: Dash's `ctx.inputs_list`, whose pattern-matching entries are lists of
            `{'id': {'type': 'fmin'|'fmax', 'factor': ...}, 'value': ...}` specs.
    """
    bounds: dict[str, list[float | None]] = {}
    for group in inputs_list:
        if not isinstance(group, list):
            continue
        for spec in group:
            identifier = spec.get('id', {}) if isinstance(spec, dict) else {}
            if identifier.get('type') not in ('fmin', 'fmax'):
                continue
            value = spec.get('value')
            slot = 0 if identifier['type'] == 'fmin' else 1
            bounds.setdefault(identifier['factor'], [None, None])[slot] = value if value not in (None, '') else None
    return {factor: (lo, hi) for factor, (lo, hi) in bounds.items()}


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
