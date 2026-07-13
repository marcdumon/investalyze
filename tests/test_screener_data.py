"""Tests for the screener pool assembly (metrics metadata + factor columns)."""
import numpy as np
import pandas as pd

from investalyze.analysis.factors import FACTORS
from investalyze.apps.screener.data import _merge_pool


def _metrics_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'Ticker': ['AAA', 'BBB'],
        'name': ['Alpha Corp', 'Beta Inc'],
        'sector': ['Technology', 'unknown'],
        'mcap_bn': [1.5, np.nan],
    })


def _factors_frame() -> pd.DataFrame:
    df = pd.DataFrame({'Ticker': ['AAA'], 'name': ['SHOULD NOT LEAK'], 'sector': ['SHOULD NOT LEAK']})
    for factor in FACTORS:
        df[factor] = 0.5
    return df


def test_merge_keeps_metrics_columns_and_adds_factors():
    pool = _merge_pool(_metrics_frame(), _factors_frame())
    assert list(pool['Ticker']) == ['AAA', 'BBB']   # left join: every metrics row survives
    assert pool.loc[0, 'name'] == 'Alpha Corp'      # metadata wins: overlapping identity columns come from metrics
    assert 'name_x' not in pool.columns and 'name_y' not in pool.columns
    for factor in FACTORS:
        assert pool.loc[0, factor] == 0.5
        assert pd.isna(pool.loc[1, factor])          # BBB has no factor row: factors are NaN
