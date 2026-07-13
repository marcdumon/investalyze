"""Tests for the screener's pure ranking / filtering / scoring logic."""
import numpy as np
import pandas as pd
import pytest

from investalyze.apps.screener.logic import apply_filters, apply_metadata_filters, composite_score, compute_ranks

FRAME = pd.DataFrame({
    'Ticker': ['A', 'B', 'C'],
    'earnings_yield': [0.10, 0.05, np.nan],
    'vol_252': [0.2, 0.4, 0.3],
})
COLS = ['earnings_yield', 'vol_252']


def test_ranks_oriented_and_nan_preserved():
    ranked = compute_ranks(FRAME, COLS)
    assert ranked.loc[0, 'rank_earnings_yield'] == pytest.approx(100.0)
    assert ranked.loc[1, 'rank_earnings_yield'] == pytest.approx(50.0)
    assert pd.isna(ranked.loc[2, 'rank_earnings_yield'])
    # vol is lower-is-better: the lowest vol (A) gets the highest rank
    assert ranked.loc[0, 'rank_vol_252'] == pytest.approx(100.0)
    assert ranked.loc[1, 'rank_vol_252'] == pytest.approx(100.0 / 3)


def test_filters_drop_nan_and_apply_bounds():
    ranked = compute_ranks(FRAME, COLS)
    out = apply_filters(ranked, {'earnings_yield': (60.0, None)})
    assert list(out['Ticker']) == ['A']
    out = apply_filters(ranked, {'earnings_yield': (None, None)})   # inactive filter is a no-op
    assert len(out) == 3


def test_composite_score_skips_nan():
    ranked = compute_ranks(FRAME, COLS)
    score = composite_score(ranked, COLS)
    assert score.iloc[0] == pytest.approx(100.0)
    assert score.iloc[2] == pytest.approx(ranked.loc[2, 'rank_vol_252'])   # only the non-NaN rank counts
    assert composite_score(ranked, []).isna().all()


# ---------- metadata filters ----------

META = pd.DataFrame({
    'Ticker': ['AAA', 'BBB', 'CCC'],
    'name': ['Alpha Corp', 'Beta Inc', 'Gamma Ltd'],
    'sector': ['Technology', 'Healthcare', 'Technology'],
    'industry': ['Software', 'Biotech', 'Hardware'],
    'mcap_bucket': ['large', 'micro', 'mid'],
    'dollar_vol': [50e6, 0.5e6, 5e6],
    'years': [12.0, 1.5, 6.0],
    'active': [True, False, True],
    'n_anomalies': [0, 4, 1],
})


def test_metadata_no_filters_is_noop():
    out = apply_metadata_filters(META, None, None, None, None, None, None, 'all', None)
    assert len(out) == 3


def test_metadata_search_matches_ticker_or_name():
    out = apply_metadata_filters(META, 'beta', None, None, None, None, None, 'all', None)
    assert list(out['Ticker']) == ['BBB']
    out = apply_metadata_filters(META, 'ccc', None, None, None, None, None, 'all', None)
    assert list(out['Ticker']) == ['CCC']


def test_metadata_categorical_filters():
    out = apply_metadata_filters(META, None, ['Technology'], None, None, None, None, 'all', None)
    assert list(out['Ticker']) == ['AAA', 'CCC']
    out = apply_metadata_filters(META, None, None, ['Biotech'], None, None, None, 'all', None)
    assert list(out['Ticker']) == ['BBB']
    out = apply_metadata_filters(META, None, None, None, ['large', 'mid'], None, None, 'all', None)
    assert list(out['Ticker']) == ['AAA', 'CCC']


def test_metadata_numeric_and_listing_filters():
    out = apply_metadata_filters(META, None, None, None, None, 4.0, None, 'all', None)   # min $vol 4mn/day
    assert list(out['Ticker']) == ['AAA', 'CCC']
    out = apply_metadata_filters(META, None, None, None, None, None, 5.0, 'all', None)
    assert list(out['Ticker']) == ['AAA', 'CCC']
    out = apply_metadata_filters(META, None, None, None, None, None, None, 'delisted', None)
    assert list(out['Ticker']) == ['BBB']
    out = apply_metadata_filters(META, None, None, None, None, None, None, 'all', 0)
    assert list(out['Ticker']) == ['AAA']


def test_metadata_filters_combine():
    out = apply_metadata_filters(META, None, ['Technology'], None, None, None, 10.0, 'active', None)
    assert list(out['Ticker']) == ['AAA']
