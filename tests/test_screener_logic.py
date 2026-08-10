"""Tests for the screener's pure ranking / filtering / scoring logic."""
import numpy as np
import pandas as pd
import pytest

from investalyze.apps.screener.logic import (apply_filters, apply_metadata_filters, collect_bounds, composite_score,
                                             compute_ranks, factor_scale)

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
    out = apply_filters(ranked, {'earnings_yield': (0.08, None)})
    assert list(out['Ticker']) == ['A']
    out = apply_filters(ranked, {'earnings_yield': (None, None)})   # inactive filter is a no-op
    assert len(out) == 3


def test_filters_bound_raw_values_not_ranks():
    ranked = compute_ranks(FRAME, COLS)
    assert list(apply_filters(ranked, {'earnings_yield': (0.05, 0.10)})['Ticker']) == ['A', 'B']
    assert list(apply_filters(ranked, {'earnings_yield': (None, 0.05)})['Ticker']) == ['B']
    # 60 is a plausible percentile bound but no raw value reaches it, so nothing survives
    assert apply_filters(ranked, {'earnings_yield': (60.0, None)}).empty


def test_filters_accept_negative_bounds():
    frame = pd.DataFrame({'Ticker': ['A', 'B', 'C'], 'earnings_yield': [-3.1, -0.5, 0.04]})
    ranked = compute_ranks(frame, ['earnings_yield'])
    assert list(apply_filters(ranked, {'earnings_yield': (-1.0, None)})['Ticker']) == ['B', 'C']
    assert list(apply_filters(ranked, {'earnings_yield': (None, -1.0)})['Ticker']) == ['A']


def test_factor_scale_steps_and_precision_follow_the_spread():
    assert factor_scale(pd.Series([0.0, 0.1326])) == (pytest.approx(0.001), 4)
    assert factor_scale(pd.Series([0.0, 0.5566])) == (pytest.approx(0.01), 3)
    assert factor_scale(pd.Series([0.0, 41.2041])) == (pytest.approx(1.0), 1)


def test_factor_scale_precision_outlives_one_step():
    # a value smaller than the step still shows its magnitude rather than rounding to zero
    scale = factor_scale(pd.Series([0.0, 41.2041]))
    assert round(1.279542, scale.decimals) == pytest.approx(1.3)


def test_factor_scale_falls_back_when_spread_is_unusable():
    assert factor_scale(pd.Series([2.0, 2.0, 2.0])) == (pytest.approx(0.01), 2)
    assert factor_scale(pd.Series([np.nan, np.nan])) == (pytest.approx(0.01), 2)
    assert factor_scale(pd.Series([], dtype=float)) == (pytest.approx(0.01), 2)


# ---------- bound collection from pattern-matching inputs ----------

def _spec(kind: str, factor: str, value):
    """One entry as Dash puts it in ctx.inputs_list for a pattern-matching input."""
    return {'id': {'type': kind, 'factor': factor}, 'property': 'value', 'value': value}


def test_collect_bounds_pairs_min_and_max_by_factor():
    inputs = [
        'a-scalar-input',
        [_spec('fmin', 'roe', 0.1), _spec('fmin', 'vol_252', None)],
        [_spec('fmax', 'roe', None), _spec('fmax', 'vol_252', 0.4)],
    ]
    assert collect_bounds(inputs) == {'roe': (0.1, None), 'vol_252': (None, 0.4)}


def test_collect_bounds_ignores_input_order_and_blanks():
    # the fmax group arrives first and a cleared NumberInput reports '' rather than None
    inputs = [[_spec('fmax', 'roe', 2.0)], [_spec('fmin', 'roe', '')], ['unrelated-group']]
    assert collect_bounds(inputs) == {'roe': (None, 2.0)}


def test_collect_bounds_is_empty_without_factor_inputs():
    assert collect_bounds(['search-value', None, [{'id': {'type': 'other'}, 'value': 3}]]) == {}


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
