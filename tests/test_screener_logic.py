"""Tests for the screener's pure ranking / filtering / scoring logic."""
import numpy as np
import pandas as pd
import pytest

from investalyze.apps.screener.logic import apply_filters, composite_score, compute_ranks

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
