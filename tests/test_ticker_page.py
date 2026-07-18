"""Tests for the ticker page's pure shaping logic (peer group, returns, TTM history)."""
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis.factors import FACTORS
from investalyze.apps.ticker.data import (
    MAX_PEERS, drawdown, peer_group, peer_percentiles, rebased, trailing_returns, ttm_history,
)

POOL = pd.DataFrame({
    'Ticker': ['ME', 'A', 'B', 'C', 'D', 'E', 'F', 'S1', 'S2'],
    'industry': ['Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Hard', 'Hard'],
    'sector': ['Tech'] * 9,
    'mcap': [10e9, 9e9, 11e9, 1e9, 100e9, 12e9, np.nan, 10e9, 8e9],
})


def test_peer_group_same_industry_ticker_first():
    group = peer_group(POOL, 'ME')
    assert group.iloc[0]['Ticker'] == 'ME'
    # F has no market cap, so it cannot be judged size-comparable and is excluded
    assert set(group['Ticker']) == {'ME', 'A', 'B', 'C', 'D', 'E'}


def test_peer_group_excludes_size_mismatch():
    pool = pd.DataFrame({
        'Ticker': ['ME', 'TINY', 'OK1', 'OK2', 'OK3', 'OK4', 'OK5'],
        'industry': ['Soft'] * 7, 'sector': ['Tech'] * 7,
        'mcap': [1e12, 2e6, 1e11, 2e11, 5e11, 2e12, 5e12],   # TINY is 500000x smaller
    })
    group = peer_group(pool, 'ME')
    assert 'TINY' not in set(group['Ticker'])
    assert set(group['Ticker']) == {'ME', 'OK1', 'OK2', 'OK3', 'OK4', 'OK5'}


def test_peer_group_no_own_mcap_falls_back_to_industry():
    pool = POOL.copy()
    pool.loc[pool['Ticker'] == 'ME', 'mcap'] = np.nan
    group = peer_group(pool, 'ME')
    assert group.iloc[0]['Ticker'] == 'ME'
    assert set(group['Ticker']) == {'ME', 'A', 'B', 'C', 'D', 'E', 'F'}


def test_peer_group_widens_to_sector_when_thin():
    pool = POOL[POOL['Ticker'].isin(['ME', 'A', 'B', 'S1', 'S2'])]
    group = peer_group(pool, 'ME')   # only 2 industry mates -> widen to sector
    assert set(group['Ticker']) == {'ME', 'A', 'B', 'S1', 'S2'}


def test_peer_group_caps_by_mcap_proximity():
    big = pd.DataFrame({
        'Ticker': ['ME'] + [f'P{i}' for i in range(30)],
        'industry': ['Soft'] * 31, 'sector': ['Tech'] * 31,
        'mcap': [10e9] + [10e9 * (1.1 ** i) for i in range(30)],
    })
    group = peer_group(big, 'ME')
    assert len(group) == MAX_PEERS + 1
    assert group.iloc[0]['Ticker'] == 'ME'
    assert 'P29' not in set(group['Ticker'])   # farthest by log-mcap is dropped


def test_peer_group_unknown_ticker_empty():
    assert peer_group(POOL, 'NOPE').empty


def test_drawdown():
    dd = drawdown(pd.Series([100.0, 110.0, 99.0, 110.0]))
    assert dd.tolist() == pytest.approx([0.0, 0.0, -0.1, 0.0])


def test_rebased_window_starts_at_100():
    history = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=6),
                            'AC': [10, 20, 40, 80, 160, 320], 'market': [1, 2, 4, 8, 16, 32]})
    out = rebased(history, 3)
    assert len(out) == 3
    assert out.iloc[0]['AC'] == pytest.approx(100.0)
    assert out.iloc[-1]['AC'] == pytest.approx(400.0)
    assert len(rebased(history, None)) == 6


def test_trailing_returns():
    n = 300
    history = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n),
                            'AC': np.linspace(100, 200, n), 'market': np.full(n, 50.0)})
    out = trailing_returns(history)
    row_1y = out[out['window'] == '1y'].iloc[0]
    assert row_1y['ticker'] == pytest.approx(history['AC'].iloc[-1] / history['AC'].iloc[-253] - 1)
    assert row_1y['market'] == pytest.approx(0.0)
    assert np.isnan(out[out['window'] == '3y'].iloc[0]['ticker'])   # not enough history


def test_ttm_history_sums_margins_and_per_share():
    quarters = pd.DataFrame({
        'Report Date': pd.date_range('2020-03-31', periods=5, freq='QE'),
        'Revenue': [100.0] * 5, 'Gross Profit': [50.0] * 5, 'Operating Income (Loss)': [20.0] * 5,
        'Net Income (Common)': [10.0] * 5, 'Shares (Diluted)': [8.0] * 5,
        'Net Cash from Operating Activities': [15.0] * 5, 'Change in Fixed Assets & Intangibles': [-5.0] * 5,
        'Total Equity': [200.0] * 5, 'Short Term Debt': [10.0] * 5, 'Long Term Debt': [np.nan] * 5,
        'Cash, Cash Equivalents & Short Term Investments': [30.0] * 5,
    })
    out = ttm_history(quarters)
    assert len(out) == 2   # first 3 quarters lack a full TTM window
    last = out.iloc[-1]
    assert last['revenue'] == pytest.approx(400.0)
    assert last['net_margin'] == pytest.approx(0.1)
    assert last['eps'] == pytest.approx(40.0 / 8.0)
    assert last['fcf'] == pytest.approx(40.0)          # cfo 60 + capex -20
    assert last['debt'] == pytest.approx(10.0)          # NaN long-term debt counts as 0
    assert last['equity'] == pytest.approx(200.0)       # balance items stay point-in-time
    assert last['cash'] == pytest.approx(30.0)


def test_peer_percentiles_orientation_and_families():
    peers = pd.DataFrame({'Ticker': ['ME', 'X', 'Y']})
    for factor in FACTORS:
        peers[factor] = [0.1, 0.2, 0.3]
    families, ranks = peer_percentiles(peers, 'ME')
    assert ranks['earnings_yield'] == pytest.approx(100.0 / 3)   # higher is better, ME lowest
    assert ranks['vol_252'] == pytest.approx(100.0)              # lower is better, ME lowest
    assert families['momentum'] == pytest.approx(100.0 / 3)
