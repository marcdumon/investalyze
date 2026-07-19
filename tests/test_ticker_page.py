"""Tests for the ticker page's pure shaping logic (peer group, returns, TTM history)."""
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis.factors import FACTORS
from investalyze.apps.ticker.data import (
    MAX_UNIVERSE_PEERS, drawdown, peer_percentiles, rebased, scope_peer_group, trailing_returns, ttm_history,
    universe_peer_group,
)

POOL = pd.DataFrame({
    'Ticker': ['ME', 'A', 'B', 'C', 'D', 'E', 'F', 'S1', 'S2'],
    'industry': ['Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Soft', 'Hard', 'Hard'],
    'sector': ['Tech'] * 9,
    'mcap': [10e9, 9e9, 11e9, 1e9, 100e9, 12e9, np.nan, 10e9, 8e9],
})


def test_scope_peer_group_industry_ticker_first():
    group = scope_peer_group(POOL, 'ME', 'industry')
    assert group.iloc[0]['Ticker'] == 'ME'
    assert set(group['Ticker']) == {'ME', 'A', 'B', 'C', 'D', 'E', 'F'}


def test_scope_peer_group_sector_takes_whole_sector():
    group = scope_peer_group(POOL, 'ME', 'sector')
    assert set(group['Ticker']) == set(POOL['Ticker'])


def test_scope_peer_group_unknown_value_ticker_alone():
    pool = POOL.copy()
    pool.loc[pool['Ticker'] == 'ME', 'industry'] = 'unknown'
    group = scope_peer_group(pool, 'ME', 'industry')
    assert group['Ticker'].tolist() == ['ME']


def test_scope_peer_group_caps_by_mcap_proximity():
    n = MAX_UNIVERSE_PEERS + 20
    pool = pd.DataFrame({
        'Ticker': ['ME'] + [f'P{i}' for i in range(n)],
        'industry': ['Soft'] * (n + 1), 'sector': ['Tech'] * (n + 1),
        'mcap': [10e9] + [10e9 * (1.1 ** i) for i in range(n)],
    })
    group = scope_peer_group(pool, 'ME', 'industry')
    assert len(group) == MAX_UNIVERSE_PEERS + 1
    assert f'P{n - 1}' not in set(group['Ticker'])   # farthest by log-mcap is dropped
    assert len(scope_peer_group(pool, 'ME', 'industry', cap=None)) == n + 1


def test_scope_peer_group_unknown_ticker_empty():
    assert scope_peer_group(POOL, 'NOPE', 'industry').empty


def test_universe_peer_group_ticker_first_and_deduped():
    group = universe_peer_group(POOL, 'ME', ['ME', 'A', 'S1'])
    assert group.iloc[0]['Ticker'] == 'ME'
    assert group['Ticker'].tolist().count('ME') == 1
    assert set(group['Ticker']) == {'ME', 'A', 'S1'}


def test_universe_peer_group_keeps_members_found_in_pool():
    group = universe_peer_group(POOL, 'ME', ['A', 'GONE', 'S2'])
    # no size filter or cap: the list is taken as curated
    assert set(group['Ticker']) == {'ME', 'A', 'S2'}


def test_universe_peer_group_unknown_ticker_empty():
    assert universe_peer_group(POOL, 'NOPE', ['A', 'B']).empty


def test_universe_peer_group_caps_by_mcap_proximity():
    n = MAX_UNIVERSE_PEERS + 20
    pool = pd.DataFrame({
        'Ticker': ['ME'] + [f'P{i}' for i in range(n)],
        'industry': ['Soft'] * (n + 1), 'sector': ['Tech'] * (n + 1),
        'mcap': [10e9] + [10e9 * (1.1 ** i) for i in range(n)],
    })
    group = universe_peer_group(pool, 'ME', pool['Ticker'].tolist())
    assert len(group) == MAX_UNIVERSE_PEERS + 1
    assert group.iloc[0]['Ticker'] == 'ME'
    assert f'P{n - 1}' not in set(group['Ticker'])   # farthest by log-mcap is dropped
    assert len(universe_peer_group(pool, 'ME', pool['Ticker'].tolist(), cap=None)) == n + 1


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
