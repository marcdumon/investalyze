"""Tests for the screener pool assembly: metrics metadata, built-in factor columns and factor library columns."""

import numpy as np
import pandas as pd
import pytest

from investalyze.analysis.factors import FACTORS
from investalyze.apps.screener import data
from investalyze.research import factors as factor_library


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


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test starts and ends with nothing cached, so no test sees another's library."""
    data.clear_cache()
    yield
    data.clear_cache()


def _use_library(monkeypatch, library: dict) -> None:
    """Serve `library` as the Screener's factor library."""
    monkeypatch.setattr(data, 'library', lambda: library)


def _no_library(monkeypatch):
    _use_library(monkeypatch, {})


def test_merge_keeps_metrics_columns_and_adds_factors(monkeypatch):
    _no_library(monkeypatch)
    pool = data._merge_pool(_metrics_frame(), _factors_frame(), pd.DataFrame(columns=['Ticker']))
    assert list(pool['Ticker']) == ['AAA', 'BBB']   # left join: every metrics row survives
    assert pool.loc[0, 'name'] == 'Alpha Corp'      # metadata wins: overlapping identity columns come from metrics
    assert 'name_x' not in pool.columns and 'name_y' not in pool.columns
    for factor in FACTORS:
        assert pool.loc[0, factor] == 0.5
        assert pd.isna(pool.loc[1, factor])          # BBB has no factor row: factors are NaN


def test_factor_library_columns_join_the_pool(monkeypatch):
    _use_library(monkeypatch, {'momentum_12_1': factor_library.Factor(name='momentum_12_1', expression='ret_12_1')})
    library_values = pd.DataFrame({'Ticker': ['BBB'], 'momentum_12_1': [0.3]})
    pool = data._merge_pool(_metrics_frame(), _factors_frame(), library_values)
    assert pd.isna(pool.loc[0, 'momentum_12_1'])
    assert pool.loc[1, 'momentum_12_1'] == 0.3


def test_a_library_factor_that_failed_to_compute_still_gets_an_empty_column(monkeypatch):
    _use_library(monkeypatch, {'broken': factor_library.Factor(name='broken', expression='x')})
    pool = data._merge_pool(_metrics_frame(), _factors_frame(), pd.DataFrame(columns=['Ticker']))
    assert pool['broken'].isna().all()


def test_the_factor_library_is_listed_after_the_built_in_factors(monkeypatch):
    _use_library(monkeypatch, {'zeta': factor_library.Factor(name='zeta', expression='x'),
                               'alpha': factor_library.Factor(name='alpha', expression='x')})
    names = data.factor_names()
    assert names[:len(FACTORS)] == list(FACTORS)
    assert names[len(FACTORS):] == ['alpha', 'zeta']
    assert data.families()[data.LIBRARY_FAMILY] == ['alpha', 'zeta']


def test_a_library_factor_brings_its_direction_and_definition(monkeypatch):
    _use_library(monkeypatch, {'leverage': factor_library.Factor(
        name='leverage', expression='total_liabilities / total_assets', description='Debt load.', higher_is_better=False)})
    assert data.higher_is_better()['leverage'] is False
    assert data.factor_docs()['leverage'].formula == 'total_liabilities / total_assets'


def test_a_library_factor_named_like_a_built_in_one_is_reported_and_left_out(monkeypatch, tmp_path):
    factor_library.save(factor_library.Factor(name='roe', expression='net_income_common / total_equity'), tmp_path)
    factor_library.save(factor_library.Factor(name='momentum_12_1', expression='ret_12_1'), tmp_path)
    monkeypatch.setattr(data.paths, 'FACTORS_DIR', tmp_path)
    assert data.shadowed_factors() == ['roe']
    assert 'roe' not in data.library() and 'momentum_12_1' in data.library()
