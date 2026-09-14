"""Tests for the result cache: current results are found, stale results are not."""

import pandas as pd
import pytest

from investalyze.research import cache


@pytest.fixture
def store(tmp_path) -> cache.Cache:
    return cache.Cache(tmp_path / 'cache')


def _frame(value: int = 1) -> pd.DataFrame:
    return pd.DataFrame({'Ticker': ['AAA'], 'value': [value]})


def test_a_result_round_trips(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1')['value'].tolist() == [7]


def test_a_result_never_computed_is_absent(store):
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is None


def test_a_different_fingerprint_does_not_read_the_old_result(store):
    """A redefined factor must not read results from its previous definition."""
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp2') is None


def test_storing_a_new_result_removes_the_superseded_one(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    store.put(cache.VALUES, 'margin', 'q1', 'fp2', _frame(8))
    assert store.stale(cache.VALUES, 'margin', 'q1', 'fp2') == []
    assert not store.path(cache.VALUES, 'margin', 'q1', 'fp1').is_file()


def test_only_one_result_per_factor_and_params_survives(store):
    for fingerprint in ('fp1', 'fp2', 'fp3'):
        store.put(cache.VALUES, 'margin', 'q1', fingerprint, _frame())
    stored = list((store.root / cache.VALUES).glob('margin__q1__*.parquet'))
    assert len(stored) == 1


def test_different_params_for_the_same_factor_are_kept(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame())
    store.put(cache.VALUES, 'margin', 'q2', 'fp1', _frame())
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is not None
    assert store.get(cache.VALUES, 'margin', 'q2', 'fp1') is not None


def test_another_factor_is_untouched(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame())
    store.put(cache.VALUES, 'other', 'q1', 'fp1', _frame())
    store.put(cache.VALUES, 'margin', 'q1', 'fp2', _frame())
    assert store.get(cache.VALUES, 'other', 'q1', 'fp1') is not None


def test_metadata_records_what_was_computed(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(), meta={'names': 904})
    recorded = store.meta(cache.VALUES, 'margin', 'q1', 'fp1')
    assert recorded['names'] == 904
    assert recorded['rows'] == 1
    assert 'computed_at' in recorded


def test_forgetting_a_factor_drops_all_its_results(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame())
    store.put(cache.IC, 'margin', 'q2', 'fp1', _frame())
    store.forget('margin')
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is None
    assert store.get(cache.IC, 'margin', 'q2', 'fp1') is None


def test_the_same_params_always_hash_the_same(store):
    assert cache.params_hash(a=1, b='x') == cache.params_hash(b='x', a=1)


def test_different_params_hash_differently(store):
    assert cache.params_hash(horizon=21) != cache.params_hash(horizon=63)


def test_a_summary_accumulates_results_for_one_definition(store):
    store.put_summary('margin', 'fp1', {'ic': 0.02})
    store.put_summary('margin', 'fp1', {'sharpe': 0.8})
    assert store.get_summary('margin', 'fp1') == {'ic': 0.02, 'sharpe': 0.8}


def test_a_new_definition_starts_with_no_summary(store):
    store.put_summary('margin', 'fp1', {'ic': 0.02})
    assert store.get_summary('margin', 'fp2') == {}


def test_writing_a_summary_for_a_new_definition_deletes_the_old_one(store):
    store.put_summary('margin', 'fp1', {'ic': 0.02})
    store.put_summary('margin', 'fp2', {'ic': 0.03})
    assert list((store.root / cache.SUMMARY).glob('margin__*.json')) == [store.root / cache.SUMMARY / 'margin__fp2.json']
