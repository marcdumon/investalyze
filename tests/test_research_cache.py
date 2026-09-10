"""Tests for the answer cache: that a current answer is found, and a stale one cannot be."""

import pandas as pd
import pytest

from investalyze.research import cache


@pytest.fixture
def store(tmp_path) -> cache.Cache:
    return cache.Cache(tmp_path / 'cache')


def _frame(value: int = 1) -> pd.DataFrame:
    return pd.DataFrame({'Ticker': ['AAA'], 'value': [value]})


def test_an_answer_comes_back_as_it_went_in(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1')['value'].tolist() == [7]


def test_an_answer_never_computed_is_absent(store):
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is None


def test_a_different_fingerprint_does_not_read_the_old_answer(store):
    """The whole point: a redefined idea cannot serve the previous definition's numbers."""
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp2') is None


def test_storing_a_new_answer_removes_the_superseded_one(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame(7))
    store.put(cache.VALUES, 'margin', 'q1', 'fp2', _frame(8))
    assert store.stale(cache.VALUES, 'margin', 'q1', 'fp2') == []
    assert not store.path(cache.VALUES, 'margin', 'q1', 'fp1').is_file()


def test_only_one_answer_per_idea_and_question_survives(store):
    for fingerprint in ('fp1', 'fp2', 'fp3'):
        store.put(cache.VALUES, 'margin', 'q1', fingerprint, _frame())
    stored = list((store.root / cache.VALUES).glob('margin__q1__*.parquet'))
    assert len(stored) == 1


def test_a_different_question_about_the_same_idea_is_kept(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame())
    store.put(cache.VALUES, 'margin', 'q2', 'fp1', _frame())
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is not None
    assert store.get(cache.VALUES, 'margin', 'q2', 'fp1') is not None


def test_another_idea_is_untouched(store):
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


def test_forgetting_an_idea_drops_every_answer_about_it(store):
    store.put(cache.VALUES, 'margin', 'q1', 'fp1', _frame())
    store.put(cache.PREDICT, 'margin', 'q2', 'fp1', _frame())
    store.forget('margin')
    assert store.get(cache.VALUES, 'margin', 'q1', 'fp1') is None
    assert store.get(cache.PREDICT, 'margin', 'q2', 'fp1') is None


def test_the_same_parameters_always_sign_the_same(store):
    assert cache.signature(a=1, b='x') == cache.signature(b='x', a=1)


def test_different_parameters_sign_differently(store):
    assert cache.signature(horizon=21) != cache.signature(horizon=63)
