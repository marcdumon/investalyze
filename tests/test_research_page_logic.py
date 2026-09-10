"""Tests for what the workbench says: its thresholds, its wording and its warnings."""

import numpy as np
import pandas as pd
import pytest

from investalyze.apps.research import logic
from investalyze.research import ideas

KNOWN = {'revenue', 'income', 'sector'}


# --- validation as you type ---

def test_a_valid_expression_says_so():
    ok, message = logic.validate_expression('income / revenue', KNOWN)
    assert ok and 'valid' in message.lower()


def test_an_empty_expression_asks_for_one():
    ok, message = logic.validate_expression('   ', KNOWN)
    assert not ok and 'write an expression' in message.lower()


def test_an_unknown_name_is_reported_by_name():
    ok, message = logic.validate_expression('income / wibble', KNOWN)
    assert not ok and 'wibble' in message


# --- the library reads in decision order ---

def test_kept_ideas_come_before_drafts():
    library = {'b': ideas.Idea(name='b', expression='revenue', status=ideas.DRAFT),
               'a': ideas.Idea(name='a', expression='revenue', status=ideas.KEPT)}
    assert [row['name'] for row in logic.library_rows(library)] == ['a', 'b']


def test_archived_ideas_come_last():
    library = {'a': ideas.Idea(name='a', expression='revenue', status=ideas.ARCHIVED),
               'b': ideas.Idea(name='b', expression='revenue', status=ideas.DRAFT)}
    assert [row['name'] for row in logic.library_rows(library)] == ['b', 'a']


# --- coverage is the first thing that can kill an idea ---

def _coverage(names: int, universe: int) -> pd.DataFrame:
    return pd.DataFrame([{'year': 2024, 'names': names, 'universe': universe,
                          'share': names / universe * 100}])


def test_thin_coverage_is_flagged():
    text, colour = logic.coverage_headline(_coverage(53, 6241))
    assert colour == 'orange'
    assert 'corner of the market' in text


def test_broad_coverage_is_not_flagged():
    _text, colour = logic.coverage_headline(_coverage(5000, 6241))
    assert colour == 'green'


def test_the_headline_names_both_counts():
    text, _colour = logic.coverage_headline(_coverage(53, 6241))
    assert '53' in text and '6,241' in text


# --- an idea that already exists ---

def test_a_near_duplicate_is_called_out():
    warning = logic.duplicate_warning(pd.Series({'earnings_yield': 0.94}))
    assert 'earnings_yield' in warning and 'rewording' in warning


def test_a_substantial_overlap_is_flagged_rather_than_waved_through():
    """0.85 against an existing idea is not a new idea, whatever the old threshold said."""
    text, colour = logic.similarity(pd.Series({'op_margin': 0.85}))
    assert colour == 'orange' and 'largely the same names' in text


def test_a_near_duplicate_reads_red():
    _text, colour = logic.similarity(pd.Series({'op_margin': 0.94}))
    assert colour == 'red'


def test_a_genuinely_new_idea_reads_green():
    _text, colour = logic.similarity(pd.Series({'op_margin': 0.31}))
    assert colour == 'green' 


def test_a_strong_negative_correlation_is_also_a_duplicate():
    """The same idea with its sign flipped is still the same idea."""
    assert logic.duplicate_warning(pd.Series({'earnings_yield': -0.95})) is not None


def test_a_genuinely_new_idea_draws_no_warning():
    assert logic.duplicate_warning(pd.Series({'earnings_yield': 0.31})) is None


def test_no_similarity_to_report_when_there_is_nothing_to_compare():
    assert logic.similarity(pd.Series(dtype=float)) is None


def test_no_other_ideas_means_no_warning():
    assert logic.duplicate_warning(pd.Series(dtype=float)) is None


# --- turnover, known before a backtest ---

def test_a_slow_ranking_is_described_as_cheap_to_trade():
    assert 'rarely' in logic.turnover_note(0.97)


def test_a_churning_ranking_warns_about_costs():
    assert 'costs' in logic.turnover_note(0.4)


# --- the level two headline ---

def _answer(t_stat: float, monotonic: bool = True, names: float = 900.0) -> dict:
    return {'summary': {'oriented_ic': 0.02, 'oriented_t': t_stat, 'n_periods': 84,
                        'names_median': names},
            'monotonic': monotonic, 'tested': 12, 'expected_false_positives': 0.6}


def test_a_weak_t_statistic_is_called_luck():
    text, colour = logic.verdict_headline(_answer(1.2))
    assert colour == 'gray' and 'luck' in text


def test_a_significant_ordered_result_reads_green():
    _text, colour = logic.verdict_headline(_answer(3.1, monotonic=True))
    assert colour == 'green'


def test_significance_without_ordered_quantiles_is_hedged():
    text, colour = logic.verdict_headline(_answer(3.1, monotonic=False))
    assert colour == 'orange' and 'not ordered' in text


def test_the_headline_always_carries_the_name_count():
    text, _colour = logic.verdict_headline(_answer(3.1, names=82))
    assert '82 names' in text


def test_an_unjudgeable_idea_says_so():
    _text, colour = logic.verdict_headline(_answer(float('nan')))
    assert colour == 'gray'


# --- multiple testing ---

def test_the_multiple_testing_note_names_both_numbers():
    note = logic.multiple_testing_note(_answer(3.0))
    assert '12 ideas' in note and '0.6' in note


# --- decay argues for a rebalance frequency ---

def test_decay_points_at_the_strongest_horizon():
    decay = pd.DataFrame({'horizon': [5, 21, 63], 'mean_ic': [0.01, 0.03, 0.02]})
    assert '21 sessions' in logic.decay_note(decay)


def test_decay_with_nothing_readable_says_so():
    decay = pd.DataFrame({'horizon': [5], 'mean_ic': [np.nan]})
    assert 'No decay' in logic.decay_note(decay)
