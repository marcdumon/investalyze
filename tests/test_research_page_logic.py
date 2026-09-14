"""Tests for the Factors page text and thresholds."""

import numpy as np
import pandas as pd

from investalyze.apps.research import logic
from investalyze.research import factors

KNOWN = {'revenue', 'income', 'sector'}


# --- expression validation ---

def test_a_valid_expression_says_so():
    ok, message = logic.validate_expression('income / revenue', KNOWN)
    assert ok and 'valid' in message.lower()


def test_an_empty_expression_asks_for_one():
    ok, message = logic.validate_expression('   ', KNOWN)
    assert not ok and 'enter an expression' in message.lower()


def test_an_unknown_name_is_reported_by_name():
    ok, message = logic.validate_expression('income / wibble', KNOWN)
    assert not ok and 'wibble' in message


# --- factor library order ---

def test_accepted_factors_come_before_candidates():
    library = {'b': factors.Factor(name='b', expression='revenue', status=factors.CANDIDATE),
               'a': factors.Factor(name='a', expression='revenue', status=factors.ACCEPTED)}
    assert [row['name'] for row in logic.library_rows(library)] == ['a', 'b']


def test_rejected_factors_come_last():
    library = {'a': factors.Factor(name='a', expression='revenue', status=factors.REJECTED),
               'b': factors.Factor(name='b', expression='revenue', status=factors.CANDIDATE)}
    assert [row['name'] for row in logic.library_rows(library)] == ['b', 'a']


# --- coverage ---

def _coverage(names: int, universe: int) -> pd.DataFrame:
    return pd.DataFrame([{'year': 2024, 'names': names, 'universe': universe, 'share': names / universe * 100}])


def test_low_coverage_is_flagged():
    text, colour = logic.coverage_note(_coverage(53, 6241))
    assert colour == 'orange' and 'Low coverage' in text


def test_broad_coverage_is_not_flagged():
    _text, colour = logic.coverage_note(_coverage(5000, 6241))
    assert colour == 'green'


def test_the_coverage_note_names_both_counts():
    text, _colour = logic.coverage_note(_coverage(53, 6241))
    assert '53' in text and '6,241' in text


# --- correlation with existing factors ---

def test_a_redundant_factor_reads_red():
    text, colour = logic.correlation_note(pd.Series({'op_margin': 0.94}))
    assert colour == 'red' and 'Redundant' in text


def test_high_overlap_reads_orange():
    """0.85 against an existing factor is high overlap, not a new factor."""
    text, colour = logic.correlation_note(pd.Series({'op_margin': 0.85}))
    assert colour == 'orange' and 'High overlap' in text


def test_a_strong_negative_correlation_is_also_redundant():
    """A factor with its sign flipped is the same factor."""
    _text, colour = logic.correlation_note(pd.Series({'earnings_yield': -0.95}))
    assert colour == 'red'


def test_low_overlap_reads_green():
    _text, colour = logic.correlation_note(pd.Series({'op_margin': 0.31}))
    assert colour == 'green'


def test_the_redundancy_warning_names_the_other_factor():
    assert 'earnings_yield' in logic.redundancy_warning(pd.Series({'earnings_yield': 0.94}))


def test_low_overlap_draws_no_warning():
    assert logic.redundancy_warning(pd.Series({'earnings_yield': 0.31})) is None


def test_no_other_factors_means_no_correlation_note():
    assert logic.correlation_note(pd.Series(dtype=float)) is None


# --- turnover ---

def test_high_rank_autocorrelation_is_low_turnover():
    assert 'low turnover' in logic.turnover_note(0.97)


def test_low_rank_autocorrelation_warns_about_costs():
    assert 'transaction costs' in logic.turnover_note(0.4)


# --- IC summary line ---

def _result(t_stat: float, monotonic: bool = True, names: float = 900.0) -> dict:
    return {'summary': {'oriented_ic': 0.02, 'oriented_t': t_stat, 'n_periods': 84, 'names_median': names},
            'monotonic': monotonic, 'tested': 12, 'expected_false_positives': 0.6}


def test_a_low_t_stat_is_not_significant():
    text, colour = logic.ic_note(_result(1.2))
    assert colour == 'gray' and 'Not significant' in text


def test_significant_and_monotonic_reads_green():
    _text, colour = logic.ic_note(_result(3.1, monotonic=True))
    assert colour == 'green'


def test_significant_but_not_monotonic_reads_orange():
    text, colour = logic.ic_note(_result(3.1, monotonic=False))
    assert colour == 'orange' and 'not monotonic' in text


def test_the_ic_note_carries_the_cross_section_size():
    text, _colour = logic.ic_note(_result(3.1, names=82))
    assert '82 names' in text


def test_too_few_dates_says_so():
    _text, colour = logic.ic_note(_result(float('nan')))
    assert colour == 'gray'


def test_the_multiple_testing_note_names_both_numbers():
    note = logic.multiple_testing_note(_result(3.0))
    assert '12 factors' in note and '0.6' in note


# --- IC decay ---

def test_ic_decay_names_the_peak_horizon():
    decay = pd.DataFrame({'horizon': [5, 21, 63], 'mean_ic': [0.01, 0.03, 0.02]})
    assert 'peaks at the 21-day' in logic.ic_decay_note(decay)


def test_a_large_negative_ic_is_not_called_the_peak():
    """Seen on the page: 126-day IC of -0.0265 was reported as the peak."""
    decay = pd.DataFrame({'horizon': [5, 21, 63, 126], 'mean_ic': [0.019, 0.009, -0.012, -0.0265]})
    note = logic.ic_decay_note(decay)
    assert 'peaks at the 5-day' in note
    assert 'turns negative at 63-day, 126-day' in note


def test_a_lower_is_better_factor_peaks_at_its_most_negative_ic():
    decay = pd.DataFrame({'horizon': [5, 21], 'mean_ic': [-0.01, -0.04]})
    assert 'peaks at the 21-day' in logic.ic_decay_note(decay, higher_is_better=False)


def test_ic_negative_everywhere_says_the_factor_is_reversed():
    decay = pd.DataFrame({'horizon': [5, 21], 'mean_ic': [-0.01, -0.02]})
    assert 'opposite direction' in logic.ic_decay_note(decay)


def test_ic_decay_with_no_readable_horizon_says_so():
    decay = pd.DataFrame({'horizon': [5], 'mean_ic': [np.nan]})
    assert 'no readable horizon' in logic.ic_decay_note(decay)


# --- library results ---

def test_library_rows_carry_recorded_results():
    library = {'a': factors.Factor(name='a', expression='revenue')}
    rows = logic.library_rows(library, {'a': {'t_stat': 2.4, 'sharpe': 0.9}})
    assert rows[0]['t_stat'] == 2.4 and rows[0]['sharpe'] == 0.9


def test_a_factor_never_analysed_has_empty_results():
    rows = logic.library_rows({'a': factors.Factor(name='a', expression='revenue')}, {})
    assert rows[0]['t_stat'] is None


# --- holdout ---

def _split(inside: float, outside: float) -> pd.DataFrame:
    return pd.DataFrame([
        {'period': 'In-sample', 'start': '2016-02-29', 'end': '2021-12-31', 'mean_ic': 0.02, 't_stat': inside, 'n_periods': 70},
        {'period': 'Out-of-sample', 'start': '2022-01-31', 'end': '2024-12-31', 'mean_ic': 0.01, 't_stat': outside, 'n_periods': 36},
    ])


def test_significant_in_sample_only_reads_orange():
    text, colour = logic.holdout_note(_split(2.8, 0.6))
    assert colour == 'orange' and 'in-sample only' in text


def test_a_sign_reversal_out_of_sample_reads_red():
    _text, colour = logic.holdout_note(_split(2.8, -1.1))
    assert colour == 'red'


def test_significant_out_of_sample_reads_green():
    _text, colour = logic.holdout_note(_split(2.8, 2.3))
    assert colour == 'green'


def test_no_split_means_no_holdout_note():
    assert logic.holdout_note(pd.DataFrame()) is None
