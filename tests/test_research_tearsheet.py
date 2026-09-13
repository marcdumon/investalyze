"""Tests for the tear sheet: IC summary, cross-section size, monotonicity and multiple testing."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investalyze.research import tearsheet


def _ic(values: list, names: list | None = None) -> pd.DataFrame:
    dates = pd.date_range('2020-01-31', periods=len(values), freq='ME')
    return pd.DataFrame({'Date': dates, 'ic': values,
                         'names': names if names is not None else [500] * len(values)})


# --- IC summary ---

def test_a_consistent_ic_has_a_high_t_stat():
    summary = tearsheet.ic_summary(_ic([0.02, 0.03, 0.025, 0.028, 0.022]))
    assert summary['mean'] > 0
    assert summary['t_stat'] > 3


def test_an_alternating_ic_averages_to_zero():
    summary = tearsheet.ic_summary(_ic([0.05, -0.05, 0.05, -0.05]))
    assert summary['mean'] == pytest.approx(0.0)


def test_orientation_flips_a_factor_whose_low_end_is_good():
    values = [-0.02, -0.03, -0.025]
    assert tearsheet.ic_summary(_ic(values), higher_is_better=False)['oriented_ic'] > 0
    assert tearsheet.ic_summary(_ic(values), higher_is_better=True)['oriented_ic'] < 0


def test_an_empty_series_reports_no_periods():
    assert tearsheet.ic_summary(_ic([np.nan, np.nan]))['n_periods'] == 0


# --- cross-section size ---

def test_the_summary_says_how_many_names_it_measured():
    summary = tearsheet.ic_summary(_ic([0.02, 0.03], names=[904, 906]))
    assert summary['names_median'] == pytest.approx(905.0)


def test_the_smallest_cross_section_is_reported_too():
    """A median of 900 hides a date with 12 names; the minimum shows it."""
    summary = tearsheet.ic_summary(_ic([0.02, 0.03, 0.01], names=[900, 12, 950]))
    assert summary['names_min'] == 12


def test_dates_without_an_ic_do_not_count_towards_the_names():
    summary = tearsheet.ic_summary(_ic([0.02, np.nan], names=[900, 3]))
    assert summary['names_median'] == 900


# --- quantile monotonicity ---

def _quantiles(returns: list) -> pd.DataFrame:
    return pd.DataFrame({'Quantile': range(len(returns)), 'mean_return': returns})


def test_returns_rising_across_quantiles_are_monotonic():
    assert tearsheet.is_monotonic(_quantiles([-0.01, 0.0, 0.01, 0.02, 0.03])) is True


def test_a_jumbled_spread_is_not_monotonic():
    assert tearsheet.is_monotonic(_quantiles([0.01, -0.02, 0.03, 0.0, 0.01])) is False


def test_monotonicity_respects_the_direction_a_factor_reads():
    falling = _quantiles([0.03, 0.02, 0.01, 0.0, -0.01])
    assert tearsheet.is_monotonic(falling, higher_is_better=False) is True
    assert tearsheet.is_monotonic(falling, higher_is_better=True) is False


def test_a_quantile_with_no_return_is_not_monotonic():
    assert tearsheet.is_monotonic(_quantiles([0.01, np.nan, 0.03])) is False


# --- cache key ---

def _panel(tickers: list, dates: list) -> pd.DataFrame:
    rows = [{'Ticker': ticker, 'Date': when, 'value': 1.0} for when in dates for ticker in tickers]
    return pd.DataFrame(rows)


def test_a_different_universe_gets_a_different_cache_key():
    """Seen on the page: a CLI run on 800 names served its IC to a page run on 1,500 names."""
    dates = [date(2020, 1, 31), date(2020, 2, 29)]
    assert tearsheet.ic_params(_panel(['AAA', 'BBB'], dates), 21) != tearsheet.ic_params(_panel(['AAA', 'CCC'], dates), 21)


def test_a_different_date_range_gets_a_different_cache_key():
    tickers = ['AAA', 'BBB']
    assert tearsheet.ic_params(_panel(tickers, [date(2020, 1, 31)]), 21) != \
        tearsheet.ic_params(_panel(tickers, [date(2021, 1, 31)]), 21)


def test_the_same_universe_and_dates_share_a_cache_key():
    dates = [date(2020, 1, 31)]
    assert tearsheet.ic_params(_panel(['BBB', 'AAA'], dates), 21) == tearsheet.ic_params(_panel(['AAA', 'BBB'], dates), 21)


# --- multiple testing ---

def test_a_larger_library_expects_more_false_positives():
    assert tearsheet.expected_false_positives(20) == pytest.approx(1.0)
    assert tearsheet.expected_false_positives(100) == pytest.approx(5.0)


def test_an_empty_library_expects_none():
    assert tearsheet.expected_false_positives(0) == 0.0
