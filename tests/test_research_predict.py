"""Tests for level two: how an IC series is read, and the honesty carried beside it."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investalyze.research import predict


def _ic(values: list, names: list | None = None) -> pd.DataFrame:
    dates = pd.date_range('2020-01-31', periods=len(values), freq='ME')
    return pd.DataFrame({'Date': dates, 'ic': values,
                         'names': names if names is not None else [500] * len(values)})


# --- reading the series ---

def test_a_consistent_edge_shows_a_positive_t_statistic():
    summary = predict.summarise(_ic([0.02, 0.03, 0.025, 0.028, 0.022]))
    assert summary['mean'] > 0
    assert summary['t_stat'] > 3


def test_a_series_averaging_nothing_has_no_edge():
    summary = predict.summarise(_ic([0.05, -0.05, 0.05, -0.05]))
    assert summary['mean'] == pytest.approx(0.0)


def test_orientation_flips_a_factor_whose_low_end_is_good():
    values = [-0.02, -0.03, -0.025]
    assert predict.summarise(_ic(values), higher_is_better=False)['oriented_ic'] > 0
    assert predict.summarise(_ic(values), higher_is_better=True)['oriented_ic'] < 0


def test_an_empty_series_reports_no_periods():
    assert predict.summarise(_ic([np.nan, np.nan]))['n_periods'] == 0


# --- the name count travels with the number ---

def test_the_summary_says_how_many_names_it_measured():
    summary = predict.summarise(_ic([0.02, 0.03], names=[904, 906]))
    assert summary['names_median'] == pytest.approx(905.0)


def test_the_smallest_cross_section_is_reported_too():
    """A median of 900 hides a date that scored 12 names; the minimum does not."""
    summary = predict.summarise(_ic([0.02, 0.03, 0.01], names=[900, 12, 950]))
    assert summary['names_min'] == 12


def test_dates_without_an_ic_do_not_count_towards_the_names():
    summary = predict.summarise(_ic([0.02, np.nan], names=[900, 3]))
    assert summary['names_median'] == 900


# --- monotonicity is the shape a real edge has ---

def _buckets(returns: list) -> pd.DataFrame:
    return pd.DataFrame({'Quantile': range(len(returns)), 'mean_return': returns})


def test_returns_climbing_across_buckets_are_monotonic():
    assert predict.is_monotonic(_buckets([-0.01, 0.0, 0.01, 0.02, 0.03])) is True


def test_a_jumbled_spread_is_not_monotonic():
    assert predict.is_monotonic(_buckets([0.01, -0.02, 0.03, 0.0, 0.01])) is False


def test_monotonicity_respects_the_direction_a_factor_reads():
    falling = _buckets([0.03, 0.02, 0.01, 0.0, -0.01])
    assert predict.is_monotonic(falling, higher_is_better=False) is True
    assert predict.is_monotonic(falling, higher_is_better=True) is False


def test_a_bucket_with_no_return_is_not_monotonic():
    assert predict.is_monotonic(_buckets([0.01, np.nan, 0.03])) is False


# --- testing many ideas manufactures winners ---

def test_a_larger_library_expects_more_false_positives():
    assert predict.expected_false_positives(20) == pytest.approx(1.0)
    assert predict.expected_false_positives(100) == pytest.approx(5.0)


def test_an_empty_library_expects_none():
    assert predict.expected_false_positives(0) == 0.0
