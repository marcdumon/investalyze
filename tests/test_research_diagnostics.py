"""Tests for factor diagnostics: coverage, distribution, top/bottom names, correlation, turnover."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investalyze.research import diagnostics

JAN, FEB = date(2020, 1, 31), date(2021, 2, 28)


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = [
        {'Ticker': 'AAA', 'Date': JAN, 'value': 1.0, 'other': 10.0, 'input': 5.0},
        {'Ticker': 'BBB', 'Date': JAN, 'value': 2.0, 'other': 20.0, 'input': 6.0},
        {'Ticker': 'CCC', 'Date': JAN, 'value': np.nan, 'other': 30.0, 'input': 7.0},
        {'Ticker': 'AAA', 'Date': FEB, 'value': 3.0, 'other': 30.0, 'input': 8.0},
        {'Ticker': 'BBB', 'Date': FEB, 'value': 4.0, 'other': 20.0, 'input': 9.0},
        {'Ticker': 'CCC', 'Date': FEB, 'value': 5.0, 'other': 10.0, 'input': 1.0},
    ]
    return pd.DataFrame(rows)


def test_coverage_counts_names_with_a_value_per_year(frame):
    table = diagnostics.coverage(frame, 'value')
    assert table.set_index('year')['names'].to_dict() == {2020: 2, 2021: 3}
    assert table.set_index('year')['universe'].to_dict() == {2020: 3, 2021: 3}


def test_coverage_reports_the_share_that_is_usable(frame):
    table = diagnostics.coverage(frame, 'value').set_index('year')
    assert table.loc[2020, 'share'] == pytest.approx(66.7)


def test_distribution_separates_usable_values_from_missing(frame):
    summary = diagnostics.distribution(frame['value'])
    assert summary['count'] == 5
    assert summary['missing'] == 1


def test_distribution_treats_infinity_as_missing():
    values = pd.Series([1.0, 2.0, np.inf, -np.inf])
    summary = diagnostics.distribution(values)
    assert summary['count'] == 2
    assert summary['missing'] == 2


def test_distribution_of_nothing_usable_says_so():
    assert diagnostics.distribution(pd.Series([np.nan, np.nan]))['count'] == 0


def test_top_bottom_reads_the_latest_date_only(frame):
    ends = diagnostics.top_bottom(frame, 'value', n=1)
    assert ends['top']['Ticker'].tolist() == ['CCC']
    assert ends['bottom']['Ticker'].tolist() == ['AAA']


def test_top_bottom_shows_the_inputs(frame):
    ends = diagnostics.top_bottom(frame, 'value', n=1, inputs=['input'])
    assert 'input' in ends['top'].columns


def test_an_opposed_factor_correlates_negatively(frame):
    scores = diagnostics.correlations(frame, 'value', ['other'])
    assert scores['other'] < 0


def test_correlation_excludes_the_factor_itself(frame):
    assert 'value' not in diagnostics.correlations(frame, 'value', ['value', 'other'])


def test_a_stable_ranking_has_high_autocorrelation():
    rows = []
    for when in (JAN, FEB):
        for position, ticker in enumerate(('AAA', 'BBB', 'CCC')):
            rows.append({'Ticker': ticker, 'Date': when, 'value': float(position)})
    assert diagnostics.rank_autocorrelation(pd.DataFrame(rows), 'value') == pytest.approx(1.0)


def test_a_reversing_ranking_has_negative_autocorrelation():
    rows = []
    for when, order in ((JAN, (0, 1, 2)), (FEB, (2, 1, 0))):
        for position, ticker in zip(order, ('AAA', 'BBB', 'CCC')):
            rows.append({'Ticker': ticker, 'Date': when, 'value': float(position)})
    assert diagnostics.rank_autocorrelation(pd.DataFrame(rows), 'value') == pytest.approx(-1.0)


def test_coverage_counts_only_names_in_the_universe(frame):
    marked = frame.assign(in_universe=[True, True, False, True, True, True])
    table = diagnostics.coverage(marked, 'value').set_index('year')
    assert table.loc[2020, 'universe'] == 2
    assert table.loc[2020, 'share'] == pytest.approx(100.0)


def test_coverage_averages_names_per_date_rather_than_counting_rows():
    """Seen on the page: 400 names over 12 month-ends was reported as 4,754 names."""
    rows = []
    for month in (1, 2):
        for ticker in ('AAA', 'BBB', 'CCC', 'DDD'):
            rows.append({'Ticker': ticker, 'Date': pd.Timestamp(2024, month, 28), 'value': 1.0 if ticker != 'DDD' else np.nan})
    table = diagnostics.coverage(pd.DataFrame(rows), 'value').set_index('year')
    assert table.loc[2024, 'names'] == 3
    assert table.loc[2024, 'universe'] == 4
