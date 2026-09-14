"""Tests for resolving the data fields a factor needs before any query runs."""

import pandas as pd
import pytest

from investalyze.research import compute, factors, fields

CATALOG = {name: fields.Field(name, name.title(), 'income')
           for name in ('revenue', 'income_net_income', 'total_assets')}


def _factor(name: str, expression: str) -> factors.Factor:
    return factors.Factor(name=name, expression=expression)


def test_a_factor_needs_only_its_own_fields():
    library = {'margin': _factor('margin', 'income_net_income / revenue')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_fields_are_followed_through_a_referenced_factor():
    library = {'margin': _factor('margin', 'income_net_income / revenue'),
               'ranked': _factor('ranked', 'rank(margin)')}
    assert compute.required_fields(['ranked'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_unrelated_factor_contributes_nothing():
    library = {'margin': _factor('margin', 'income_net_income / revenue'),
               'other': _factor('other', 'total_assets')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_identity_column_counts_as_a_field():
    library = {'neutral': _factor('neutral', 'neutralize(revenue, by=sector)')}
    assert compute.required_fields(['neutral'], library, CATALOG) == {'revenue', 'sector'}


def test_an_unknown_name_is_refused_naming_the_factor_that_reads_it():
    library = {'margin': _factor('margin', 'income_net_income / wibble')}
    with pytest.raises(ValueError, match="factor 'margin' reads unknown name 'wibble'"):
        compute.required_fields(['margin'], library, CATALOG)


def test_apply_universe_marks_members_per_date():
    frame = pd.DataFrame({'Date': pd.to_datetime(['2020-01-31', '2020-01-31', '2020-02-28']),
                          'Ticker': ['AAA', 'BBB', 'AAA']})
    members = pd.DataFrame({'Date': pd.to_datetime(['2020-01-31', '2020-02-28']), 'Ticker': ['AAA', 'AAA']})
    assert compute.apply_universe(frame, members)['in_universe'].tolist() == [True, False, True]


def test_membership_holds_until_the_next_rebalance_date():
    frame = pd.DataFrame({'Date': pd.to_datetime(['2020-01-30', '2020-01-31', '2020-02-03', '2020-02-28', '2020-03-02']),
                          'Ticker': ['AAA'] * 5})
    members = pd.DataFrame({'Date': pd.to_datetime(['2020-01-31']), 'Ticker': ['AAA']})
    marked = compute.apply_universe(frame, members)
    # before the first rebalance date the name is outside; afterwards membership carries forward
    assert marked['in_universe'].tolist() == [False, True, True, True, True]


def test_a_name_dropped_at_a_rebalance_leaves_the_universe_from_that_date():
    frame = pd.DataFrame({'Date': pd.to_datetime(['2020-02-03', '2020-03-02']), 'Ticker': ['AAA', 'AAA']})
    members = pd.DataFrame({'Date': pd.to_datetime(['2020-01-31', '2020-02-28']), 'Ticker': ['AAA', 'BBB']})
    assert compute.apply_universe(frame, members)['in_universe'].tolist() == [True, False]


def test_lookback_counts_windows_through_referenced_factors():
    library = {'trend': _factor('trend', 'mean(revenue, 50)'), 'signal': _factor('signal', 'trend > lag(trend, 20)')}
    assert compute.lookback_days(['signal'], library) == 70


def test_a_factor_without_time_series_functions_needs_no_history():
    assert compute.lookback_days(['margin'], {'margin': _factor('margin', 'income_net_income / revenue')}) == 0
