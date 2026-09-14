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


def test_several_factors_need_the_history_of_the_most_demanding_one():
    library = {'long': _factor('long', 'mean(revenue, 252)'), 'short': _factor('short', 'lag(revenue, 5)')}
    assert compute.lookback_days(['long', 'short'], library) == 252


def test_combining_panels_keeps_the_rows_and_columns_of_both():
    daily = pd.DataFrame({'Ticker': ['AAA'], 'Date': pd.to_datetime(['2020-01-31']), 'in_universe': [True], 'trend': [1.0]})
    dated = pd.DataFrame({'Ticker': ['AAA', 'BBB'], 'Date': pd.to_datetime(['2020-01-31'] * 2),
                          'in_universe': [True, False], 'margin': [0.1, 0.2]})
    combined = compute._combine(daily, dated)
    assert combined[['Ticker', 'trend', 'margin']].fillna(-1).values.tolist() == [['AAA', 1.0, 0.1], ['BBB', -1, 0.2]]
    assert combined['in_universe'].tolist() == [True, False]


def test_factors_with_and_without_history_are_computed_on_their_own_grids(monkeypatch):
    library = {'trend': _factor('trend', 'mean(revenue, 2)'), 'margin': _factor('margin', 'income_net_income / revenue')}
    calls = []

    def fake_panel(con, needed, *, as_of, version, tickers):
        calls.append((sorted(needed), len(as_of)))
        dates = pd.DatetimeIndex(as_of)
        return pd.DataFrame({'Ticker': ['AAA'] * len(dates), 'Date': dates, 'revenue': [10.0] * len(dates),
                             'income_net_income': [1.0] * len(dates)})

    sessions = pd.bdate_range('2020-01-01', '2020-01-31')
    monkeypatch.setattr(compute.fields, 'catalog', lambda con: CATALOG)
    monkeypatch.setattr(compute.fields, 'panel', fake_panel)
    monkeypatch.setattr(compute.panels, 'trading_calendar', lambda con, start, end: sessions)
    frame = compute.compute(None, ['trend', 'margin'], library, as_of=pd.DatetimeIndex(['2020-01-31']))

    assert (['revenue'], len(sessions)) in calls and (['income_net_income', 'revenue'], 1) in calls
    assert frame[['trend', 'margin']].values.tolist() == [[10.0, 0.1]]


def test_a_session_without_data_still_counts_in_a_time_series_window():
    sessions = pd.bdate_range('2020-01-01', periods=3)
    frame = pd.DataFrame({'Ticker': ['AAA', 'AAA'], 'Date': sessions[[0, 2]], 'close': [1.0, 3.0]})
    completed = compute.complete_grid(frame, sessions)
    assert completed['Date'].tolist() == list(sessions)
    assert completed['close'].isna().tolist() == [False, True, False]
