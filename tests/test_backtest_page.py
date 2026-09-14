"""Tests for the Backtest page: job commands, run settings, periods, and the callbacks that launch and delete."""

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from investalyze.apps.backtest import logic


def _page():
    from investalyze.apps.backtest import page
    return page


# --- curves ---

def test_rebase_starts_every_column_at_100():
    curve = pd.DataFrame({'Equity': [50.0, 75.0], 'BenchmarkEquity': [200.0, 100.0]})
    rebased = logic.rebase(curve, ['Equity', 'BenchmarkEquity'])
    assert rebased['Equity'].tolist() == [100.0, 150.0]
    assert rebased['BenchmarkEquity'].tolist() == [100.0, 50.0]


def test_rebase_leaves_an_all_missing_column_alone():
    curve = pd.DataFrame({'Equity': [50.0, 75.0], 'BenchmarkEquity': [np.nan, np.nan]})
    assert logic.rebase(curve, ['Equity', 'BenchmarkEquity'])['BenchmarkEquity'].isna().all()


def test_drawdown_is_measured_from_the_running_peak():
    assert logic.drawdown_from_equity(pd.Series([100.0, 120.0, 90.0])).tolist() == pytest.approx([0.0, 0.0, -0.25])


# --- job commands ---

def test_a_factor_portfolio_runs_the_research_backtest_command():
    title, argv = logic.factor_portfolio_argv(['momentum_12_1', 'earnings_yield'], '2016-01-31', '2024-12-31',
                                              'Q', 5, 1000, '2022-01-01')
    assert argv == ['-m', 'investalyze.research', 'backtest', 'momentum_12_1,earnings_yield',
                    '--start', '2016-01-31', '--end', '2024-12-31', '--rebalance', 'Q', '--quantiles', '5',
                    '--size', '1000', '--holdout', '2022-01-01']
    assert 'momentum_12_1 + earnings_yield' in title


def test_a_blank_holdout_means_no_out_of_sample_split():
    _title, argv = logic.factor_portfolio_argv(['value'], '2016-01-31', '2024-12-31', 'M', 5, 1000, None)
    assert argv[-2:] == ['--holdout', '']


def _timing_rule(**overrides) -> tuple[str, list[str]]:
    settings = dict(exit_factor=None, exit_on_entry_false=True, trailing_stop_percent=15.0, max_positions=20,
                    priority_factor=None, universe=None, universe_size=500, start='2016-01-01', end='2024-12-31',
                    holdout='2022-01-01')
    return logic.timing_rule_argv('new_252d_high', **(settings | overrides))


def test_a_timing_rule_runs_the_research_timing_command():
    title, argv = _timing_rule()
    assert argv == ['-m', 'investalyze.research', 'timing', 'new_252d_high', '--start', '2016-01-01',
                    '--end', '2024-12-31', '--max-positions', '20', '--size', '500', '--holdout', '2022-01-01',
                    '--trailing-stop', '0.15']
    assert 'new_252d_high, exit on entry false, trailing stop 15%' in title


def test_a_timing_rule_passes_its_exit_priority_and_universe():
    _title, argv = _timing_rule(exit_factor='close_above_prior_high', exit_on_entry_false=False,
                                trailing_stop_percent=None, priority_factor='momentum_12_1', universe='solar')
    assert argv[-7:] == ['--exit-factor', 'close_above_prior_high', '--keep-on-entry-false',
                         '--priority-factor', 'momentum_12_1', '--universe', 'solar']


def test_a_timing_rule_without_any_exit_is_refused():
    problem = logic.timing_rule_problem(exit_factor=None, exit_on_entry_false=False, trailing_stop_percent=None)
    assert 'at least one exit' in problem


def test_a_trailing_stop_of_100_percent_or_more_is_refused():
    problem = logic.timing_rule_problem(exit_factor=None, exit_on_entry_false=True, trailing_stop_percent=100.0)
    assert 'between 0 and 100' in problem


def test_a_timing_rule_with_an_exit_can_run():
    assert logic.timing_rule_problem(exit_factor='close_above_prior_high', exit_on_entry_false=False,
                                     trailing_stop_percent=None) is None


def test_deleting_several_runs_is_one_command():
    title, argv = logic.delete_argv(['aaa', 'bbb'])
    assert argv == ['-m', 'investalyze.backtest', 'delete', '--run-id', 'aaa', 'bbb']
    assert '2 runs' in title


def test_the_delete_confirmation_names_each_run_and_totals_the_fills():
    runs = [{'Created': '2026-09-13 10:00', 'Signal': 'value', 'StartDate': '2016-01-31', 'EndDate': '2024-12-31',
             'NTrades': 1200, 'RunId': 'aaa'},
            {'Created': '2026-09-13 11:00', 'Signal': 'new_252d_high, trailing stop 15%', 'StartDate': '2016-01-01',
             'EndDate': '2024-12-31',
             'NTrades': 300, 'RunId': 'bbb'}]
    text = logic.delete_prompt(runs)
    assert '1,500 fills in total' in text and 'run aaa' in text and 'run bbb' in text


# --- what a stored run was made with ---

def _factor_config(holdout: str | None = '2022-01-01') -> str:
    return json.dumps({
        'signal': 'value + momentum', 'benchmark': '^NDX',
        'params': {'factors': ['value', 'momentum'], 'n_quantiles': 5, 'rebalance': 'Q', 'universe_size': 1000,
                   'holdout': holdout, 'expressions': {'value': 'income_net_income / market_cap', 'momentum': 'ret_12_1'}},
        'costs': {'slippage_bps': 5.0, 'commission_bps': 2.0, 'impact_coef': 0.1},
    })


def test_run_settings_list_each_factor_with_its_expression():
    rows = {row['setting']: row['value'] for row in logic.run_settings_rows(_factor_config())}
    assert rows['Factor value'] == 'income_net_income / market_cap'
    assert rows['Factor momentum'] == 'ret_12_1'


def test_run_settings_describe_the_portfolio_rules():
    rows = {row['setting']: row['value'] for row in logic.run_settings_rows(_factor_config())}
    assert rows['Quantiles'] == '5'
    assert rows['Rebalance'] == 'quarterly'
    assert rows['Universe size'] == '1000'
    assert rows['Out-of-sample from'] == '2022-01-01'
    assert rows['Benchmark'] == '^NDX'
    assert '5 bps slippage' in rows['Costs']


def test_run_settings_describe_the_timing_rule():
    config = json.dumps({'benchmark': '^NDX', 'params': {
        'entry_factor': 'new_252d_high', 'exit_on_entry_false': False, 'exit_factor': 'close_above_prior_high',
        'trailing_stop': 0.15, 'max_positions': 20, 'rebalance': 'D'}})
    rows = {row['setting']: row['value'] for row in logic.run_settings_rows(config)}
    assert rows['Entry factor'] == 'new_252d_high'
    assert rows['Exit when entry is false'] == 'no'
    assert rows['Exit factor'] == 'close_above_prior_high'
    assert rows['Trailing stop'] == '15%'
    assert rows['Max positions'] == '20'
    assert rows['Rebalance'] == 'daily'


def test_a_run_without_a_config_has_no_settings():
    assert logic.run_settings_rows(None) == []


def test_the_holdout_is_read_from_the_run_config():
    assert logic.holdout_of(_factor_config()) == pd.Timestamp('2022-01-01')
    assert logic.holdout_of(_factor_config(holdout=None)) is None
    assert logic.holdout_of(None) is None


def test_run_periods_split_a_stored_curve_at_the_holdout():
    dates = pd.bdate_range('2021-01-01', '2023-12-29')
    curve = pd.DataFrame({'Date': dates, 'Equity': np.linspace(100, 150, len(dates)), 'BenchmarkEquity': np.nan})
    table = logic.run_periods(curve, pd.Timestamp('2023-01-01'))
    assert table['period'].tolist() == ['Full sample', 'In-sample', 'Out-of-sample']


# --- choosing factors, including arriving from the Factors page ---

ROWS = [{'name': 'value', 'status': 'accepted'}, {'name': 'momentum', 'status': 'candidate'}]


def test_the_factor_library_is_listed_as_table_rows(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.data, 'factor_rows', lambda: ROWS)
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-runs-tick'))
    rows, _selected = page.load_factor_library(0, '', None)
    assert [row['name'] for row in rows] == ['value', 'momentum']


def test_a_factor_in_the_url_is_ticked_on_the_first_load(monkeypatch):
    """The first load reports no trigger, and that is when a link from the Factors page arrives."""
    page = _page()
    monkeypatch.setattr(page.data, 'factor_rows', lambda: ROWS)
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id=None))
    assert page.load_factor_library(0, '?factor=momentum', None)[1] == {'ids': ['momentum']}


def test_a_job_finishing_does_not_retick_the_linked_factor(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.data, 'factor_rows', lambda: ROWS)
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-runs-tick'))
    assert page.load_factor_library(1, '?factor=momentum', [])[1] == {'ids': []}


def test_an_unknown_factor_in_the_url_is_ignored(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.data, 'factor_rows', lambda: ROWS)
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='url'))
    assert page.load_factor_library(0, '?factor=deleted_factor', [])[1] == {'ids': []}


def test_a_reload_keeps_the_ticked_factors_that_still_exist(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.data, 'factor_rows', lambda: ROWS)
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-runs-tick'))
    current = [{'name': 'value'}, {'name': 'gone'}]
    assert page.load_factor_library(1, '', current)[1] == {'ids': ['value']}


def test_a_selection_still_in_id_form_is_read():
    """Seen in the browser: right after a load the grid reports {'ids': [...]}, which crashed the label."""
    page = _page()
    assert 'Single factor: momentum' in str(page.show_selected_factors({'ids': ['momentum']}))


def test_the_ticked_factors_are_named_above_the_run_button():
    page = _page()
    assert 'No factor ticked' in str(page.show_selected_factors([]))
    assert 'Single factor: value' in str(page.show_selected_factors([{'name': 'value'}]))
    assert 'value + momentum' in str(page.show_selected_factors([{'name': 'value'}, {'name': 'momentum'}]))


def test_the_factors_page_links_to_the_backtest_with_the_factor_selected():
    from investalyze.apps.research import page as research_page
    assert research_page.backtest_link('momentum_12_1') == '/backtest?factor=momentum_12_1'
    assert research_page.backtest_link(None) == '/backtest'


# --- launching ---

def _launch_args(**overrides) -> dict:
    args = dict(_factor_run=1, _timing_run=None, selected_factors=[{'name': 'value'}], fp_start='2016-01-31',
                fp_end='2024-12-31', rebalance='M', quantiles='5', size='1000', fp_holdout='2022-01-01',
                entry='new_252d_high', exit_factor=None, stop=15, exit_on_false=True, max_positions=20, priority=None,
                universe=None, tr_size=500, tr_start='2016-01-01', tr_end='2024-12-31', tr_holdout='2022-01-01')
    return args | overrides


def _call_launch(page, **overrides):
    values = _launch_args(**overrides)
    return page.launch(*values.values())


def test_a_factor_backtest_needs_at_least_one_factor(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-factor-run'))
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    note = _call_launch(page, selected_factors=[])
    assert started == [] and 'Tick at least one factor' in str(note)


def test_pressing_run_starts_the_factor_backtest(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-factor-run'))
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    assert _call_launch(page) is None
    assert started[0][:4] == ['-m', 'investalyze.research', 'backtest', 'value']


def _started_timing_rule(monkeypatch, **overrides) -> tuple[list, object]:
    """Press Run on the timing rule card and return the started argv list and the note shown."""
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-timing-run'))
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    note = _call_launch(page, _factor_run=None, _timing_run=1, **overrides)
    return started, note


def test_pressing_run_on_the_timing_card_starts_that_rule(monkeypatch):
    started, _note = _started_timing_rule(monkeypatch)
    assert started[0][:4] == ['-m', 'investalyze.research', 'timing', 'new_252d_high']


def test_a_timing_rule_needs_an_entry_factor(monkeypatch):
    started, note = _started_timing_rule(monkeypatch, entry=None)
    assert started == [] and 'Pick an entry factor' in str(note)


def test_a_timing_rule_without_an_exit_does_not_start(monkeypatch):
    started, note = _started_timing_rule(monkeypatch, stop=None, exit_on_false=False)
    assert started == [] and 'at least one exit' in str(note)


def test_pressing_run_while_a_job_runs_says_so_instead_of_doing_nothing(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-factor-run'))
    monkeypatch.setattr(page.MANAGER, 'current', object())
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    assert 'already running' in str(_call_launch(page))
    assert started == []


# --- runs and deleting them ---

def test_the_delete_button_is_dead_until_a_run_is_selected():
    page = _page()
    assert page.toggle_delete([]) is True
    assert page.toggle_delete([{'RunId': 'aaa'}]) is False


def test_asking_to_delete_names_the_selected_runs():
    page = _page()
    run = {'Created': '2026-09-13 10:00', 'Signal': 'value', 'StartDate': '2016-01-31', 'EndDate': '2024-12-31',
           'NTrades': 10, 'RunId': 'aaa'}
    opened, text, pending = page.ask_delete(1, [run])
    assert opened is True and 'run aaa' in text and pending == {'ids': ['aaa']}


def test_asking_to_delete_with_nothing_selected_does_nothing():
    with pytest.raises(PreventUpdate):
        _page().ask_delete(1, [])


def test_confirming_starts_the_delete(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    opened, pending, _note = page.resolve_delete(1, None, {'ids': ['aaa']})
    assert (opened, pending) == (False, None)
    assert started == [['-m', 'investalyze.backtest', 'delete', '--run-id', 'aaa']]


def test_a_delete_button_that_reports_no_click_starts_nothing(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    page.resolve_delete(None, None, {'ids': ['aaa']})
    assert started == []


def test_cancelling_leaves_the_run_where_it_is(monkeypatch):
    page = _page()
    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-no'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv) or True)
    opened, pending, _note = page.resolve_delete(None, 1, {'ids': ['aaa']})
    assert (opened, pending, started) == (False, None, [])


def test_deleting_while_a_job_runs_says_nothing_was_deleted(monkeypatch):
    page = _page()
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: False)
    _opened, _pending, note = page.resolve_delete(1, None, {'ids': ['aaa']})
    assert 'nothing was deleted' in str(note)


def test_the_tables_reload_once_a_job_has_finished(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'history', [SimpleNamespace(finished=datetime(2026, 9, 13, 12, 0, 0))])
    assert page.refresh_after_job(1, 0) == '2026-09-13T12:00:00'


def test_a_finished_job_reloads_the_tables_only_once(monkeypatch):
    page = _page()
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'history', [SimpleNamespace(finished=datetime(2026, 9, 13, 12, 0, 0))])
    with pytest.raises(PreventUpdate):
        page.refresh_after_job(2, '2026-09-13T12:00:00')


def test_no_run_selected_shows_an_empty_chart():
    figure, trades, caveats, periods, settings = _page().show_run([], True)
    assert (trades, caveats, periods, settings) == ([], None, [], [])


# --- the retired factor workflow stays gone ---

def test_the_page_has_no_factor_evaluation_controls():
    layout = str(_page().layout())
    for retired in ('bt-btn-evaluate', 'bt-factors-period', 'bt-btn-delete-evaluation', 'bt-res-'):
        assert retired not in layout
    assert 'bt-factor-library' in layout


def test_the_layout_accepts_the_factor_query_parameter():
    """Seen in the browser: /backtest?factor=momentum_12_1 crashed, since Dash passes query parameters to the layout."""
    assert 'bt-factor-library' in str(_page().layout(factor='momentum_12_1'))


def test_the_factors_layout_accepts_query_parameters():
    from investalyze.apps.research import page as research_page
    assert 'rs-library' in str(research_page.layout(anything='x'))
