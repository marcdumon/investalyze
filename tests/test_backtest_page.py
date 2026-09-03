"""Tests for the backtest page's pure shaping and the control panel's backtest argv builders.

The orientation test is the one that matters: the stored coefficient is on raw factor values, so a
factor whose lower values are better reads negative, and a table that ranked on the raw column would
put the working factors at the bottom.
"""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from investalyze.analysis.factors import BUILTIN_FACTORS as BUILTIN
from investalyze.apps.backtest import logic


def _ic_table() -> pd.DataFrame:
    """Stored IC rows for one factor read the good way up and one read the other way."""
    return pd.DataFrame({
        'Factor': ['roe', 'vol_252', 'accruals'],
        'Horizon': [21, 21, 21],
        'Periods': [100, 100, 100],
        'MeanIc': [0.05, -0.05, -0.01],
        'StdIc': [0.10, 0.10, 0.10],
        'HitRate': [0.60, 0.40, 0.48],
        'MeanNames': [500, 500, 500],
    })


# --- factor orientation ---

def test_orienting_flips_the_factors_whose_lower_values_are_better():
    oriented = logic.orient_ic(_ic_table()).set_index('Factor')
    assert oriented.loc['roe', 'OrientedIc'] == pytest.approx(0.05)
    assert oriented.loc['vol_252', 'OrientedIc'] == pytest.approx(0.05)     # raw -0.05, lower is better
    assert oriented.loc['accruals', 'OrientedIc'] == pytest.approx(0.01)


def test_the_oriented_table_ranks_the_strongest_factor_first():
    oriented = logic.orient_ic(_ic_table())
    assert oriented.iloc[0]['Factor'] in {'roe', 'vol_252'}
    assert oriented.iloc[-1]['Factor'] == 'accruals'


def test_the_t_statistic_is_the_oriented_mean_over_its_standard_error():
    oriented = logic.orient_ic(_ic_table()).set_index('Factor')
    expected = 0.05 / (0.10 / np.sqrt(100))
    assert oriented.loc['roe', 'TStat'] == pytest.approx(expected)


def test_orienting_an_empty_table_returns_it_unchanged():
    assert logic.orient_ic(pd.DataFrame()).empty


# --- curve shaping ---

def test_rebasing_starts_every_named_column_at_one_hundred():
    curve = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=3),
                          'Equity': [200.0, 220.0, 240.0], 'BenchmarkEquity': [50.0, 55.0, 60.0]})
    rebased = logic.rebase(curve, ['Equity', 'BenchmarkEquity'])
    assert rebased['Equity'].tolist() == pytest.approx([100.0, 110.0, 120.0])
    assert rebased['BenchmarkEquity'].tolist() == pytest.approx([100.0, 110.0, 120.0])


def test_rebasing_a_column_that_starts_with_gaps_uses_its_first_real_value():
    curve = pd.DataFrame({'Equity': [np.nan, 200.0, 220.0]})
    assert logic.rebase(curve, ['Equity'])['Equity'].tolist()[1:] == pytest.approx([100.0, 110.0])


def test_rebasing_an_all_empty_column_leaves_it_empty():
    curve = pd.DataFrame({'BenchmarkEquity': [np.nan, np.nan]})
    assert logic.rebase(curve, ['BenchmarkEquity'])['BenchmarkEquity'].isna().all()


def test_rebasing_ignores_a_column_the_curve_does_not_have():
    curve = pd.DataFrame({'Equity': [100.0, 110.0]})
    assert list(logic.rebase(curve, ['Missing']).columns) == ['Equity']


def test_the_drawdown_band_is_zero_at_a_new_high():
    equity = pd.Series([100.0, 110.0, 99.0, 120.0])
    drawdown = logic.drawdown_from_equity(equity)
    assert drawdown.tolist() == pytest.approx([0.0, 0.0, 99 / 110 - 1, 0.0])


def test_the_run_summary_names_the_signal_and_the_span():
    run = pd.Series({'Signal': 'factor_composite', 'NTickers': 500, 'StartDate': '2015-01-01',
                     'EndDate': '2024-12-31', 'Benchmark': '^NDX'})
    summary = logic.run_summary_text(run)
    assert 'factor_composite' in summary and '500' in summary and '^NDX' in summary


# --- the control panel buttons ---

def test_the_factors_button_builds_the_cli_argv():
    title, argv = logic.factors_argv('2015-01-01', '2024-12-31', 'M', 'as_reported')
    assert argv[:3] == ['-m', 'investalyze.backtest', 'factors']
    assert '--start' in argv and '2015-01-01' in argv
    assert 'as_reported' in title


def test_the_evaluate_button_builds_the_cli_argv():
    title, argv = logic.evaluate_argv(63, 'restated')
    assert argv[:3] == ['-m', 'investalyze.backtest', 'evaluate']
    assert '63' in argv
    assert 'restated' in title


def test_the_run_button_builds_the_cli_argv():
    title, argv = logic.run_argv('sma_cross', '2018-01-01', '2020-12-31', 'W')
    assert argv[:3] == ['-m', 'investalyze.backtest', 'run']
    assert 'sma_cross' in argv and '--freq' in argv and 'W' in argv
    assert 'sma_cross' in title


def test_every_registered_signal_is_offered_by_the_backtest_card():
    from investalyze.apps.backtest import page
    from investalyze.backtest import registry

    rendered = str(page._backtest_panel())
    for name in registry.SIGNALS:
        assert name in rendered


def test_each_card_owns_its_own_fields():
    """Duplicated on purpose: editing the research range must not stale a backtest."""
    from investalyze.apps.backtest import page

    research, backtest = str(page._research_panel()), str(page._backtest_panel())
    for field in ('start', 'end', 'universe', 'vintage'):
        assert f'bt-res-{field}' in research
        assert f'bt-run-{field}' in backtest
        assert f'bt-run-{field}' not in research
        assert f'bt-res-{field}' not in backtest


def test_the_research_card_carries_only_what_an_evaluation_needs():
    from investalyze.apps.backtest import page

    research = str(page._research_panel())
    assert 'bt-res-horizon' in research
    assert 'signal' not in research.lower().replace('bt-res', '')
    assert 'bt-res-freq' not in research
    assert 'bt-res-ntop' not in research


def test_the_backtest_card_carries_the_sizing_an_evaluation_has_no_use_for():
    from investalyze.apps.backtest import page

    backtest = str(page._backtest_panel())
    assert 'bt-run-ntop' in backtest
    assert 'bt-run-rankby' in backtest
    assert 'bt-run-horizon' not in backtest


def test_both_universe_pickers_list_the_saved_universes():
    from investalyze.apps import universes
    from investalyze.apps.backtest import page

    saved = universes.list_universes()
    assert page.load_universe_options(0) == (saved, saved)


def test_the_backtest_page_layout_builds():
    from investalyze.apps.backtest import page

    assert page.layout() is not None


# --- the factor editor ---

def _fields(**overrides) -> dict:
    """Form values for a valid expression factor, with overrides applied."""
    base = dict(kind='expression', expression='0.5 * roe + 0.5 * roa', numerator=None,
                denominator=None, higher_is_better=True, description='a blend')
    base |= overrides
    return logic.definition_fields(base['kind'], base['expression'], base['numerator'],
                                   base['denominator'], base['higher_is_better'], base['description'])


def test_definition_rows_describe_each_factor():
    from investalyze.analysis import custom_factors

    defined = custom_factors.parse("[factors.blend]\nexpression = 'roe + roa'\n"
                                   "higher_is_better = false\ndescription = 'mine'", BUILTIN)
    row = logic.definition_rows(defined)[0]
    assert row == {'name': 'blend', 'kind': 'expression', 'formula': 'roe + roa',
                   'direction': 'lower', 'description': 'mine'}


def test_a_ratio_definition_shows_its_two_terms():
    from investalyze.analysis import custom_factors

    defined = custom_factors.parse("[factors.turn]\nnumerator = 'Revenue'\ndenominator = 'Total Assets'",
                                   BUILTIN)
    assert logic.definition_rows(defined)[0]['formula'] == 'Revenue / Total Assets'


def test_a_valid_form_reports_no_problem():
    assert logic.validate_definition('my_blend', _fields(), {}) is None


def test_a_form_with_no_name_is_refused():
    assert logic.validate_definition('  ', _fields(), {}) == 'give the factor a name'


def test_a_name_already_defined_is_refused():
    from investalyze.analysis import custom_factors

    existing = custom_factors.parse("[factors.taken]\nexpression = 'roe'", BUILTIN)
    problem = logic.validate_definition('taken', _fields(), existing)
    assert 'already defined' in problem


def test_a_form_referencing_an_unknown_column_is_refused():
    problem = logic.validate_definition('x', _fields(expression='roe + nonsense'), {})
    assert 'references unknown column' in problem


def test_a_ratio_form_needs_both_terms():
    fields = _fields(kind='ratio', expression=None, numerator='Revenue', denominator=None)
    assert 'needs either an expression' in logic.validate_definition('x', fields, {})


def test_the_preview_is_the_toml_that_would_be_written():
    preview = logic.preview_definition('my_blend', _fields())
    assert preview.startswith('[factors.my_blend]')
    assert "expression = '0.5 * roe + 0.5 * roa'" in preview
    assert 'higher_is_better = true' in preview


def test_the_form_switches_fields_with_the_chosen_kind():
    from investalyze.analysis import custom_factors
    from investalyze.apps.screener import page as screener_page

    expression_style, ratio_style = screener_page.switch_definition_form(custom_factors.EXPRESSION)
    assert expression_style == {'display': 'block'} and ratio_style == {'display': 'none'}
    expression_style, ratio_style = screener_page.switch_definition_form(custom_factors.RATIO)
    assert expression_style == {'display': 'none'} and ratio_style == {'display': 'block'}


def test_the_factor_editor_lives_on_the_screener_page():
    from investalyze.apps.screener import page as screener_page

    rendered = str(screener_page.define_panel())
    assert 'bt-def-name' in rendered
    assert 'bt-btn-save-factor' in rendered


def test_the_backtest_page_no_longer_carries_the_editor():
    from investalyze.apps.backtest import page

    assert 'bt-def-name' not in str(page.layout())


def test_the_control_panel_no_longer_carries_a_backtest_card():
    from investalyze.apps.control_panel import page as control_panel

    assert not hasattr(control_panel, 'backtest_card')
    assert 'cp-btn-bt-run' not in str(control_panel.layout())


# --- the universe picker on the control panel ---

def test_the_run_button_passes_the_chosen_universe_and_sizing():
    _, argv = logic.run_argv('factor_composite', '2015-01-01', '2024-12-31', 'M', 'my_universe', 300, 'mcap')
    assert '--universe' in argv and 'my_universe' in argv
    assert '--n-top' in argv and '300' in argv
    assert '--rank-by' in argv and 'mcap' in argv


def test_the_factors_button_passes_the_chosen_universe():
    title, argv = logic.factors_argv('2015-01-01', '2024-12-31', 'M', 'as_reported', 'my_universe', 100, 'ticker')
    assert '--universe' in argv and 'my_universe' in argv
    assert 'my_universe' in title


def test_leaving_the_universe_blank_runs_over_the_whole_database():
    title, argv = logic.run_argv('hold_cash', '2020-01-01', '2020-12-31', 'M')
    assert '--universe' not in argv
    assert 'whole database' in title


# --- deleting a run ---

def _selected(**overrides) -> list[dict]:
    """The grid's selection payload for one stored run, with overrides applied."""
    row = {'RunId': 'abc123def456', 'Signal': 'sma_cross', 'StartDate': '2015-01-02',
           'EndDate': '2020-12-31', 'NTrades': 2538, 'Created': '2026-09-03 08:15'}
    return [row | overrides]


def test_the_delete_button_builds_the_cli_argv():
    title, argv = logic.delete_argv(['abc123def456'])
    assert argv == ['-m', 'investalyze.backtest', 'delete', '--run-id', 'abc123def456']
    assert 'abc123def456' in title


def test_deleting_several_runs_builds_one_command_naming_them_all():
    title, argv = logic.delete_argv(['aaa', 'bbb', 'ccc'])
    assert argv == ['-m', 'investalyze.backtest', 'delete', '--run-id', 'aaa', 'bbb', 'ccc']
    assert '3 runs' in title


def test_the_runs_panel_offers_a_delete_button():
    from investalyze.apps.backtest import page

    assert 'bt-btn-delete-run' in str(page.layout())


def test_the_delete_button_is_dead_until_a_run_is_selected():
    from investalyze.apps.backtest import page

    assert page.toggle_delete([]) is True
    assert page.toggle_delete(_selected()) is False


def test_the_confirmation_names_the_run_and_what_goes_with_it():
    from investalyze.apps.backtest import page

    opened, text, pending = page.ask_delete(1, _selected())
    assert opened is True
    assert pending == {'kind': 'runs', 'ids': ['abc123def456']}
    assert 'abc123def456' in text
    assert 'sma_cross' in text
    assert '2,538 fills' in text
    assert '2026-09-03 08:15' in text


def test_the_confirmation_totals_the_fills_of_every_selected_run():
    from investalyze.apps.backtest import page

    selection = _selected() + _selected(RunId='def456abc123', NTrades=1_000)
    opened, text, pending = page.ask_delete(1, selection)
    assert pending['ids'] == ['abc123def456', 'def456abc123']
    assert 'these 2 runs' in text
    assert '3,538 fills in total' in text
    assert 'def456abc123' in text


def test_confirming_starts_the_delete_job(monkeypatch):
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    opened, pending = page.resolve_delete(1, None, {'kind': 'runs', 'ids': ['abc123def456']})
    assert (opened, pending) == (False, None)
    assert started == [['-m', 'investalyze.backtest', 'delete', '--run-id', 'abc123def456']]


def test_cancelling_leaves_the_run_where_it_is(monkeypatch):
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-no'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    opened, pending = page.resolve_delete(None, 1, {'kind': 'runs', 'ids': ['abc123def456']})
    assert (opened, pending) == (False, None)
    assert started == []


def test_the_tables_reload_once_a_job_has_finished(monkeypatch):
    from investalyze.apps.backtest import page

    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'history', [SimpleNamespace(finished=datetime(2026, 9, 3, 12, 0, 0))])
    assert page.refresh_after_job(1, 0) == '2026-09-03T12:00:00'


def test_a_finished_job_reloads_the_tables_once_and_not_on_every_poll(monkeypatch):
    from investalyze.apps.backtest import page

    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'history', [SimpleNamespace(finished=datetime(2026, 9, 3, 12, 0, 0))])
    with pytest.raises(PreventUpdate):
        page.refresh_after_job(2, '2026-09-03T12:00:00')


def test_the_tables_are_left_alone_while_a_job_is_still_running(monkeypatch):
    from investalyze.apps.backtest import page

    monkeypatch.setattr(page.MANAGER, 'current', SimpleNamespace(title='backtest run'))
    with pytest.raises(PreventUpdate):
        page.refresh_after_job(1, 0)


def test_the_runs_grid_offers_a_checkbox_on_every_row():
    from investalyze.apps.backtest import page

    assert page._MULTI_SELECT['mode'] == 'multiRow'
    assert page._MULTI_SELECT['checkboxes'] is True
    assert page._MULTI_SELECT['headerCheckbox'] is True


def test_the_run_columns_are_sized_to_their_contents():
    from investalyze.apps.backtest import page

    grid = page._grid('bt-runs', logic.RUN_COLUMNS, 220, selectable=True)
    assert grid.columnSize == 'autoSize'
    assert not any('width' in column for column in logic.RUN_COLUMNS)


def test_the_run_table_leads_with_when_the_run_was_made_and_ends_with_its_id():
    fields = [column['field'] for column in logic.RUN_COLUMNS]
    assert fields[0] == 'Created'
    assert fields[-1] == 'RunId'


# --- following the theme ---

def test_a_figure_carries_the_template_of_the_mode_the_switch_is_on():
    from investalyze.apps import theming

    assert theming.figure_layout(True)['template'] == 'plotly_dark'
    assert theming.figure_layout(False)['template'] == 'plotly_white'


def test_a_figure_lets_the_page_behind_it_show_through():
    from investalyze.apps import theming

    for dark in (True, False):
        layout = theming.figure_layout(dark)
        assert layout['paper_bgcolor'] == 'rgba(0,0,0,0)'
        assert layout['plot_bgcolor'] == 'rgba(0,0,0,0)'


def test_the_empty_chart_follows_the_theme():
    from investalyze.apps.backtest import page

    assert page._empty_figure('select a run', True).layout.template.layout.paper_bgcolor is not None
    assert page._empty_figure('select a run', False).layout.paper_bgcolor == 'rgba(0,0,0,0)'


def test_the_run_chart_follows_the_theme():
    from investalyze.apps.backtest import page

    curve = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=4),
                          'Equity': [100.0, 101.0, 102.0, 103.0],
                          'BenchmarkEquity': [100.0, 100.5, 101.0, 101.5]})
    drawdown = logic.drawdown_from_equity(curve['Equity'])
    run = {'Signal': 'sma_cross', 'Benchmark': '^NDX'}
    dark = page._curve_figure(curve, drawdown, run, True)
    light = page._curve_figure(curve, drawdown, run, False)
    assert dark.layout.template is not light.layout.template
    assert light.layout.paper_bgcolor == 'rgba(0,0,0,0)'


def test_the_run_chart_is_redrawn_when_the_theme_switch_moves():
    """The switch is an input of the chart callback, so a toggle redraws rather than leaves it behind."""
    from dash._callback import GLOBAL_CALLBACK_MAP

    from investalyze.apps.backtest import page  # noqa: F401  registers the callback

    key = next(name for name in GLOBAL_CALLBACK_MAP if 'bt-curve.figure' in name)
    assert 'theme-switch' in [item['id'] for item in GLOBAL_CALLBACK_MAP[key]['inputs']]


def test_the_quality_evidence_is_redrawn_when_the_theme_switch_moves():
    """It used to read the switch as State, which left an open chart in the mode it was drawn in."""
    from dash._callback import GLOBAL_CALLBACK_MAP

    from investalyze.apps.data_quality import page as quality_page  # noqa: F401  registers the callback

    key = next(name for name in GLOBAL_CALLBACK_MAP if 'dq-evidence.children' in name
               and 'allow_duplicate' not in name)
    assert 'theme-switch' in [item['id'] for item in GLOBAL_CALLBACK_MAP[key]['inputs']]


def test_a_delete_button_that_reports_no_click_starts_nothing(monkeypatch):
    """A remounted button fires the callback with no clicks; deleting must need a real one."""
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    assert page.resolve_delete(None, None, {'kind': 'runs', 'ids': ['abc']}) == (False, None)
    assert started == []


def test_the_selection_is_cleared_once_the_dialog_closes(monkeypatch):
    """Clearing it stops a payload from a closed dialog being spent by a later firing."""
    from investalyze.apps.backtest import page

    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-no'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: None)
    _opened, pending = page.resolve_delete(None, 1, {'kind': 'runs', 'ids': ['abc', 'def']})
    assert pending is None


def test_the_evaluate_button_passes_the_chosen_universe():
    title, argv = logic.evaluate_argv(21, 'as_reported', 'my_universe')
    assert '--universe' in argv and 'my_universe' in argv
    assert 'my_universe' in title


def test_evaluating_without_a_universe_covers_the_whole_database():
    title, argv = logic.evaluate_argv(21, 'as_reported')
    assert '--universe' not in argv
    assert 'whole database' in title


def test_the_factors_tab_offers_the_universes_an_evaluation_was_stored_for():
    from investalyze.apps.backtest import page

    assert 'bt-factors-universe' in str(page.layout())




# --- what a card says a press will do ---

def _gaps(**overrides) -> dict:
    """A request the stored panel can answer in full, with overrides applied."""
    base = {'requested_dates': 120, 'missing_dates': 0, 'n_tickers': 7, 'missing_factors': [],
            'changed_factors': [], 'seconds_estimate': None, 'has_panel': True,
            'build_frequency': 'M', 'panel_frequency': 'M', 'rebalance': 'M'}
    return base | overrides


def test_a_covered_evaluation_says_it_reads_the_stored_panel():
    ready, sentence = logic.research_plan(_gaps(), 'solar')
    assert ready is True
    assert 'from the stored panel' in sentence
    assert '7 names in solar' in sentence


def test_an_evaluation_missing_dates_says_what_it_builds_and_how_long():
    ready, sentence = logic.research_plan(_gaps(missing_dates=42, seconds_estimate=15.0), 'solar')
    assert ready is False
    assert 'Builds the as reported panel: 42 monthly dates for 7 names in solar' in sentence
    assert 'about 15 seconds' in sentence
    assert '42 of the 120 monthly dates it needs do not exist yet' in sentence


def test_a_vintage_with_no_panel_says_so_rather_than_counting_to_itself():
    _ready, sentence = logic.research_plan(_gaps(missing_dates=120), 'solar', 'restated')
    assert 'Builds the restated panel: 120 monthly dates' in sentence
    assert 'holds nothing for this vintage' in sentence
    assert '120 of the 120' not in sentence


def test_a_redefined_factor_makes_a_card_ask_for_a_rebuild():
    """The dates are all there; only the fingerprint says the values came from another formula."""
    ready, sentence = logic.research_plan(_gaps(changed_factors=['quality_blend']), 'solar')
    assert ready is False
    assert 'Quality_blend was redefined since the last build' in sentence


def test_a_factor_with_no_values_makes_a_card_ask_for_a_rebuild():
    ready, sentence = logic.research_plan(_gaps(missing_factors=['my_blend']), None)
    assert ready is False
    assert 'no values for my_blend' in sentence


def test_a_timing_signal_never_mentions_the_panel():
    """It has no use for one, so naming it would only invite the question of whether to build it."""
    ready, sentence = logic.backtest_plan(_gaps(), 'timing', 'sma_cross', 'solar')
    assert ready is True
    assert sentence == 'sma_cross reads prices only. Nothing to build.'


def test_a_covered_backtest_names_the_signal_it_will_run():
    ready, sentence = logic.backtest_plan(_gaps(), 'cross_sectional', 'factor_composite', None)
    assert ready is True
    assert 'factor_composite' in sentence


def test_a_backtest_past_the_panel_is_not_ready():
    ready, _ = logic.backtest_plan(_gaps(missing_dates=9), 'cross_sectional', 'factor_composite', None)
    assert ready is False


# --- the build travels with the command, never on its own ---

def test_neither_command_builds_unless_asked():
    _t, evaluate = logic.evaluate_argv(21, 'as_reported', 'solar', '2015-01-01', '2024-12-31')
    _t2, run = logic.run_argv('factor_composite', '2015-01-01', '2024-12-31', 'M')
    assert '--build-panel' not in evaluate
    assert '--build-panel' not in run


def test_a_confirmed_build_travels_with_the_command_and_shows_in_its_title():
    title, argv = logic.run_argv('factor_composite', '2015-01-01', '2024-12-31', 'M',
                                 build_panel=True)
    assert '--build-panel' in argv
    assert 'build panel, then run' in title


def test_a_confirmed_build_travels_with_an_evaluation_too():
    title, argv = logic.evaluate_argv(21, 'as_reported', build_panel=True)
    assert '--build-panel' in argv
    assert 'build panel, then evaluate' in title


def test_a_rebalance_finer_than_the_panel_runs_and_says_what_it_trades_on():
    """As-of reading makes this work, so it is a note about staleness, not a reason to rebuild."""
    ready, sentence = logic.backtest_plan(_gaps(rebalance='W'), 'cross_sectional',
                                          'factor_composite', None)
    assert ready is True
    assert 'panel updates monthly' in sentence


def test_a_rebalance_matching_the_panel_says_nothing_extra():
    _ready, sentence = logic.backtest_plan(_gaps(), 'cross_sectional', 'factor_composite', None)
    assert 'panel updates' not in sentence


def test_a_press_that_needs_a_build_launches_one_command_that_does_both(monkeypatch):
    """The line under the button already said this would happen, so it happens on one press."""
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-run'))
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    monkeypatch.setattr(page, '_gaps_for', lambda *args: _gaps(missing_dates=42))

    page.launch(None, 1, '2015-01-01', '2024-12-31', None, 'as_reported', 21, 'M',
                'factor_composite', '2015-01-01', '2024-12-31', 'M', None, 500, 'liquidity',
                'as_reported', 'M')

    assert len(started) == 1
    assert '--build-panel' in started[0]


def test_a_press_that_needs_nothing_built_leaves_the_flag_off(monkeypatch):
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-btn-evaluate'))
    monkeypatch.setattr(page.MANAGER, 'current', None)
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    monkeypatch.setattr(page, '_gaps_for', lambda *args: _gaps())

    page.launch(1, None, '2015-01-01', '2024-12-31', 'solar', 'as_reported', 21, 'M',
                'factor_composite', '2015-01-01', '2024-12-31', 'M', None, 500, 'liquidity',
                'as_reported', 'M')

    assert '--build-panel' not in started[0]


def test_a_running_job_offers_a_way_out_of_it(monkeypatch):
    from investalyze.apps.backtest import page

    monkeypatch.setattr(page.MANAGER, 'current', SimpleNamespace(title='backtest run', lines=[]))
    assert 'bt-btn-cancel' in str(page.show_job_status(1))


def test_cancelling_stops_the_running_job(monkeypatch):
    from investalyze.apps.backtest import page

    cancelled = []
    monkeypatch.setattr(page.MANAGER, 'cancel', lambda: cancelled.append(True))
    page.cancel_job(1)
    assert cancelled == [True]


def test_a_cancel_button_that_reports_no_click_stops_nothing(monkeypatch):
    from investalyze.apps.backtest import page

    cancelled = []
    monkeypatch.setattr(page.MANAGER, 'cancel', lambda: cancelled.append(True))
    page.cancel_job(None)
    assert cancelled == []


# --- throwing a stored evaluation away ---

def _scope(**overrides) -> dict:
    """One stored evaluation, shaped as `get_evaluation_scopes` returns it."""
    base = {'vintage': 'as_reported', 'universe': 'unrecorded', 'factors': 29, 'ic_rows': 9128,
            'quantile_rows': 43895, 'first': '2000-01-31', 'last': '2026-09-02'}
    return base | overrides


def test_deleting_an_evaluation_builds_the_cli_argv():
    title, argv = logic.delete_evaluation_argv('as_reported', 'unrecorded')
    assert argv == ['-m', 'investalyze.backtest', 'delete-evaluation', '--vintage', 'as_reported',
                    '--universe', 'unrecorded']
    assert 'unrecorded' in title


def test_the_confirmation_names_the_evaluation_and_what_it_holds():
    body = logic.delete_evaluation_prompt(_scope())
    assert 'as reported evaluation stored under unrecorded' in body
    assert '29 factors, 2000-01-31 to 2026-09-02' in body
    assert '9,128 IC rows and 43,895 quantile rows' in body
    assert 'rebuilds it' in body


def test_the_delete_evaluation_button_is_dead_until_one_is_picked():
    from investalyze.apps.backtest import page

    assert page.toggle_delete_evaluation(None) is True
    assert page.toggle_delete_evaluation('unrecorded') is False


def test_confirming_starts_the_evaluation_delete(monkeypatch):
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv))
    pending = {'kind': 'evaluation', 'vintage': 'as_reported', 'universe': 'unrecorded'}

    assert page.resolve_delete(1, None, pending) == (False, None)
    assert started == [['-m', 'investalyze.backtest', 'delete-evaluation', '--vintage',
                        'as_reported', '--universe', 'unrecorded']]


def test_one_dialog_tells_the_two_kinds_of_delete_apart(monkeypatch):
    from investalyze.apps.backtest import page

    started = []
    monkeypatch.setattr(page, 'ctx', SimpleNamespace(triggered_id='bt-delete-yes'))
    monkeypatch.setattr(page.MANAGER, 'start', lambda title, argv: started.append(argv[2]))

    page.resolve_delete(1, None, {'kind': 'runs', 'ids': ['abc']})
    page.resolve_delete(1, None, {'kind': 'evaluation', 'vintage': 'as_reported', 'universe': 'x'})

    assert started == ['delete', 'delete-evaluation']


# --- both cadences are the card's, never the file's ---

def test_both_cards_carry_a_panel_cadence():
    """The research card used to have none, so its builds took a value from backtest.toml."""
    from investalyze.apps.backtest import page

    assert 'bt-res-panel-freq' in str(page._research_panel())
    assert 'bt-run-panel-freq' in str(page._backtest_panel())


def test_only_the_backtest_card_carries_a_rebalance():
    from investalyze.apps.backtest import page

    assert 'bt-run-freq' in str(page._backtest_panel())
    assert 'bt-res-freq' not in str(page._research_panel())


def test_the_two_frequencies_reach_a_run_separately():
    _title, argv = logic.run_argv('factor_composite', '2015-01-01', '2024-12-31', 'W',
                                  panel_freq='Q')
    assert argv[argv.index('--freq') + 1] == 'W'
    assert argv[argv.index('--panel-freq') + 1] == 'Q'


def test_an_evaluation_states_the_cadence_it_would_build_at():
    _title, argv = logic.evaluate_argv(21, 'as_reported', panel_freq='W')
    assert argv[argv.index('--panel-freq') + 1] == 'W'


def test_the_build_wording_uses_the_cadence_being_built_not_the_one_stored():
    _ready, sentence = logic.research_plan(_gaps(missing_dates=42, build_frequency='W',
                                                 panel_frequency='M'), None)
    assert '42 weekly dates' in sentence
