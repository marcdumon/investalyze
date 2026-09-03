"""Tests for the backtest page's pure shaping and the control panel's backtest argv builders.

The orientation test is the one that matters: the stored coefficient is on raw factor values, so a
factor whose lower values are better reads negative, and a table that ranked on the raw column would
put the working factors at the bottom.
"""

import numpy as np
import pandas as pd
import pytest

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


def test_every_registered_signal_is_offered_by_the_run_panel():
    from investalyze.apps.backtest import page
    from investalyze.backtest import registry

    rendered = str(page._run_panel())
    for name in registry.SIGNALS:
        assert name in rendered


def test_the_run_panel_offers_a_universe_picker():
    from investalyze.apps.backtest import page

    rendered = str(page._run_panel())
    assert 'bt-run-universe' in rendered
    assert 'bt-run-ntop' in rendered
    assert 'bt-run-rankby' in rendered


def test_the_universe_picker_lists_the_saved_universes():
    from investalyze.apps.backtest import page
    from investalyze.apps import universes

    assert page.load_universe_options('bt-run-universe') == universes.list_universes()


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
    _, argv = logic.run_argv('factor_composite', '2015-01-01', '2024-12-31',
                                                    'M', 'my_universe', 300, 'mcap')
    assert '--universe' in argv and 'my_universe' in argv
    assert '--n-top' in argv and '300' in argv
    assert '--rank-by' in argv and 'mcap' in argv


def test_the_factors_button_passes_the_chosen_universe():
    title, argv = logic.factors_argv('2015-01-01', '2024-12-31', 'M',
                                                            'as_reported', 'my_universe', 100, 'ticker')
    assert '--universe' in argv and 'my_universe' in argv
    assert 'my_universe' in title


def test_leaving_the_universe_blank_runs_over_the_whole_database():
    title, argv = logic.run_argv('hold_cash', '2020-01-01', '2020-12-31', 'M')
    assert '--universe' not in argv
    assert 'whole database' in title
