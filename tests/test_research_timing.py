"""Tests for timing rule holdings: entries, exits, the trailing stop and the position cap."""

import numpy as np
import pytest

from investalyze.research import timing


def _run(entry, close, **overrides):
    settings = dict(exit_signal=None, exit_on_entry_false=True, trailing_stop=None, max_positions=10, priority=None)
    return timing.holdings(np.array(entry, dtype=bool), np.array(close, dtype=float), **(settings | overrides))


def test_a_name_is_held_while_its_entry_rule_is_true():
    held, _weights = _run([[True], [True], [False], [False]], [[10], [11], [12], [13]])
    assert held[:, 0].tolist() == [True, True, False, False]


def test_weights_are_set_only_when_holdings_change():
    _held, weights = _run([[True], [True], [False]], [[10], [11], [12]], max_positions=4)
    assert weights[0, 0] == pytest.approx(0.25)
    assert np.isnan(weights[1]).all()
    assert weights[2, 0] == 0.0


def test_without_exit_on_entry_false_a_name_stays_until_another_exit():
    held, _weights = _run([[True], [False], [False]], [[10], [11], [12]], exit_on_entry_false=False)
    assert held[:, 0].tolist() == [True, True, True]


def test_an_exit_rule_closes_the_position():
    exit_signal = np.array([[False], [False], [True], [False]])
    held, _weights = _run([[True], [False], [False], [False]], [[10], [11], [12], [13]], exit_on_entry_false=False,
                          exit_signal=exit_signal)
    assert held[:, 0].tolist() == [True, True, False, False]


def test_a_trailing_stop_exits_after_a_drop_from_the_highest_close_since_entry():
    closes = [[100], [120], [110], [101]]
    held, _weights = _run([[True], [False], [False], [False]], closes, exit_on_entry_false=False, trailing_stop=0.15)
    # the high since entry is 120, so the stop is 102; 110 holds and 101 exits
    assert held[:, 0].tolist() == [True, True, True, False]


def test_the_position_cap_takes_the_highest_priority_entries():
    entry = [[True, True, True]]
    priority = np.array([[1.0, 3.0, 2.0]])
    held, weights = _run(entry, [[10, 10, 10]], max_positions=2, priority=priority)
    assert held[0].tolist() == [False, True, True]
    assert weights[0].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_a_full_book_takes_no_new_entries_until_a_slot_frees():
    entry = [[True, False], [True, True], [False, True]]
    held, _weights = _run(entry, [[10, 10], [10, 10], [10, 10]], max_positions=1)
    assert held.tolist() == [[True, False], [True, False], [False, True]]


def test_a_name_without_a_price_is_not_entered():
    held, _weights = _run([[True]], [[np.nan]])
    assert not held[0, 0]


def test_a_name_that_exits_is_not_re_entered_on_the_same_session():
    exit_signal = np.array([[False], [True]])
    held, _weights = _run([[True], [True]], [[10], [11]], exit_signal=exit_signal)
    assert held[:, 0].tolist() == [True, False]


def test_the_label_names_the_entry_and_exits():
    rule = timing.TimingRule('new_252d_high', exit_on_entry_false=False, trailing_stop=0.15)
    assert rule.label == 'new_252d_high, trailing stop 15%'
    rule = timing.TimingRule('sma_50_above_sma_200', exit_factor='close_above_prior_high')
    assert rule.label == 'sma_50_above_sma_200, exit on entry false or close_above_prior_high'


def test_a_rule_without_any_exit_is_refused():
    with pytest.raises(ValueError, match='at least one exit'):
        timing.TimingRule('new_252d_high', exit_on_entry_false=False)


def test_a_trailing_stop_must_be_a_fraction_below_one():
    with pytest.raises(ValueError, match='between 0 and 1'):
        timing.TimingRule('new_252d_high', trailing_stop=15)


def test_a_rule_reads_back_from_the_command_it_writes():
    from investalyze.research import __main__ as cli
    rule = timing.TimingRule('new_252d_high', exit_factor='close_above_prior_high', exit_on_entry_false=False,
                             trailing_stop=0.15, max_positions=12, priority_factor='momentum_12_1')
    assert timing.TimingRule.from_arguments(cli._parser().parse_args(['timing', *rule.arguments()])) == rule


def test_stored_timing_run_ids_are_unchanged():
    """A re-run of a stored configuration must replace its run, so the id stays what earlier versions made."""
    from investalyze.backtest import config as backtest_config
    from investalyze.research import factors
    library = {'value': factors.Factor(name='value', expression='income_net_income / market_cap'),
               'new_high': factors.Factor(name='new_high', expression='adj_close >= max(adj_close, 252)'),
               'breakout': factors.Factor(name='breakout', expression='close > lag(high, 1)')}
    cfg = backtest_config.read()
    rule = timing.TimingRule('new_high', exit_factor='breakout', exit_on_entry_false=False, trailing_stop=0.15)
    params = timing.run_params(rule, library, universe=None, universe_size=500, start='2016-01-01', end='2024-12-31',
                               holdout=None)
    assert timing.run_id(params, cfg) == '92d33d31d233'
    rule = timing.TimingRule('new_high', max_positions=10, priority_factor='value')
    params = timing.run_params(rule, library, universe='solar', universe_size=500, start='2016-01-01', end='2024-12-31',
                               holdout='2022-01-01')
    assert timing.run_id(params, cfg) == 'e078e11b609c'
