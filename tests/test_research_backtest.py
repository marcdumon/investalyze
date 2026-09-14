"""Tests for turning factor values into backtest target weights and summarising the result."""

import numpy as np
import pandas as pd
import pytest

from investalyze.research import backtest

SESSIONS = pd.bdate_range('2020-01-01', periods=6)
TICKERS = np.array(['AAA', 'BBB', 'CCC', 'DDD', 'EEE'])


def _scores(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows, index=TICKERS).T.rename_axis('Date')


def test_only_rebalance_sessions_carry_targets():
    scores = _scores({SESSIONS[1]: [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=1.0)
    assert np.isnan(weights[0]).all() and np.isnan(weights[2]).all()
    assert not np.isnan(weights[1]).any()


def test_the_top_quantile_is_held_equal_weight():
    scores = _scores({SESSIONS[1]: [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=1.0)
    assert weights[1].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_two_quantiles_hold_the_upper_half():
    scores = _scores({SESSIONS[1]: [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=2, max_weight=1.0)
    # quantile_buckets places the median name in the upper half, so three names are held
    assert weights[1].tolist() == pytest.approx([0.0, 0.0, 1 / 3, 1 / 3, 1 / 3])


def test_a_lower_is_better_factor_holds_the_lowest_values():
    scores = _scores({SESSIONS[1]: [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=1.0,
                                      higher_is_better=False)
    assert weights[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_weights_are_capped_and_the_rest_stays_in_cash():
    scores = _scores({SESSIONS[1]: [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=0.05)
    assert weights[1].max() == pytest.approx(0.05)


def test_the_top_quantile_matches_the_ic_analysis_buckets():
    """The backtest must hold exactly the names IC analysis put in the top quantile."""
    from investalyze.backtest import evaluation
    rng = np.random.default_rng(1)
    scores = _scores({SESSIONS[1]: rng.normal(size=5)})
    top = evaluation.quantile_buckets(scores, n_quantiles=2).iloc[0] == 1
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=2, max_weight=1.0)
    assert (weights[1] > 0).tolist() == top.tolist()


def test_a_name_missing_from_the_scores_is_sold():
    scores = pd.DataFrame({'AAA': [1.0], 'BBB': [2.0]}, index=pd.DatetimeIndex([SESSIONS[1]], name='Date'))
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=2, max_weight=1.0)
    assert weights[1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_a_date_with_no_scores_is_not_a_rebalance():
    scores = _scores({SESSIONS[1]: [np.nan] * 5})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=1.0)
    assert np.isnan(weights[1]).all()


def test_a_score_date_outside_the_calendar_is_ignored():
    scores = _scores({pd.Timestamp('2019-06-03'): [1.0, 2.0, 3.0, 4.0, 5.0]})
    weights = backtest.target_weights(scores, SESSIONS, TICKERS, n_quantiles=5, max_weight=1.0)
    assert np.isnan(weights).all()


def test_period_stats_split_the_equity_curve_at_the_holdout_date():
    dates = pd.bdate_range('2021-01-01', '2023-12-29')
    equity = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    table = backtest.period_stats(equity, None, holdout_start=pd.Timestamp('2023-01-01'))
    assert table['period'].tolist() == ['Full sample', 'In-sample', 'Out-of-sample']
    assert table.set_index('period').loc['Out-of-sample', 'start'] == '2023-01-02'


def test_a_composite_averages_oriented_ranks():
    frame = pd.DataFrame({'Ticker': ['AAA', 'BBB', 'CCC'], 'Date': [SESSIONS[1]] * 3,
                          'value': [1.0, 2.0, 3.0], 'cheap': [3.0, 2.0, 1.0]})
    from investalyze.research import factors
    library = {'value': factors.Factor(name='value', expression='x'),
               'cheap': factors.Factor(name='cheap', expression='x', higher_is_better=False)}
    scores = backtest.composite_scores(frame, ['value', 'cheap'], library)
    # cheap is lower-is-better, so both factors agree that CCC is best
    assert scores.iloc[0].idxmax() == 'CCC'


def test_a_composite_skips_names_missing_a_factor():
    frame = pd.DataFrame({'Ticker': ['AAA', 'BBB', 'CCC'], 'Date': [SESSIONS[1]] * 3,
                          'value': [1.0, 2.0, 3.0], 'quality': [1.0, np.nan, 2.0]})
    from investalyze.research import factors
    library = {'value': factors.Factor(name='value', expression='x'), 'quality': factors.Factor(name='quality', expression='x')}
    scores = backtest.composite_scores(frame, ['value', 'quality'], library)
    assert np.isnan(scores.iloc[0]['BBB'])


def test_a_single_lower_is_better_factor_is_negated():
    frame = pd.DataFrame({'Ticker': ['AAA', 'BBB'], 'Date': [SESSIONS[1]] * 2, 'risk': [1.0, 2.0]})
    from investalyze.research import factors
    library = {'risk': factors.Factor(name='risk', expression='x', higher_is_better=False)}
    assert backtest.composite_scores(frame, ['risk'], library).iloc[0].idxmax() == 'AAA'


def test_the_run_id_changes_when_a_factor_is_redefined():
    from investalyze.backtest import config as backtest_config
    from investalyze.research import factors
    cfg = backtest_config.read()
    before = {'value': factors.Factor(name='value', expression='revenue')}
    after = {'value': factors.Factor(name='value', expression='revenue * 2')}
    portfolio = backtest.FactorPortfolio(names=('value',))
    settings = dict(universe_size=1000, start='2016-01-31', end='2024-12-31', holdout=None)
    assert backtest.run_id(backtest.run_params(portfolio, before, **settings), cfg) != \
        backtest.run_id(backtest.run_params(portfolio, after, **settings), cfg)


def test_the_run_id_is_stable_for_identical_settings():
    from investalyze.backtest import config as backtest_config
    from investalyze.research import factors
    cfg = backtest_config.read()
    library = {'value': factors.Factor(name='value', expression='revenue')}
    portfolio = backtest.FactorPortfolio(names=('value',))
    settings = dict(universe_size=1000, start='2016-01-31', end='2024-12-31', holdout=None)
    assert backtest.run_id(backtest.run_params(portfolio, library, **settings), cfg) == \
        backtest.run_id(backtest.run_params(portfolio, library, **settings), cfg)


def test_the_label_names_the_factors():
    assert backtest.FactorPortfolio(names=('value', 'momentum')).label == 'value + momentum'


def test_a_portfolio_reads_back_from_the_command_it_writes():
    from investalyze.research import __main__ as cli
    portfolio = backtest.FactorPortfolio(names=('value', 'momentum'), n_quantiles=10, rebalance='Q')
    assert backtest.FactorPortfolio.from_arguments(cli._parser().parse_args(['backtest', *portfolio.arguments()])) == portfolio


def test_stored_portfolio_run_ids_are_unchanged():
    """A re-run of a stored configuration must replace its run, so the id stays what earlier versions made."""
    from investalyze.backtest import config as backtest_config
    from investalyze.research import factors
    library = {'value': factors.Factor(name='value', expression='income_net_income / market_cap')}
    params = backtest.run_params(backtest.FactorPortfolio(names=('value',)), library, universe_size=1000,
                                 start='2016-01-31', end='2024-12-31', holdout='2022-01-01')
    assert backtest.run_id(params, backtest_config.read()) == 'e973cac93811'


def _simulation() -> backtest.Simulation:
    """A four-session simulation with one fill and one recovered drawdown, enough to build a stored-run report."""
    dates = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    fills = pd.DataFrame({'Date': [dates[1]], 'Ticker': ['AAA'], 'Side': ['buy'], 'Shares': [10.0], 'Price': [100.0],
                          'Notional': [1000.0], 'Cost': [1.0], 'Kind': ['market']})
    exposure = pd.Series([0.0, 0.001, 0.001, 0.001], index=dates)
    return backtest.Simulation(equity=pd.Series([1_000_000.0, 1_050_000.0, 980_000.0, 1_060_000.0], index=dates),
                               gross=exposure, net=exposure,
                               fills=fills, benchmark_equity=None, stats={'cagr': 0.1, 'sharpe': 1.0},
                               tickers=['AAA'], names_median=1.0)


def test_a_simulated_portfolio_becomes_rows_for_every_run_table():
    from investalyze.backtest import config as backtest_config
    from investalyze.backtest import writer
    from investalyze.research import factors
    library = {'value': factors.Factor(name='value', expression='revenue')}
    portfolio = backtest.FactorPortfolio(names=('value',))
    params = backtest.run_params(portfolio, library, universe_size=1000, start='2024-01-02', end='2024-01-05', holdout=None)
    finished = backtest.build_report(_simulation(), portfolio, params, backtest_config.read())
    assert list(finished.run_row().columns) == writer.RUN_COLS
    assert list(finished.equity_rows().columns) == writer.EQUITY_COLS
    assert list(finished.trade_rows().columns) == writer.TRADE_COLS
    assert finished.signal == 'value'
    assert '| 2024-01-03 | 2024-01-04 | 2024-01-05 | -6.7% | 1 |' in finished.render()


def test_a_simulated_timing_rule_becomes_rows_for_every_run_table():
    from investalyze.backtest import config as backtest_config
    from investalyze.backtest import writer
    from investalyze.research import factors, timing
    library = {'new_high': factors.Factor(name='new_high', expression='close')}
    rule = timing.TimingRule('new_high', trailing_stop=0.1)
    params = timing.run_params(rule, library, universe=None, universe_size=500, start='2024-01-02', end='2024-01-05',
                               holdout=None)
    finished = timing.build_report(_simulation(), rule, params, backtest_config.read())
    assert list(finished.run_row().columns) == writer.RUN_COLS
    assert list(finished.trade_rows().columns) == writer.TRADE_COLS
    assert finished.signal == 'new_high, exit on entry false, trailing stop 10%'
