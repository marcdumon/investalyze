"""Tests for the run report: the rows it stores and the order it renders things in.

The rule under test is that a reader meets the caveats before the numbers. A report whose headline
Sharpe appears above a caveat about the run is a report that misleads, so the ordering is asserted
rather than left to the template.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investalyze.backtest import performance, report, writer
from investalyze.backtest.config import read as read_config

DATES = pd.bdate_range('2020-01-01', periods=10)
BENCHMARK_NOTE = 'benchmark ^NDX covers only to 2019-12-31'


@pytest.fixture
def finished() -> report.BacktestReport:
    """A small finished run, with one fill and a benchmark that covers the whole range."""
    equity = pd.Series(np.linspace(100_000, 120_000, len(DATES)), index=DATES)
    gross = pd.Series(np.linspace(90_000, 110_000, len(DATES)), index=DATES)
    fills = pd.DataFrame([{'Date': DATES[1], 'Ticker': 'AAA', 'Side': 'buy', 'Shares': 10.0,
                           'Price': 100.0, 'Notional': 1_000.0, 'Cost': 2.0, 'Kind': 'market'}])
    return report.BacktestReport(
        run_id='abc123def456', signal='factor_composite', kind='cross_sectional',
        config=read_config(), params={'factors': ['roe']}, start=DATES[0], end=DATES[-1],
        tickers=['AAA', 'BBB', 'CCC'], benchmark='^NDX',
        caveats=[BENCHMARK_NOTE],
        stats=performance.summary(equity, fills), equity=equity, gross=gross, net=gross,
        cash=equity - gross, fills=fills,
        exposure=performance.exposures(gross, gross, equity),
    )


def test_the_caveats_are_rendered_above_the_performance_numbers(finished):
    rendered = finished.render()
    assert rendered.index('Read this first') < rendered.index('## Performance')
    assert rendered.index(BENCHMARK_NOTE) < rendered.index('| sharpe |')


def test_a_report_with_no_caveats_skips_the_section_entirely(finished):
    finished.caveats = []
    assert 'Read this first' not in finished.render()


def test_every_caveat_reaches_the_page(finished):
    rendered = finished.render()
    for caveat in finished.caveats:
        assert caveat in rendered


def test_the_cost_assumptions_are_rendered_beside_the_numbers(finished):
    rendered = finished.render()
    assert 'slippage 5 bps' in rendered
    assert 'commission 2 bps' in rendered
    assert rendered.index('Cost assumptions') < rendered.index('## Performance')


def test_the_run_row_matches_the_stored_column_contract(finished):
    assert list(finished.run_row().columns) == writer.RUN_COLS


def test_the_equity_rows_match_the_stored_column_contract(finished):
    rows = finished.equity_rows()
    assert list(rows.columns) == writer.EQUITY_COLS
    assert len(rows) == len(DATES)


def test_the_trade_rows_match_the_stored_column_contract(finished):
    assert list(finished.trade_rows().columns) == writer.TRADE_COLS


def test_a_run_with_no_fills_still_produces_a_valid_blotter(finished):
    finished.fills = pd.DataFrame()
    assert list(finished.trade_rows().columns) == writer.TRADE_COLS
    assert finished.trade_rows().empty


def test_the_config_json_records_the_costs_the_run_used(finished):
    payload = finished.config_json()
    assert '"slippage_bps": 5.0' in payload
    assert '"signal": "factor_composite"' in payload


def test_the_report_writes_to_a_directory_named_for_the_run(finished, tmp_path: Path):
    path = report.write_report(finished, tmp_path)
    assert path == tmp_path / 'backtests' / 'abc123def456' / 'report.md'
    assert BENCHMARK_NOTE in path.read_text()


def test_a_statistic_that_could_not_be_computed_renders_as_not_available(finished):
    finished.stats['sharpe'] = float('nan')
    assert '| sharpe | n/a |' in finished.render()


def test_the_stored_run_round_trips_through_the_writer(finished):
    import duckdb

    con = duckdb.connect()
    writer.ensure_tables(con)
    written = writer.replace_run(con, finished.run_id, finished.run_row(),
                                 finished.equity_rows(), finished.trade_rows())
    assert written == len(DATES)
    stored = con.execute('SELECT Signal FROM backtest_runs WHERE RunId = ?',
                         [finished.run_id]).fetchone()
    assert stored[0] == finished.signal


def test_deleting_a_run_reports_what_it_removed_from_each_table(finished):
    import duckdb

    con = duckdb.connect()
    writer.ensure_tables(con)
    writer.replace_run(con, finished.run_id, finished.run_row(), finished.equity_rows(),
                       finished.trade_rows())
    removed = writer.delete_run(con, finished.run_id)
    assert removed == {'backtest_runs': 1, 'backtest_equity': len(DATES), 'backtest_trades': 1}
    assert con.execute('SELECT count(*) FROM backtest_runs').fetchone()[0] == 0


def test_deleting_a_run_that_was_never_stored_removes_nothing(finished):
    import duckdb

    con = duckdb.connect()
    writer.ensure_tables(con)
    assert not any(writer.delete_run(con, 'neverstored').values())


def test_deleting_a_report_removes_the_directory_the_run_wrote(finished, tmp_path: Path):
    report.write_report(finished, tmp_path)
    assert report.delete_report(finished.run_id, tmp_path) is True
    assert not report.report_dir(finished.run_id, tmp_path).exists()


def test_deleting_a_report_that_was_never_written_says_so(tmp_path: Path):
    assert report.delete_report('abc123def456', tmp_path) is False


# --- evaluation rows are kept per universe ---

def _ic_rows(universe: str, factor: str = 'roe') -> pd.DataFrame:
    """One IC row for a universe, in the stored column order."""
    return pd.DataFrame([{'Version': 'as_reported', 'Universe': universe, 'Factor': factor,
                          'Date': DATES[0], 'Horizon': 21, 'Ic': 0.05, 'NNames': 100}])


def _quantile_rows(universe: str, factor: str = 'roe') -> pd.DataFrame:
    """One quantile row for a universe, in the stored column order."""
    return pd.DataFrame([{'Version': 'as_reported', 'Universe': universe, 'Factor': factor,
                          'Date': DATES[0], 'Horizon': 21, 'Quantile': 4, 'Return': 0.02,
                          'MedianReturn': 0.01, 'NNames': 20}])


def test_evaluating_the_same_factor_on_two_universes_keeps_both():
    import duckdb

    con = duckdb.connect()
    writer.ensure_tables(con)
    for universe in ('all', 'sp500'):
        writer.replace_evaluation(con, 'as_reported', universe, 'roe',
                                  _ic_rows(universe), _quantile_rows(universe))
    stored = con.execute('SELECT Universe FROM factor_ic ORDER BY Universe').fetchall()
    assert stored == [('all',), ('sp500',)]


def test_re_evaluating_one_universe_replaces_only_its_own_rows():
    import duckdb

    con = duckdb.connect()
    writer.ensure_tables(con)
    for universe in ('all', 'sp500'):
        writer.replace_evaluation(con, 'as_reported', universe, 'roe',
                                  _ic_rows(universe), _quantile_rows(universe))
    again = _ic_rows('sp500')
    again['Ic'] = 0.99
    writer.replace_evaluation(con, 'as_reported', 'sp500', 'roe', again, _quantile_rows('sp500'))
    stored = dict(con.execute('SELECT Universe, Ic FROM factor_ic').fetchall())
    assert stored == {'all': 0.05, 'sp500': 0.99}


def test_a_table_written_before_the_universe_column_gains_it_without_losing_its_rows():
    """The column is added to the end of an existing table, so the writer must insert by name.

    Older rows are labelled unrecorded, not all: nothing recorded what they were measured over.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("""CREATE TABLE factor_ic (Version VARCHAR, Factor VARCHAR, Date DATE,
                   Horizon BIGINT, Ic DOUBLE, NNames BIGINT)""")
    con.execute("INSERT INTO factor_ic VALUES ('as_reported', 'roe', DATE '2020-01-01', 21, 0.03, 50)")

    writer.ensure_tables(con)

    assert con.execute('SELECT Universe FROM factor_ic').fetchall() == [(writer.UNRECORDED_UNIVERSE,)]
    writer.replace_evaluation(con, 'as_reported', 'sp500', 'roe', _ic_rows('sp500'), _quantile_rows('sp500'))
    landed = con.execute("SELECT Factor, Ic, NNames FROM factor_ic WHERE Universe = 'sp500'").fetchone()
    assert landed == ('roe', 0.05, 100)
