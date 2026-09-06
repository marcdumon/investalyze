"""Tests for the signal registry, the writer DDL, and the run identifier.

The sweep tests are the point: every signal must be described, must carry a valid kind, and must
build a step function that the engine can call. Adding a signal without a description or with a
typo'd kind fails here rather than at the end of a long run.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import factors_pit
from investalyze.backtest import orders, registry, writer
from investalyze.backtest.book import Book
from investalyze.backtest.market import MarketData
from investalyze.backtest.orders import OrderBatch

EXPECTED_SIGNALS = ['breakout_52w', 'factor_composite', 'hold_cash', 'ibs_band_reversion', 'single_factor',
                    'sma_cross']


@pytest.fixture
def market() -> MarketData:
    """A three-session, two-name market with flat prices, enough for a factory to build a step."""
    dates = pd.DatetimeIndex(['2026-01-05', '2026-01-06', '2026-01-07'])
    tickers = np.array(['AAA', 'BBB'])
    prices = np.full((3, 2), 100.0)
    return MarketData(
        dates=dates, tickers=tickers,
        open_=prices, high=prices, low=prices, close=prices,
        volume=np.full((3, 2), 1e6, dtype=np.float32),
        dividend=np.zeros((3, 2)),
        adv=np.full((3, 2), 1e8, dtype=np.float32),
        tradable=np.ones((3, 2), dtype=bool),
        adjusted_close=prices,
    )


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the backtest output tables created."""
    connection = duckdb.connect()
    writer.ensure_tables(connection)
    return connection


@pytest.fixture
def panelled(market: MarketData) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB holding a factor panel over the fixture market, so a factory can read one."""
    connection = duckdb.connect()
    columns = factors_pit.PANEL_KEY + factors_pit.IDENTITY_COLS + factors_pit.FACTORS
    rows = []
    for date in market.dates:
        for position, ticker in enumerate(market.tickers):
            row = {column: np.nan for column in columns}
            row |= {'Ticker': ticker, 'Date': date.date(), 'Version': factors_pit.AS_REPORTED,
                    'name': ticker, 'sector': 'tech', 'industry': 'software',
                    'last_close': 100.0, 'mcap': 1e9, 'mcap_bn': 1.0}
            for factor in factors_pit.FACTORS:
                row[factor] = float(position + 1)   # a clean ordering the ranker can act on
            rows.append(row)
    factors_pit.store_factor_panel(connection, pd.DataFrame(rows, columns=columns))
    return connection


# --- the registry ---

def test_registry_lists_exactly_the_planned_signals():
    assert sorted(registry.SIGNALS) == EXPECTED_SIGNALS


def test_every_signal_has_a_description():
    assert set(registry.SIGNAL_DESCRIPTIONS) == set(registry.SIGNALS)


def test_every_signal_carries_a_known_kind():
    for name, (kind, _) in registry.SIGNALS.items():
        assert kind in registry.KINDS, name


def test_every_signal_factory_builds_a_callable_step(market: MarketData, panelled):
    book = Book.start(10_000.0, len(market.tickers), len(market.dates))
    for name, (_, factory) in registry.SIGNALS.items():
        step = factory(panelled, market, {'rebalance': 'D'})
        for session in range(len(market.dates)):
            decision = step(session, book)
            assert isinstance(decision, registry.Decision), name
            assert decision.targets is None or len(decision.targets) == len(market.tickers), name


def test_a_cross_sectional_signal_targets_the_best_ranked_name(market: MarketData, panelled):
    """Factor values rise with ticker order, so the top bucket of two names is the last one."""
    book = Book.start(10_000.0, len(market.tickers), len(market.dates))
    step = registry.SIGNALS['factor_composite'][1](panelled, market, {'rebalance': 'D', 'n_quantiles': 2})
    targets = step(0, book).targets
    assert targets[-1] > 0
    assert targets[0] == 0


def test_a_cross_sectional_signal_needs_a_factor_panel(market: MarketData):
    empty = duckdb.connect()
    with pytest.raises(ValueError, match='factor_panel does not exist'):
        registry.SIGNALS['factor_composite'][1](empty, market, {})


def test_hold_cash_never_asks_for_a_trade(market: MarketData):
    book = Book.start(10_000.0, 2, 3)
    step = registry.hold_cash(None, market, {})
    for session in range(len(market.dates)):
        assert step(session, book) == registry.HOLD


# --- the book ---

def test_a_started_book_holds_only_cash():
    book = Book.start(50_000.0, 4, 10)
    assert book.cash == 50_000.0
    assert not book.shares.any()
    assert len(book.equity) == 10
    assert book.blotter().empty


def test_position_value_treats_a_missing_price_as_no_value():
    book = Book.start(0.0, 2, 1)
    book.shares[:] = [10.0, 5.0]
    values = book.position_value(np.array([20.0, np.nan]))
    assert list(values) == [200.0, 0.0]


# --- orders ---

def test_an_empty_order_batch_has_the_declared_dtypes():
    batch = OrderBatch.empty()
    assert len(batch) == 0
    assert (batch.asset.dtype, batch.qty.dtype) == (np.int32, np.float64)
    assert (batch.kind.dtype, batch.age.dtype) == (np.int8, np.int16)


def test_every_order_kind_has_a_name():
    assert set(orders.KIND_NAMES) == {orders.MARKET, orders.LIMIT, orders.STOP}
    assert sorted(orders.KIND_NAMES) == [0, 1, 2]


# --- the market axes ---

def test_index_of_finds_a_ticker_and_rejects_an_unknown_one(market: MarketData):
    assert market.index_of('BBB') == 1
    with pytest.raises(KeyError, match='not in this run universe'):
        market.index_of('ZZZ')


def test_shape_reports_sessions_by_assets(market: MarketData):
    assert market.shape == (3, 2)


# --- the writer ---

def test_ensure_tables_is_safe_to_call_twice(con: duckdb.DuckDBPyConnection):
    writer.ensure_tables(con)
    tables = {row[0] for row in con.execute('SHOW TABLES').fetchall()}
    assert {'backtest_runs', 'backtest_equity', 'backtest_trades', 'factor_ic', 'factor_quantiles'} <= tables


def test_the_run_id_is_stable_for_one_config_and_differs_for_another():
    assert writer.run_id({'signal': 'a', 'start': '2010-01-01'}) == writer.run_id({'start': '2010-01-01', 'signal': 'a'})
    assert writer.run_id({'signal': 'a'}) != writer.run_id({'signal': 'b'})


def test_the_run_id_changes_when_the_engine_schema_version_changes(monkeypatch):
    before = writer.run_id({'signal': 'a'})
    monkeypatch.setattr(writer, 'RUN_SCHEMA_VERSION', writer.RUN_SCHEMA_VERSION + 1)
    assert writer.run_id({'signal': 'a'}) != before


def test_replace_run_rejects_a_summary_with_the_wrong_columns(con: duckdb.DuckDBPyConnection):
    summary = pd.DataFrame([{'RunId': 'abc', 'Signal': 'hold_cash'}])
    with pytest.raises(ValueError, match='summary columns must be'):
        writer.replace_run(con, 'abc', summary, pd.DataFrame(columns=writer.EQUITY_COLS),
                           pd.DataFrame(columns=writer.TRADE_COLS))


def test_replace_run_writes_a_run_and_rewriting_it_leaves_the_row_counts_unchanged(con: duckdb.DuckDBPyConnection):
    summary = pd.DataFrame([dict.fromkeys(writer.RUN_COLS, None)])
    summary.loc[0, ['RunId', 'Signal', 'Kind', 'ConfigJson']] = ['abc', 'hold_cash', 'timing', '{}']
    summary.loc[0, ['StartDate', 'EndDate']] = ['2026-01-05', '2026-01-07']
    summary.loc[0, 'NTickers'] = 2
    equity = pd.DataFrame({'RunId': ['abc', 'abc'], 'Date': ['2026-01-05', '2026-01-06'],
                           'Equity': [100.0, 101.0], 'Gross': [0.0, 0.0], 'Net': [0.0, 0.0],
                           'Cash': [100.0, 101.0], 'BenchmarkEquity': [100.0, 100.5]})
    trades = pd.DataFrame(columns=writer.TRADE_COLS)

    assert writer.replace_run(con, 'abc', summary, equity, trades) == 2
    assert writer.replace_run(con, 'abc', summary, equity, trades) == 2
    assert con.execute("SELECT count(*) FROM backtest_runs WHERE RunId = 'abc'").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM backtest_equity WHERE RunId = 'abc'").fetchone()[0] == 2


def test_rerunning_over_a_shorter_range_leaves_no_orphan_equity_rows(con: duckdb.DuckDBPyConnection):
    summary = pd.DataFrame([dict.fromkeys(writer.RUN_COLS, None)])
    summary.loc[0, ['RunId', 'Signal', 'Kind', 'ConfigJson']] = ['abc', 'hold_cash', 'timing', '{}']
    summary.loc[0, ['StartDate', 'EndDate']] = ['2026-01-05', '2026-01-07']
    summary.loc[0, 'NTickers'] = 2
    long_equity = pd.DataFrame({'RunId': ['abc'] * 3, 'Date': ['2026-01-05', '2026-01-06', '2026-01-07'],
                                'Equity': [100.0, 101.0, 102.0], 'Gross': [0.0] * 3, 'Net': [0.0] * 3,
                                'Cash': [100.0, 101.0, 102.0], 'BenchmarkEquity': [100.0, 100.5, 101.0]})
    trades = pd.DataFrame(columns=writer.TRADE_COLS)
    writer.replace_run(con, 'abc', summary, long_equity, trades)

    writer.replace_run(con, 'abc', summary, long_equity.head(1), trades)
    assert con.execute("SELECT count(*) FROM backtest_equity WHERE RunId = 'abc'").fetchone()[0] == 1


# --- what a panel build recorded ---

def test_a_recorded_build_reads_back_with_what_it_was_asked_for(con):
    writer.record_panel_build(con, version='as_reported', universe='sp500', start='2015-01-01',
                              end='2024-12-31', frequency='M', n_tickers=500, n_dates=120, n_rows=60_000)
    build = writer.latest_panel_build(con, 'as_reported')
    assert build['universe'] == 'sp500'
    assert build['frequency'] == 'M'
    assert build['n_tickers'] == 500


def test_the_latest_build_is_the_one_that_reads_back(con):
    writer.record_panel_build(con, version='as_reported', universe='first', start='2015-01-01',
                              end='2020-12-31', frequency='M', n_tickers=100, n_dates=72, n_rows=7_200)
    writer.record_panel_build(con, version='as_reported', universe='second', start='2015-01-01',
                              end='2024-12-31', frequency='W', n_tickers=500, n_dates=520, n_rows=260_000)
    assert writer.latest_panel_build(con, 'as_reported')['universe'] == 'second'


def test_a_version_never_built_has_no_record(con):
    assert writer.latest_panel_build(con, 'restated') is None


def test_a_build_records_how_long_it_took_and_the_formulas_it_used(con):
    writer.record_panel_build(con, version='as_reported', universe='solar', start='2015-01-01',
                              end='2024-12-31', frequency='M', n_tickers=7, n_dates=120, n_rows=840,
                              seconds=1.4, definitions='blend=rank_roe:high')
    build = writer.latest_panel_build(con, 'as_reported')
    assert build['seconds'] == 1.4
    assert build['definitions'] == 'blend=rank_roe:high'


def test_a_build_record_written_before_those_columns_reads_back_without_them():
    """The page reads read-only, so it meets the old shape until a write command widens it."""
    connection = duckdb.connect()
    connection.execute("""CREATE TABLE factor_panel_builds (Version VARCHAR, Universe VARCHAR,
                          StartDate DATE, EndDate DATE, Frequency VARCHAR, NTickers BIGINT,
                          NDates BIGINT, NRows BIGINT, BuiltAt TIMESTAMP)""")
    connection.execute("""INSERT INTO factor_panel_builds VALUES ('as_reported', 'solar',
                          DATE '2015-01-01', DATE '2024-12-31', 'M', 7, 120, 840, now())""")

    build = writer.latest_panel_build(connection, 'as_reported')

    assert build['universe'] == 'solar'
    assert build['seconds'] is None
    assert build['definitions'] is None
