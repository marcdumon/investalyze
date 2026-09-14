"""Tests for the book, order batches, market axes, the run tables and the run identifier."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.backtest import orders, writer
from investalyze.backtest.book import Book
from investalyze.backtest.market import MarketData
from investalyze.backtest.orders import OrderBatch


@pytest.fixture
def market() -> MarketData:
    """A three-session, two-name market with flat prices, enough to build a book and a batch."""
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
    assert {'backtest_runs', 'backtest_equity', 'backtest_trades'} <= tables


def test_the_run_id_is_stable_for_one_config_and_differs_for_another():
    assert writer.run_id({'signal': 'a', 'start': '2010-01-01'}) == writer.run_id({'start': '2010-01-01', 'signal': 'a'})
    assert writer.run_id({'signal': 'a'}) != writer.run_id({'signal': 'b'})


def test_the_run_id_changes_when_the_engine_schema_version_changes(monkeypatch):
    before = writer.run_id({'signal': 'a'})
    monkeypatch.setattr(writer, 'RUN_SCHEMA_VERSION', writer.RUN_SCHEMA_VERSION + 1)
    assert writer.run_id({'signal': 'a'}) != before


def test_replace_run_rejects_a_summary_with_the_wrong_columns(con: duckdb.DuckDBPyConnection):
    summary = pd.DataFrame([{'RunId': 'abc', 'Signal': 'new_252d_high'}])
    with pytest.raises(ValueError, match='summary columns must be'):
        writer.replace_run(con, 'abc', summary, pd.DataFrame(columns=writer.EQUITY_COLS),
                           pd.DataFrame(columns=writer.TRADE_COLS))


def test_replace_run_writes_a_run_and_rewriting_it_leaves_the_row_counts_unchanged(con: duckdb.DuckDBPyConnection):
    summary = pd.DataFrame([dict.fromkeys(writer.RUN_COLS, None)])
    summary.loc[0, ['RunId', 'Signal', 'Kind', 'ConfigJson']] = ['abc', 'new_252d_high', 'timing', '{}']
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
    summary.loc[0, ['RunId', 'Signal', 'Kind', 'ConfigJson']] = ['abc', 'new_252d_high', 'timing', '{}']
    summary.loc[0, ['StartDate', 'EndDate']] = ['2026-01-05', '2026-01-07']
    summary.loc[0, 'NTickers'] = 2
    long_equity = pd.DataFrame({'RunId': ['abc'] * 3, 'Date': ['2026-01-05', '2026-01-06', '2026-01-07'],
                                'Equity': [100.0, 101.0, 102.0], 'Gross': [0.0] * 3, 'Net': [0.0] * 3,
                                'Cash': [100.0, 101.0, 102.0], 'BenchmarkEquity': [100.0, 100.5, 101.0]})
    trades = pd.DataFrame(columns=writer.TRADE_COLS)
    writer.replace_run(con, 'abc', summary, long_equity, trades)

    writer.replace_run(con, 'abc', summary, long_equity.head(1), trades)
    assert con.execute("SELECT count(*) FROM backtest_equity WHERE RunId = 'abc'").fetchone()[0] == 1
