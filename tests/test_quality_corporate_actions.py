"""Tests for dividend and split checks: sign, size vs close, ratio validity."""

import duckdb
import pytest

from investalyze.quality import corporate_actions, writer


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with tiny dividends, splits and prices tables."""
    con = duckdb.connect()
    con.execute('CREATE TABLE dividends (Ticker VARCHAR, Date DATE, Dividend DOUBLE)')
    con.executemany('INSERT INTO dividends VALUES (?, ?, ?)', [
        ('OK', '2026-01-02', 0.5),     # 5% of close
        ('ZERO', '2026-01-02', 0.0),
        ('BIG', '2026-01-02', 4.0),    # 40% of close
        ('NOPX', '2026-01-02', 5.0),   # no matching price row
    ])
    con.execute('CREATE TABLE splits (Ticker VARCHAR, Date DATE, Ratio DOUBLE)')
    con.executemany('INSERT INTO splits VALUES (?, ?, ?)', [
        ('OK', '2026-01-02', 2.0),
        ('ZERO', '2026-01-02', 0.0),
        ('ONE', '2026-01-02', 1.0),
        ('NEG', '2026-01-02', -2.0),
    ])
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
        ('OK', '2026-01-02', 10.0, 10.0, 10.0, 10.0, 100, 10.0),
        ('ZERO', '2026-01-02', 10.0, 10.0, 10.0, 10.0, 100, 10.0),
        ('BIG', '2026-01-02', 10.0, 10.0, 10.0, 10.0, 100, 10.0),
    ])
    return con


def test_nonpositive_dividend_flags_only_nonpositive(con):
    df = corporate_actions.nonpositive_dividend(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert df['Ticker'].tolist() == ['ZERO']


def test_oversized_dividend_flags_dividend_beyond_close_fraction(con):
    df = corporate_actions.oversized_dividend(con)
    assert df['Ticker'].tolist() == ['BIG']


def test_oversized_dividend_skips_dividends_without_price_row(con):
    tickers = corporate_actions.oversized_dividend(con)['Ticker'].tolist()
    assert 'NOPX' not in tickers


def test_oversized_dividend_fraction_is_tunable(con):
    assert corporate_actions.oversized_dividend(con, max_dividend_frac=0.5).empty


def test_invalid_split_ratio_flags_zero_one_and_negative(con):
    df = corporate_actions.invalid_split_ratio(con)
    assert sorted(df['Ticker']) == ['NEG', 'ONE', 'ZERO']
