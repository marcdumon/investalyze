"""Tests for hard-invariant and sign checks on the fundamentals tables."""

import duckdb
import pytest

from investalyze.quality import fundamentals_sanity, writer

_ID_COLS = 'Ticker VARCHAR, Market VARCHAR, Period VARCHAR, IsRestated BOOLEAN, "Fiscal Year" BIGINT, "Fiscal Period" VARCHAR'


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with narrow income/balance/cashflow tables."""
    con = duckdb.connect()
    con.execute(f'CREATE TABLE income ({_ID_COLS}, "Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT, Revenue BIGINT)')
    con.execute(f'CREATE TABLE balance ({_ID_COLS}, "Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT, "Total Assets" BIGINT)')
    con.execute(f'CREATE TABLE cashflow ({_ID_COLS}, "Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT)')
    return con


def add_row(con, table: str, ticker: str, values: str) -> None:
    """Insert one annual restated row into `table` with `values` filling the line-item columns."""
    con.execute(f"INSERT INTO {table} VALUES ('{ticker}', 'us', 'A', true, 2025, 'FY', {values})")


# --- hard_invariants ------------------------------------------------------------


def test_hard_invariants_flags_nonpositive_shares_in_every_table(con):
    add_row(con, 'income', 'BADI', '0, 1000, 500')
    add_row(con, 'balance', 'BADB', '-5, 1000, 500')
    add_row(con, 'cashflow', 'BADC', '1000, 0')
    add_row(con, 'income', 'OK', '1000, 1000, 500')
    df = fundamentals_sanity.hard_invariants(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert sorted(df['Ticker']) == ['BADB', 'BADC', 'BADI']


def test_hard_invariants_handles_null_share_column_in_details(con):
    add_row(con, 'income', 'NULLDIL', '0, NULL, 500')
    df = fundamentals_sanity.hard_invariants(con)
    assert df['Ticker'].tolist() == ['NULLDIL']
    assert 'null' in df['Details'].iloc[0]


def test_hard_invariants_flags_negative_total_assets(con):
    add_row(con, 'balance', 'NEGASSETS', '1000, 1000, -500')
    df = fundamentals_sanity.hard_invariants(con)
    assert df['Ticker'].tolist() == ['NEGASSETS']
    assert 'Total Assets=-500' in df['Details'].iloc[0]


# --- negative_revenue -----------------------------------------------------------


def test_negative_revenue_flags_only_negative(con):
    add_row(con, 'income', 'NEGREV', '1000, 1000, -500')
    add_row(con, 'income', 'OK', '1000, 1000, 500')
    df = fundamentals_sanity.negative_revenue(con)
    assert df['Ticker'].tolist() == ['NEGREV']
    assert df['Key'].iloc[0] == 'us|A|2025|FY|true'
