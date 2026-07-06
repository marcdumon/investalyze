"""Tests for tolerance-based accounting-identity checks on balance, income and cashflow."""

import duckdb
import pytest

from investalyze.quality import fundamentals_identities, writer

_ID_COLS = 'Ticker VARCHAR, Market VARCHAR, Period VARCHAR, IsRestated BOOLEAN, "Fiscal Year" BIGINT, "Fiscal Period" VARCHAR'


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with narrow fundamentals tables holding only the referenced columns."""
    con = duckdb.connect()
    con.execute(f"""CREATE TABLE balance ({_ID_COLS},
        "Total Liabilities" BIGINT, "Total Equity" BIGINT, "Total Assets" BIGINT,
        "Total Current Assets" BIGINT, "Total Noncurrent Assets" BIGINT,
        "Total Current Liabilities" BIGINT, "Total Noncurrent Liabilities" BIGINT)""")
    con.execute(f"""CREATE TABLE income ({_ID_COLS},
        Revenue BIGINT, "Cost of Revenue" BIGINT, "Gross Profit" BIGINT,
        "Other Operating Income" BIGINT, "Operating Expenses" BIGINT, "Operating Income (Loss)" BIGINT,
        "Non-Operating Income (Loss)" BIGINT, "Pretax Income (Loss), Adj." BIGINT)""")
    con.execute(f"""CREATE TABLE cashflow ({_ID_COLS},
        "Net Cash from Operating Activities" BIGINT, "Net Cash from Investing Activities" BIGINT,
        "Net Cash from Financing Activities" BIGINT, "Effect of Foreign Exchange Rates" BIGINT,
        "Change in Cash from Disc. Operations and Other" BIGINT, "Net Change in Cash" BIGINT)""")
    return con


def add_balance(con, ticker: str = 'T', *, liab=None, equity=None, assets=None,
                cur_a=None, noncur_a=None, cur_l=None, noncur_l=None) -> None:
    """Insert one annual restated balance row (identity columns fixed, line items per test)."""
    con.execute('INSERT INTO balance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [ticker, 'us', 'A', True, 2025, 'FY', liab, equity, assets, cur_a, noncur_a, cur_l, noncur_l])


def add_income(con, ticker: str = 'T', *, revenue=None, cogs=None, gp=None,
               other_op=None, opex=None, oi=None, nonop=None, pretax=None) -> None:
    """Insert one annual restated income row."""
    con.execute('INSERT INTO income VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [ticker, 'us', 'A', True, 2025, 'FY', revenue, cogs, gp, other_op, opex, oi, nonop, pretax])


def add_cashflow(con, ticker: str = 'T', *, op=None, inv=None, fin=None, fx=None, disc=None, net=None) -> None:
    """Insert one annual restated cashflow row."""
    con.execute('INSERT INTO cashflow VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [ticker, 'us', 'A', True, 2025, 'FY', op, inv, fin, fx, disc, net])


# --- balance_identity -----------------------------------------------------------


def test_balance_identity_flags_violation_beyond_tolerance(con):
    add_balance(con, 'BAD', liab=6_000_000_000, equity=3_000_000_000, assets=10_000_000_000)
    df = fundamentals_identities.balance_identity(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert df['Ticker'].tolist() == ['BAD']
    assert df['Key'].iloc[0] == 'us|A|2025|FY|true'
    assert 'diff=10' in df['Details'].iloc[0]


def test_balance_identity_passes_within_relative_tolerance(con):
    add_balance(con, 'OK', liab=6_000_000_000, equity=3_950_000_000, assets=10_000_000_000)
    assert fundamentals_identities.balance_identity(con).empty


def test_balance_identity_passes_under_absolute_floor(con):
    add_balance(con, 'SMALL', liab=500_000, equity=450_000, assets=1_000_000)
    assert fundamentals_identities.balance_identity(con).empty


def test_balance_identity_skips_rows_with_null_required_column(con):
    add_balance(con, 'NULLEQ', liab=6_000_000_000, equity=None, assets=10_000_000_000)
    assert fundamentals_identities.balance_identity(con).empty


def test_balance_identity_tolerance_is_tunable(con):
    add_balance(con, 'BAD', liab=6_000_000_000, equity=3_000_000_000, assets=10_000_000_000)
    assert fundamentals_identities.balance_identity(con, rel_tol=0.2).empty


# --- balance_subtotals ----------------------------------------------------------


def test_balance_subtotals_flags_both_sides(con):
    add_balance(con, 'BADA', cur_a=3_000_000_000, noncur_a=3_000_000_000, assets=10_000_000_000)
    add_balance(con, 'BADL', cur_l=1_000_000_000, noncur_l=1_000_000_000, liab=10_000_000_000)
    df = fundamentals_identities.balance_subtotals(con)
    assert sorted(df['Ticker']) == ['BADA', 'BADL']


# --- income_chain ---------------------------------------------------------------


def test_income_chain_passes_consistent_signed_chain(con):
    add_income(con, 'OK', revenue=100_000_000_000, cogs=-40_000_000_000, gp=60_000_000_000,
               opex=-20_000_000_000, oi=40_000_000_000, nonop=-1_000_000_000, pretax=39_000_000_000)
    assert fundamentals_identities.income_chain(con).empty


def test_income_chain_flags_broken_gross_profit_link(con):
    add_income(con, 'BADGP', revenue=100_000_000_000, cogs=-40_000_000_000, gp=70_000_000_000)
    df = fundamentals_identities.income_chain(con)
    assert df['Ticker'].tolist() == ['BADGP']
    assert 'rev+cogs' in df['Details'].iloc[0]


def test_income_chain_treats_null_optional_as_zero_and_uses_it_when_present(con):
    add_income(con, 'WITHOPT', gp=60_000_000_000, other_op=5_000_000_000, opex=-20_000_000_000, oi=45_000_000_000)
    add_income(con, 'NULLOPT', gp=60_000_000_000, other_op=None, opex=-20_000_000_000, oi=40_000_000_000)
    assert fundamentals_identities.income_chain(con).empty


# --- cashflow_identity ----------------------------------------------------------


def test_cashflow_identity_flags_violation(con):
    add_cashflow(con, 'BAD', op=10_000_000_000, inv=-4_000_000_000, fin=-2_000_000_000, net=9_000_000_000)
    df = fundamentals_identities.cashflow_identity(con)
    assert df['Ticker'].tolist() == ['BAD']


def test_cashflow_identity_sums_optional_terms(con):
    add_cashflow(con, 'OK', op=10_000_000_000, inv=-4_000_000_000, fin=-2_000_000_000,
                 fx=500_000_000, disc=-500_000_000, net=4_000_000_000)
    assert fundamentals_identities.cashflow_identity(con).empty
