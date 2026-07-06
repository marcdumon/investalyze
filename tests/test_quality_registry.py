"""Integration sweep: every registered check runs end-to-end against a full fixture DB."""

import duckdb
import pytest

from investalyze.quality import registry, writer

_ID_COLS = 'Ticker VARCHAR, Market VARCHAR, Period VARCHAR, IsRestated BOOLEAN, "Fiscal Year" BIGINT, "Fiscal Period" VARCHAR'
_SHARES = '"Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT'

EXPECTED_CHECKS = [
    'balance_identity',
    'balance_subtotals',
    'bond_yield_bound',
    'cashflow_identity',
    'date_gap',
    'extreme_return',
    'hard_invariants',
    'income_chain',
    'invalid_split_ratio',
    'negative_revenue',
    'negative_volume',
    'nonpositive_dividend',
    'nonpositive_price',
    'ohlc_inconsistent',
    'oversized_dividend',
    'quarters_vs_fy',
    'stale_run',
]


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with every source table and every column the checks reference."""
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
        ('AAA', '2026-01-02', 10.0, 11.0, 9.0, 10.5, 1000, 10.5),
        ('AAA', '2026-01-03', -1.0, 11.0, -2.0, 10.5, 1000, 10.5),
    ])
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, AssetClass VARCHAR)')
    con.execute("INSERT INTO market_data VALUES ('10YUSY', DATE '2026-01-02', 4.0, 4.1, 3.9, 4.0, 'bonds')")
    con.execute('CREATE TABLE dividends (Ticker VARCHAR, Date DATE, Dividend DOUBLE)')
    con.execute("INSERT INTO dividends VALUES ('AAA', DATE '2026-01-02', 0.5)")
    con.execute('CREATE TABLE splits (Ticker VARCHAR, Date DATE, Ratio DOUBLE)')
    con.execute("INSERT INTO splits VALUES ('AAA', DATE '2026-01-02', 2.0)")
    con.execute(f"""CREATE TABLE income ({_ID_COLS}, {_SHARES},
        Revenue BIGINT, "Cost of Revenue" BIGINT, "Gross Profit" BIGINT,
        "Other Operating Income" BIGINT, "Operating Expenses" BIGINT, "Operating Income (Loss)" BIGINT,
        "Non-Operating Income (Loss)" BIGINT, "Pretax Income (Loss), Adj." BIGINT, "Net Income" BIGINT)""")
    con.execute("""INSERT INTO income VALUES ('AAA', 'us', 'A', true, 2025, 'FY', 1000, 1000,
        100000000000, -40000000000, 60000000000, NULL, -20000000000, 40000000000, -1000000000, 39000000000, 30000000000)""")
    con.execute(f"""CREATE TABLE balance ({_ID_COLS}, {_SHARES},
        "Total Liabilities" BIGINT, "Total Equity" BIGINT, "Total Assets" BIGINT,
        "Total Current Assets" BIGINT, "Total Noncurrent Assets" BIGINT,
        "Total Current Liabilities" BIGINT, "Total Noncurrent Liabilities" BIGINT)""")
    con.execute("""INSERT INTO balance VALUES ('AAA', 'us', 'A', true, 2025, 'FY', 1000, 1000,
        6000000000, 4000000000, 10000000000, 3000000000, 7000000000, 2000000000, 4000000000)""")
    con.execute(f"""CREATE TABLE cashflow ({_ID_COLS}, {_SHARES},
        "Net Cash from Operating Activities" BIGINT, "Net Cash from Investing Activities" BIGINT,
        "Net Cash from Financing Activities" BIGINT, "Effect of Foreign Exchange Rates" BIGINT,
        "Change in Cash from Disc. Operations and Other" BIGINT, "Net Change in Cash" BIGINT)""")
    con.execute("""INSERT INTO cashflow VALUES ('AAA', 'us', 'A', true, 2025, 'FY', 1000, 1000,
        10000000000, -4000000000, -2000000000, NULL, NULL, 4000000000)""")
    return con


def test_registry_lists_exactly_the_planned_checks():
    assert sorted(registry.CHECKS) == EXPECTED_CHECKS


def test_registry_severities_are_valid():
    for name, (severity, _) in registry.CHECKS.items():
        assert severity in ('error', 'warn'), name


def test_every_check_runs_end_to_end_through_the_writer(con):
    writer.ensure_table(con)
    for name, (severity, check) in registry.CHECKS.items():
        findings = check(con)
        assert list(findings.columns) == writer.FINDING_COLS, name
        n = writer.replace_findings(con, name, severity, findings)
        assert n == len(findings), name
    flagged = con.execute("SELECT count(*) FROM anomalies WHERE CheckName = 'nonpositive_price'").fetchone()[0]
    assert flagged == 1
