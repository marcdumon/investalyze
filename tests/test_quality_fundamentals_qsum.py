"""Tests for the quarters-vs-fiscal-year sum check on flow statements."""

import duckdb
import pytest

from investalyze.quality import fundamentals_qsum, writer

_ID_COLS = 'Ticker VARCHAR, Market VARCHAR, Period VARCHAR, IsRestated BOOLEAN, "Fiscal Year" BIGINT, "Fiscal Period" VARCHAR'


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with narrow income and cashflow tables."""
    con = duckdb.connect()
    con.execute(f'CREATE TABLE income ({_ID_COLS}, Revenue BIGINT, "Net Income" BIGINT)')
    con.execute(f'CREATE TABLE cashflow ({_ID_COLS}, "Net Cash from Operating Activities" BIGINT)')
    return con


def seed_year(con, ticker: str, quarters: list[int], fy_total: int | None) -> None:
    """Insert income quarter rows (Revenue only) and, when given, the matching FY row."""
    for i, revenue in enumerate(quarters):
        con.execute('INSERT INTO income VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    [ticker, 'us', 'Q', True, 2025, f'Q{i + 1}', revenue, None])
    if fy_total is not None:
        con.execute('INSERT INTO income VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    [ticker, 'us', 'A', True, 2025, 'FY', fy_total, None])


def test_quarters_matching_fy_pass(con):
    seed_year(con, 'OK', [25_000_000_000] * 4, 100_000_000_000)
    assert fundamentals_qsum.quarters_vs_fy(con).empty


def test_quarters_off_beyond_tolerance_flagged(con):
    seed_year(con, 'BAD', [25_000_000_000] * 4, 105_000_000_000)
    df = fundamentals_qsum.quarters_vs_fy(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert df['Ticker'].tolist() == ['BAD']
    assert df['Key'].iloc[0] == 'us|A|2025|FY|true'
    assert 'Revenue' in df['Details'].iloc[0]


def test_incomplete_quarter_set_skipped(con):
    seed_year(con, 'THREEQ', [25_000_000_000] * 3, 105_000_000_000)
    assert fundamentals_qsum.quarters_vs_fy(con).empty


def test_missing_fy_row_skipped(con):
    seed_year(con, 'NOFY', [25_000_000_000] * 4, None)
    assert fundamentals_qsum.quarters_vs_fy(con).empty


def test_tolerance_is_tunable(con):
    seed_year(con, 'BAD', [25_000_000_000] * 4, 105_000_000_000)
    assert fundamentals_qsum.quarters_vs_fy(con, rel_tol=0.10).empty
