"""Tests for the quality writer: anomalies table creation and the replace-findings contract."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from investalyze.quality import writer


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the anomalies table created."""
    con = duckdb.connect()
    writer.ensure_table(con)
    return con


def make_findings(n: int = 2, ticker: str = 'AAPL') -> pd.DataFrame:
    """Findings frame with `n` rows following the writer's column contract."""
    rows = []
    for i in range(n):
        rows.append({'SrcTable': 'prices', 'Ticker': ticker, 'Date': date(2026, 1, 1 + i), 'Key': None, 'Details': f'C=-{i + 1}'})
    return pd.DataFrame(rows, columns=writer.FINDING_COLS)


# --- ensure_table ---------------------------------------------------------------


def test_ensure_table_creates_anomalies_schema(con):
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'anomalies' ORDER BY ordinal_position"
    ).fetchall()
    assert [c[0] for c in cols] == ['CheckName', 'Severity', 'SrcTable', 'Ticker', 'Date', 'Key', 'Details', 'DetectedAt']


def test_ensure_table_is_idempotent(con):
    writer.ensure_table(con)


# --- replace_findings -----------------------------------------------------------


def test_replace_findings_inserts_stamped_rows(con):
    n = writer.replace_findings(con, 'nonpositive_price', 'error', make_findings(2))
    assert n == 2
    rows = con.execute('SELECT CheckName, Severity, SrcTable, Ticker, Date, Details FROM anomalies ORDER BY Date').fetchall()
    assert rows == [
        ('nonpositive_price', 'error', 'prices', 'AAPL', date(2026, 1, 1), 'C=-1'),
        ('nonpositive_price', 'error', 'prices', 'AAPL', date(2026, 1, 2), 'C=-2'),
    ]
    stamped = con.execute('SELECT count(*) FROM anomalies WHERE DetectedAt IS NOT NULL').fetchone()[0]
    assert stamped == 2


def test_replace_findings_rerun_replaces_own_rows(con):
    writer.replace_findings(con, 'stale_run', 'warn', make_findings(3))
    n = writer.replace_findings(con, 'stale_run', 'warn', make_findings(1))
    assert n == 1
    assert con.execute("SELECT count(*) FROM anomalies WHERE CheckName = 'stale_run'").fetchone()[0] == 1


def test_replace_findings_keeps_other_checks_rows(con):
    writer.replace_findings(con, 'stale_run', 'warn', make_findings(2))
    writer.replace_findings(con, 'date_gap', 'warn', make_findings(1, ticker='MSFT'))
    counts = dict(con.execute('SELECT CheckName, count(*) FROM anomalies GROUP BY CheckName').fetchall())
    assert counts == {'stale_run': 2, 'date_gap': 1}


def test_replace_findings_empty_frame_clears_previous(con):
    writer.replace_findings(con, 'stale_run', 'warn', make_findings(2))
    n = writer.replace_findings(con, 'stale_run', 'warn', make_findings(0))
    assert n == 0
    assert con.execute('SELECT count(*) FROM anomalies').fetchone()[0] == 0


def test_replace_findings_rejects_wrong_columns(con):
    bad = pd.DataFrame({'Ticker': ['AAPL'], 'Details': ['x']})
    with pytest.raises(ValueError, match='findings columns'):
        writer.replace_findings(con, 'stale_run', 'warn', bad)
