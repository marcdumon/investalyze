"""Read-only status queries for the control panel's monitor cards.

Every function takes an already-open connection; the caller opens one read-only connection
per refresh and closes it immediately after, so this module never holds the DB open (that
would block a running ingest/cleaning subprocess, which needs the single writer lock).
"""

from datetime import date

import duckdb
import pandas as pd

_FRESHNESS_SOURCES = [
    ('prices', 'Date'),
    ('market_data', 'Date'),
    ('dividends', 'Date'),
    ('income', 'Report Date'),
    ('anomalies', 'DetectedAt'),
]


def freshness(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Most recent date per key table, plus how many days ago that was."""
    rows = []
    for table, column in _FRESHNESS_SOURCES:
        last = con.execute(f'SELECT max("{column}") FROM "{table}"').fetchone()[0]
        last_date = last.date() if hasattr(last, 'date') else last
        days_ago = (date.today() - last_date).days if last_date is not None else None
        rows.append({'table': table, 'last': last_date, 'days_ago': days_ago})
    return pd.DataFrame(rows)


def row_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Row count of every table in the database, largest first."""
    tables = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").df()['table_name']
    counts = [{'table': t, 'rows': con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]} for t in tables]
    return pd.DataFrame(counts).sort_values('rows', ascending=False).reset_index(drop=True)


def anomaly_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Anomaly counts grouped by check name and severity."""
    return con.execute("""
        SELECT CheckName, Severity, count(*) AS n
        FROM anomalies GROUP BY CheckName, Severity ORDER BY Severity, n DESC
    """).df()
