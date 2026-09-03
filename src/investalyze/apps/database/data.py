"""Read-only access for the Database page: what tables exist, what is in them, and ad-hoc queries.

Every call opens a short-lived read-only connection, matching the other pages. Read-only is what
makes the query box safe: DuckDB itself rejects a write, which is a stronger guarantee than reading
SQL and guessing whether it looks dangerous.

Column statistics are the one expensive read here. Counting nulls costs 0.07s on the million-row
factor panel and 5.3s on the 27-million-row price table, so above `STATS_ROW_LIMIT` they are skipped
and the page says so rather than stalling.
"""

import threading
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'

STATS_ROW_LIMIT = 5_000_000   # above this, counting nulls is slow enough to be worth skipping
SAMPLE_ROWS = 50
QUERY_ROW_LIMIT = 1_000
QUERY_TIMEOUT_SECONDS = 20.0

DATE_TYPES = ('DATE', 'TIMESTAMP')


def _connect() -> duckdb.DuckDBPyConnection:
    """A read-only connection to the project database."""
    return storage.connect(DATA_ROOT, read_only=True)


def list_tables() -> pd.DataFrame:
    """Every table with its row and column counts. Both counts come from metadata, so this is cheap."""
    con = _connect()
    try:
        return con.execute("""
            SELECT t.table_name AS name,
                   (SELECT count(*) FROM information_schema.columns c
                    WHERE c.table_name = t.table_name) AS columns
            FROM information_schema.tables t ORDER BY t.table_name
        """).df().assign(rows=lambda frame: [storage.count_rows(con, name) for name in frame['name']])
    finally:
        con.close()


def describe_table(name: str) -> dict:
    """One table's columns and types, its row count, and the span of its date column where it has one.

    Null counts are part of it only when the table is small enough for the pass to be quick; `stats`
    says which happened, so the page can explain an empty column rather than showing a blank.
    """
    con = _connect()
    try:
        if not storage.table_exists(con, name):
            return {}
        columns = con.execute("""
            SELECT column_name AS column, data_type AS type FROM information_schema.columns
            WHERE table_name = ? ORDER BY ordinal_position
        """, [name]).df()
        rows = storage.count_rows(con, name)
        stats = rows <= STATS_ROW_LIMIT
        if stats and len(columns):
            # count(col) skips nulls, so total minus it is the null count, in one pass
            counted = ', '.join(f'count("{column}")' for column in columns['column'])
            filled = con.execute(f'SELECT count(*), {counted} FROM "{name}"').fetchone()
            columns['nulls'] = [filled[0] - value for value in filled[1:]]
        span = _date_span(con, name, columns)
        return {'name': name, 'rows': rows, 'columns': columns, 'stats': stats, **span}
    finally:
        con.close()


def _date_span(con: duckdb.DuckDBPyConnection, name: str, columns: pd.DataFrame) -> dict:
    """First and last date a table covers, and how many tickers, where it carries those columns."""
    span: dict = {}
    dated = [row.column for row in columns.itertuples()
             if row.type in DATE_TYPES and row.column in ('Date', 'BuiltAt', 'DetectedAt')]
    if dated:
        first, last = con.execute(f'SELECT min("{dated[0]}"), max("{dated[0]}") FROM "{name}"').fetchone()
        span |= {'first': str(first)[:10] if first is not None else None,
                 'last': str(last)[:10] if last is not None else None}
    if 'Ticker' in set(columns['column']):
        span['tickers'] = con.execute(f'SELECT count(DISTINCT Ticker) FROM "{name}"').fetchone()[0]
    return span


def sample_rows(name: str, limit: int = SAMPLE_ROWS) -> pd.DataFrame:
    """The first rows of a table, so a reader can see what its values actually look like."""
    con = _connect()
    try:
        if not storage.table_exists(con, name):
            return pd.DataFrame()
        return con.execute(f'SELECT * FROM "{name}" LIMIT {int(limit)}').df()
    finally:
        con.close()


def run_query(sql: str, limit: int = QUERY_ROW_LIMIT,
              timeout: float = QUERY_TIMEOUT_SECONDS) -> tuple[pd.DataFrame, str | None]:
    """Run one read-only statement, capped and interrupted if it runs long.

    Returns the rows and an error message, exactly one of which is meaningful. The connection is
    read-only, so a write comes back as the database's own refusal rather than something this has to
    recognise. DuckDB has no statement timeout, so a watchdog interrupts a query that overruns.
    """
    con = _connect()
    expired = threading.Event()

    def stop() -> None:
        """Interrupt the query once it has run longer than the caller allowed."""
        expired.set()
        con.interrupt()

    watchdog = threading.Timer(timeout, stop)
    watchdog.start()
    try:
        frame = con.execute(f'SELECT * FROM ({sql.rstrip().rstrip(";")}) LIMIT {int(limit)}').df()
        return frame, None
    except duckdb.Error as failure:
        if expired.is_set():
            return pd.DataFrame(), f'stopped after {timeout:.0f} seconds'
        return pd.DataFrame(), str(failure)
    finally:
        watchdog.cancel()
        con.close()
