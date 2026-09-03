"""Read access for the database page: what tables exist, what is in them, and arbitrary queries.

Every call opens a short-lived read-only connection, matching the other pages. Read-only is also
what makes the query box safe: DuckDB itself refuses to write on such a connection, which is a
stronger guarantee than reading SQL and hoping to recognise every dangerous form of it.

Row counts and date ranges are read from metadata and zone maps, so they are instant even on the
27 million row price table. Null counts are not, so they are skipped above a size where they would
make selecting a table feel broken.
"""

import threading
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'

SAMPLE_ROWS = 50
QUERY_ROWS = 500          # a page cannot show more, and a runaway query should not try
QUERY_SECONDS = 20.0
NULL_COUNT_LIMIT = 5_000_000   # measured: 1M rows costs 0.07s, 27M costs 5.3s


def _connect() -> duckdb.DuckDBPyConnection:
    """A read-only connection to the project database."""
    return storage.connect(DATA_ROOT, read_only=True)


def list_tables() -> pd.DataFrame:
    """Every table with its row and column counts, both read from metadata rather than scanned."""
    con = _connect()
    try:
        return con.execute("""
            SELECT t.table_name AS name,
                   (SELECT count(*) FROM information_schema.columns c
                    WHERE c.table_name = t.table_name) AS columns
            FROM information_schema.tables t
            WHERE t.table_schema = 'main' ORDER BY t.table_name
        """).df().assign(rows=lambda frame: [_row_count(con, name) for name in frame['name']])
    finally:
        con.close()


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """How many rows a table holds. DuckDB answers this from metadata, whatever the size."""
    return con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]


def describe_table(name: str) -> dict:
    """One table's columns, their types, how empty each is, and the span of its dates."""
    con = _connect()
    try:
        if not storage.table_exists(con, name):
            return {}
        rows = _row_count(con, name)
        columns = con.execute('SELECT column_name, data_type FROM information_schema.columns '
                              'WHERE table_name = ? ORDER BY ordinal_position', [name]).df()
        counted = rows <= NULL_COUNT_LIMIT
        nulls = _null_counts(con, name, columns['column_name'].tolist()) if counted else {}
        span = _date_span(con, name, columns)
        return {'name': name, 'rows': rows, 'columns': columns.to_dict('records'),
                'nulls': nulls, 'nulls_counted': counted, 'span': span,
                'tickers': _ticker_count(con, name, columns)}
    finally:
        con.close()


def _null_counts(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> dict[str, int]:
    """How many rows are empty in each column, in one pass over the table."""
    counts = ', '.join(f'count(*) - count("{column}")' for column in columns)
    row = con.execute(f'SELECT {counts} FROM "{table}"').fetchone()
    return dict(zip(columns, row))


def _date_span(con: duckdb.DuckDBPyConnection, table: str, columns: pd.DataFrame) -> tuple | None:
    """First and last value of the table's date column, read from zone maps rather than scanned."""
    types = dict(zip(columns['column_name'], columns['data_type']))
    kind = types.get('Date', '').upper()
    if not ('DATE' in kind or 'TIMESTAMP' in kind):
        return None
    first, last = con.execute(f'SELECT min("Date"), max("Date") FROM "{table}"').fetchone()
    return (first, last) if first is not None else None


def _ticker_count(con: duckdb.DuckDBPyConnection, table: str, columns: pd.DataFrame) -> int | None:
    """How many distinct tickers a table covers, where it has any."""
    if 'Ticker' not in set(columns['column_name']):
        return None
    return con.execute(f'SELECT count(DISTINCT "Ticker") FROM "{table}"').fetchone()[0]


def sample_rows(name: str, limit: int = SAMPLE_ROWS) -> pd.DataFrame:
    """The first rows of a table, for seeing what the values actually look like."""
    con = _connect()
    try:
        if not storage.table_exists(con, name):
            return pd.DataFrame()
        return con.execute(f'SELECT * FROM "{name}" LIMIT {int(limit)}').df()
    finally:
        con.close()


def run_query(sql: str, limit: int = QUERY_ROWS,
              timeout: float = QUERY_SECONDS) -> tuple[pd.DataFrame, bool, str | None]:
    """Run one read-only statement. Returns its rows, whether they were cut short, and any error.

    The statement runs as typed rather than wrapped in a `SELECT * FROM (...)`, which would turn
    `DESCRIBE` and `SHOW TABLES` into syntax errors and say nothing useful about a write. Writes are
    refused by DuckDB itself: a read-only connection rejects every statement of a modifying type,
    which is a stronger guarantee than reading SQL and hoping to recognise each dangerous form.

    Rows are taken `limit` at a time from the cursor rather than capped in the SQL, so a query that
    would return millions never materialises them. DuckDB has no statement timeout, so a watchdog
    thread interrupts one that outstays its welcome; without it a careless cross join would hold the
    page open indefinitely.
    """
    con = _connect()
    finished = threading.Event()
    watchdog = threading.Timer(timeout, lambda: None if finished.is_set() else con.interrupt())
    watchdog.start()
    try:
        cursor = con.execute(sql)
        rows = cursor.fetchmany(limit + 1)
        names = [column[0] for column in cursor.description]
        return pd.DataFrame(rows[:limit], columns=names), len(rows) > limit, None
    except duckdb.Error as failure:
        return pd.DataFrame(), False, str(failure)
    finally:
        finished.set()
        watchdog.cancel()
        con.close()
