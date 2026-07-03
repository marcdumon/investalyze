"""Fix type: delete all rows for the given tickers within a date range.

For vendor data that is wrong at the source (e.g. backfilled proxy history) and comes back
on every full reload; running `apply` after each reload keeps the DB clean.
"""

import duckdb

from investalyze.cleaning.fix import Fix


def _predicate(fix: Fix) -> tuple[str, list]:
    """WHERE clause and bind params selecting the fix's target rows."""
    placeholders = ', '.join('?' for _ in fix.tickers)
    clauses = [f'Ticker IN ({placeholders})']
    params: list = list(fix.tickers)
    if fix.start is not None:
        clauses.append('Date >= ?')
        params.append(fix.start)
    if fix.end is not None:
        clauses.append('Date <= ?')
        params.append(fix.end)
    return ' AND '.join(clauses), params


def detect(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Count of rows currently matching the fix's predicate."""
    where, params = _predicate(fix)
    return int(con.execute(f'SELECT count(*) FROM {fix.table} WHERE {where}', params).fetchone()[0])  # type: ignore[index]


def apply(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Delete the matching rows, returning the number of rows deleted."""
    where, params = _predicate(fix)
    return int(con.execute(f'DELETE FROM {fix.table} WHERE {where}', params).fetchone()[0])  # type: ignore[index]
