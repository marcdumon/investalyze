"""Fix type: set one column on rows selected by ticker and inclusive date range.

With `value` configured the column is set to that number; with `value` omitted the column is
cleared to SQL NULL (for fields where any stored number would be a guess). An empty tickers
list applies the rule to every ticker.
"""

import re

import duckdb

from investalyze.cleaning.fix import Fix

_COLUMN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_ ().,-]*$')


def _quoted_column(fix: Fix) -> str:
    """The fix's column as a quoted SQL identifier; raises ValueError on a malformed name."""
    if fix.column is None or not _COLUMN_RE.match(fix.column):
        raise ValueError(f'set_value needs a plain column name, got {fix.column!r}')
    return f'"{fix.column}"'


def _predicate(fix: Fix) -> tuple[str, list]:
    """WHERE clause and bind params selecting the fix's target rows that still hold a different value."""
    clauses = [f'{_quoted_column(fix)} IS DISTINCT FROM ?']
    params: list = [fix.value]
    if fix.tickers:
        placeholders = ', '.join('?' for _ in fix.tickers)
        clauses.append(f'Ticker IN ({placeholders})')
        params += list(fix.tickers)
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
    """Set the column to the configured value (NULL when omitted) on the matching rows, returning rows changed."""
    where, params = _predicate(fix)
    return int(con.execute(
        f'UPDATE {fix.table} SET {_quoted_column(fix)} = ? WHERE {where}', [fix.value, *params],
    ).fetchone()[0])  # type: ignore[index]
