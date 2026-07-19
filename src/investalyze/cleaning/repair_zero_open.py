"""Fix type: restore a nonpositive Open on bars anchored by intact H/C/AC.

For vendor rows where O is bad (0 or negative, i.e. no open was recorded) while H, C and AC
are positive, the repair sets O = C so the bar stays self-consistent. An empty tickers list
applies the rule to every ticker.
"""

import duckdb

from investalyze.cleaning.fix import Fix

_BROKEN = 'O <= 0 AND H > 0 AND C > 0 AND AC > 0'


def _predicate(fix: Fix) -> tuple[str, list]:
    """WHERE clause and bind params selecting the fix's target rows."""
    clauses = [_BROKEN]
    params: list = []
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
    """Set O = C on the matching rows, returning the number of rows changed."""
    where, params = _predicate(fix)
    return int(con.execute(f'UPDATE {fix.table} SET O = C WHERE {where}', params).fetchone()[0])  # type: ignore[index]
