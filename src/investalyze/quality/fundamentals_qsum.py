"""Quarters-vs-fiscal-year sum check on flow statements.

For each (Ticker, Market, IsRestated, Fiscal Year) with exactly four quarter values,
the quarter sum must match the annual row within `greatest(rel_tol * |FY|, abs_floor)`.
Balance is a stock statement and excluded by construction.
"""

import duckdb
import pandas as pd

from investalyze.quality.fundamentals_identities import ABS_FLOOR, REL_TOL

_SPECS = [
    ('income', 'Revenue'),
    ('income', 'Net Income'),
    ('cashflow', 'Net Cash from Operating Activities'),
]


def quarters_vs_fy(con: duckdb.DuckDBPyConnection, *, rel_tol: float = REL_TOL, abs_floor: int = ABS_FLOOR) -> pd.DataFrame:
    """Fiscal years where the four-quarter sum misses the annual value beyond tolerance."""
    frames = []
    for table, col in _SPECS:
        frames.append(con.execute(f"""
            WITH q AS (
                SELECT Ticker, Market, IsRestated, "Fiscal Year" AS fy, sum("{col}") AS q_sum
                FROM {table}
                WHERE Period = 'Q' AND "{col}" IS NOT NULL
                GROUP BY ALL
                HAVING count(*) = 4
            )
            SELECT '{table}' AS SrcTable, q.Ticker, NULL::DATE AS Date,
                   q.Market || '|A|' || q.fy || '|FY|' || q.IsRestated AS Key,
                   '{col}: Q1..Q4=' || q.q_sum || ' FY=' || a."{col}"
                   || coalesce(' diff=' || round(100.0 * abs(q.q_sum - a."{col}") / nullif(abs(a."{col}"), 0), 2) || '%', '') AS Details
            FROM q
            JOIN {table} a ON a.Ticker = q.Ticker AND a.Market = q.Market AND a.IsRestated = q.IsRestated
                          AND a."Fiscal Year" = q.fy AND a.Period = 'A' AND a."{col}" IS NOT NULL
            WHERE abs(q.q_sum - a."{col}") > greatest(? * abs(a."{col}"), ?)
        """, [rel_tol, abs_floor]).df())
    return pd.concat(frames, ignore_index=True)
