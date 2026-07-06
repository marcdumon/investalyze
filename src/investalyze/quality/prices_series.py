"""Per-series checks over `prices`: extreme daily returns, stale runs, date gaps.

All three use raw close (AC rewrites history on every dividend) and compare
consecutive rows per ticker in date order.
"""

import math

import duckdb
import pandas as pd

MAX_ABS_LOG_RETURN = math.log(2)  # flag daily moves beyond 2x up or 0.5x down
MIN_STALE_RUN = 20                # consecutive identical closes that suggest a stale/dead series
MAX_GAP_DAYS = 30                 # calendar days between consecutive rows that suggest a missing chunk


def extreme_return(con: duckdb.DuckDBPyConnection, *, max_abs_log_return: float = MAX_ABS_LOG_RETURN) -> pd.DataFrame:
    """Days where |log return| exceeds the threshold and no split explains the move.

    Single-day glitches are tagged '(spike-and-revert)' when the next close lands back
    within a quarter of the jump's magnitude.
    """
    return con.execute("""
        WITH r AS (
            SELECT Ticker, Date, C,
                   lag(C) OVER w AS prev_c,
                   lead(C) OVER w AS next_c
            FROM prices
            WHERE C > 0
            WINDOW w AS (PARTITION BY Ticker ORDER BY Date)
        )
        SELECT 'prices' AS SrcTable, r.Ticker, r.Date, NULL::VARCHAR AS Key,
               'ret=' || round(100 * (r.C / r.prev_c - 1), 1) || '%'
               || CASE WHEN r.next_c IS NOT NULL AND abs(ln(r.next_c / r.prev_c)) < 0.25 * abs(ln(r.C / r.prev_c))
                       THEN ' (spike-and-revert)' ELSE '' END AS Details
        FROM r
        LEFT JOIN splits s ON s.Ticker = r.Ticker AND s.Date = r.Date
        WHERE r.prev_c > 0 AND s.Ticker IS NULL AND abs(ln(r.C / r.prev_c)) > ?
    """, [max_abs_log_return]).df()


def stale_run(con: duckdb.DuckDBPyConnection, *, min_stale_run: int = MIN_STALE_RUN) -> pd.DataFrame:
    """Runs of at least `min_stale_run` consecutive identical closes; one finding per run, Date = run end."""
    return con.execute("""
        WITH marks AS (
            SELECT Ticker, Date, C,
                   CASE WHEN C = lag(C) OVER (PARTITION BY Ticker ORDER BY Date) THEN 0 ELSE 1 END AS new_run
            FROM prices
        ), runs AS (
            SELECT Ticker, Date, C,
                   sum(new_run) OVER (PARTITION BY Ticker ORDER BY Date) AS run_id
            FROM marks
        )
        SELECT 'prices' AS SrcTable, Ticker, max(Date) AS Date, NULL::VARCHAR AS Key,
               count(*) || ' identical closes ' || min(Date) || ' .. ' || max(Date) || ' (C=' || min(C) || ')' AS Details
        FROM runs
        GROUP BY Ticker, run_id
        HAVING count(*) >= ?
    """, [min_stale_run]).df()


def date_gap(con: duckdb.DuckDBPyConnection, *, max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Consecutive rows more than `max_gap_days` calendar days apart; Date = the row after the gap."""
    return con.execute("""
        WITH d AS (
            SELECT Ticker, Date, lag(Date) OVER (PARTITION BY Ticker ORDER BY Date) AS prev_d
            FROM prices
        )
        SELECT 'prices' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key,
               'gap ' || (Date - prev_d) || ' days after ' || prev_d AS Details
        FROM d
        WHERE Date - prev_d > ?
    """, [max_gap_days]).df()
