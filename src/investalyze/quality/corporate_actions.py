"""Checks over `dividends` and `splits`: sign, dividend size vs close, split ratio validity."""

import duckdb
import pandas as pd

MAX_DIVIDEND_FRAC = 0.25  # dividends above this fraction of the same-day close look like data errors


def nonpositive_dividend(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Dividend rows with a non-positive amount."""
    return con.execute("""
        SELECT 'dividends' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, 'Dividend=' || Dividend AS Details
        FROM dividends
        WHERE Dividend <= 0
    """).df()


def oversized_dividend(con: duckdb.DuckDBPyConnection, *, max_dividend_frac: float = MAX_DIVIDEND_FRAC) -> pd.DataFrame:
    """Dividends above `max_dividend_frac` of the same-day close; rows without a price row are skipped."""
    return con.execute("""
        SELECT 'dividends' AS SrcTable, d.Ticker, d.Date, NULL::VARCHAR AS Key,
               'Dividend=' || d.Dividend || ' C=' || p.C
               || ' (' || round(100 * d.Dividend / p.C, 1) || '%)' AS Details
        FROM dividends d
        JOIN prices p ON p.Ticker = d.Ticker AND p.Date = d.Date
        WHERE p.C > 0 AND d.Dividend > ? * p.C
    """, [max_dividend_frac]).df()


def invalid_split_ratio(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Split rows whose ratio is non-positive or exactly 1 (a no-op split)."""
    return con.execute("""
        SELECT 'splits' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, 'Ratio=' || Ratio AS Details
        FROM splits
        WHERE Ratio <= 0 OR Ratio = 1
    """).df()
