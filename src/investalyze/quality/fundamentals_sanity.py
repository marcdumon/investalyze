"""Hard-invariant and sign checks on the fundamentals tables."""

import duckdb
import pandas as pd

_STATEMENT_TABLES = ['income', 'balance', 'cashflow']

_KEY_SQL = """Market || '|' || Period || '|' || "Fiscal Year" || '|' || "Fiscal Period" || '|' || IsRestated"""


def hard_invariants(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rows with non-positive share counts (all statements) or negative total assets (balance)."""
    frames = []
    for table in _STATEMENT_TABLES:
        frames.append(con.execute(f"""
            SELECT '{table}' AS SrcTable, Ticker, NULL::DATE AS Date, {_KEY_SQL} AS Key,
                   'Shares (Basic)=' || coalesce("Shares (Basic)"::VARCHAR, 'null')
                   || ' Shares (Diluted)=' || coalesce("Shares (Diluted)"::VARCHAR, 'null') AS Details
            FROM {table}
            WHERE "Shares (Basic)" <= 0 OR "Shares (Diluted)" <= 0
        """).df())
    frames.append(con.execute(f"""
        SELECT 'balance' AS SrcTable, Ticker, NULL::DATE AS Date, {_KEY_SQL} AS Key,
               'Total Assets=' || "Total Assets" AS Details
        FROM balance
        WHERE "Total Assets" < 0
    """).df())
    return pd.concat(frames, ignore_index=True)


def negative_revenue(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Income rows with negative revenue; legitimate for some financials, so review per ticker."""
    return con.execute(f"""
        SELECT 'income' AS SrcTable, Ticker, NULL::DATE AS Date, {_KEY_SQL} AS Key,
               'Revenue=' || Revenue AS Details
        FROM income
        WHERE Revenue < 0
    """).df()
