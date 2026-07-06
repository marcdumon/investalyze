"""Tolerance-based accounting-identity checks on the fundamentals tables.

SimFin stores expenses signed (negative), so every identity is additive: components
sum to their total. A row is flagged when the summed components miss the reported
total by more than `greatest(rel_tol * |total|, abs_floor)`.
"""

import duckdb
import pandas as pd

REL_TOL = 0.01       # fraction of the total's magnitude
ABS_FLOOR = 100_000  # whole USD; absorbs SimFin's rounding on small companies

_KEY_SQL = """Market || '|' || Period || '|' || "Fiscal Year" || '|' || "Fiscal Period" || '|' || IsRestated"""


def _sum_check(con: duckdb.DuckDBPyConnection, table: str, required: list[str], optional: list[str],
               total: str, rel_tol: float, abs_floor: int, label: str) -> pd.DataFrame:
    """Rows of `table` where the required components plus optional ones (NULL as 0) miss `total` beyond tolerance."""
    terms = [f'"{c}"' for c in required] + [f'coalesce("{c}", 0)' for c in optional]
    lhs = ' + '.join(terms)
    not_null = ' AND '.join(f'"{c}" IS NOT NULL' for c in [*required, total])
    sql = f"""
        SELECT '{table}' AS SrcTable, Ticker, NULL::DATE AS Date, {_KEY_SQL} AS Key,
               '{label}: lhs=' || ({lhs}) || ' rhs=' || "{total}"
               || coalesce(' diff=' || round(100.0 * abs(({lhs}) - "{total}") / nullif(abs("{total}"), 0), 2) || '%', '') AS Details
        FROM {table}
        WHERE {not_null}
          AND abs(({lhs}) - "{total}") > greatest(? * abs("{total}"), ?)
    """
    return con.execute(sql, [rel_tol, abs_floor]).df()


def balance_identity(con: duckdb.DuckDBPyConnection, *, rel_tol: float = REL_TOL, abs_floor: int = ABS_FLOOR) -> pd.DataFrame:
    """Balance rows where liabilities plus equity misses total assets beyond tolerance."""
    return _sum_check(con, 'balance', ['Total Liabilities', 'Total Equity'], [], 'Total Assets', rel_tol, abs_floor, 'liab+equity')


def balance_subtotals(con: duckdb.DuckDBPyConnection, *, rel_tol: float = REL_TOL, abs_floor: int = ABS_FLOOR) -> pd.DataFrame:
    """Balance rows where current plus noncurrent subtotals miss the assets or liabilities total."""
    specs = [
        (['Total Current Assets', 'Total Noncurrent Assets'], 'Total Assets', 'cur+noncur assets'),
        (['Total Current Liabilities', 'Total Noncurrent Liabilities'], 'Total Liabilities', 'cur+noncur liab'),
    ]
    frames = []
    for required, total, label in specs:
        frames.append(_sum_check(con, 'balance', required, [], total, rel_tol, abs_floor, label))
    return pd.concat(frames, ignore_index=True)


def income_chain(con: duckdb.DuckDBPyConnection, *, rel_tol: float = REL_TOL, abs_floor: int = ABS_FLOOR) -> pd.DataFrame:
    """Income rows breaking a link of the signed chain revenue -> gross profit -> operating -> pretax."""
    links = [
        (['Revenue', 'Cost of Revenue'], [], 'Gross Profit', 'rev+cogs'),
        (['Gross Profit', 'Operating Expenses'], ['Other Operating Income'], 'Operating Income (Loss)', 'gp+opex'),
        (['Operating Income (Loss)', 'Non-Operating Income (Loss)'], [], 'Pretax Income (Loss), Adj.', 'oi+nonop'),
    ]
    frames = []
    for required, optional, total, label in links:
        frames.append(_sum_check(con, 'income', required, optional, total, rel_tol, abs_floor, label))
    return pd.concat(frames, ignore_index=True)


def cashflow_identity(con: duckdb.DuckDBPyConnection, *, rel_tol: float = REL_TOL, abs_floor: int = ABS_FLOOR) -> pd.DataFrame:
    """Cashflow rows where operating + investing + financing (+ FX and discontinued ops) misses net change in cash."""
    required = ['Net Cash from Operating Activities', 'Net Cash from Investing Activities', 'Net Cash from Financing Activities']
    optional = ['Effect of Foreign Exchange Rates', 'Change in Cash from Disc. Operations and Other']
    return _sum_check(con, 'cashflow', required, optional, 'Net Change in Cash', rel_tol, abs_floor, 'op+inv+fin')
