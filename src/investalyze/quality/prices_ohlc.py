"""Row-local checks over `prices` and `market_data`: sign, OHLC consistency, volume, bond bounds."""

import duckdb
import pandas as pd

MAX_ABS_YIELD = 50.0  # bond rows are yields in percent; levels beyond this look price-quoted

# NULL-safe Details: price fields can be NULL, and a plain || would null the whole string.
# Rounded to 4 decimals: vendor prices come in as float32 cast to double, which otherwise prints
# as noise (e.g. 10.8100004196167 instead of 10.81).
_OHLC = (
    """'O=' || coalesce(ROUND(O, 4)::VARCHAR, 'null') || ' H=' || coalesce(ROUND(H, 4)::VARCHAR, 'null')"""
    """ || ' L=' || coalesce(ROUND(L, 4)::VARCHAR, 'null') || ' C=' || coalesce(ROUND(C, 4)::VARCHAR, 'null')"""
)
_OHLC_AC = _OHLC + """ || ' AC=' || coalesce(ROUND(AC, 4)::VARCHAR, 'null')"""


def nonpositive_price(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rows with any non-positive price field; bonds excluded (negative yields are genuine)."""
    return con.execute(f"""
        SELECT 'prices' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, {_OHLC_AC} AS Details
        FROM prices
        WHERE O <= 0 OR H <= 0 OR L <= 0 OR C <= 0 OR AC <= 0
        UNION ALL
        SELECT 'market_data', Ticker, Date, NULL::VARCHAR, {_OHLC}
        FROM market_data
        WHERE AssetClass <> 'bonds' AND (O <= 0 OR H <= 0 OR L <= 0 OR C <= 0)
    """).df()


def ohlc_inconsistent(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rows where High is below Low or Open/Close falls outside [Low, High], in either table."""
    return con.execute(f"""
        SELECT 'prices' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, {_OHLC} AS Details
        FROM prices
        WHERE H < L OR O > H OR O < L OR C > H OR C < L
        UNION ALL
        SELECT 'market_data', Ticker, Date, NULL::VARCHAR, {_OHLC}
        FROM market_data
        WHERE H < L OR O > H OR O < L OR C > H OR C < L
    """).df()


def negative_volume(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Price rows with negative volume."""
    return con.execute("""
        SELECT 'prices' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, 'V=' || V AS Details
        FROM prices
        WHERE V < 0
    """).df()


def bond_yield_bound(con: duckdb.DuckDBPyConnection, *, max_abs_yield: float = MAX_ABS_YIELD) -> pd.DataFrame:
    """Bond rows whose |close| exceeds `max_abs_yield`: likely price-quoted series filed as yields."""
    return con.execute("""
        SELECT 'market_data' AS SrcTable, Ticker, Date, NULL::VARCHAR AS Key, 'C=' || C AS Details
        FROM market_data
        WHERE AssetClass = 'bonds' AND abs(C) > ?
    """, [max_abs_yield]).df()
