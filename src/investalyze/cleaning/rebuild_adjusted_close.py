"""Fix type: rebuild the adjusted close from raw closes + dividends, excluding unrepresentable events.

The total-return back-adjustment (the same method the yahoo ingest derives AC with) multiplies,
for every ex-date, the factor (1 - dividend/previous close) into all earlier days. A recorded
distribution at or above the previous close makes that factor nonpositive and flips the sign of
the entire earlier series; the rebuild keeps every other event and excludes those, whose true
price effect cannot be derived from prices alone. Tickers must be listed explicitly; start/end
are ignored because the method always spans the full series.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.cleaning.fix import Fix


def rebuild_series(close: pd.Series, dividends: pd.Series) -> tuple[pd.Series, list]:
    """(rebuilt AC, excluded ex-dates) for one ticker's ascending close series.

    `dividends` is indexed like `close` and 0 where there is no ex-date event; an event whose
    factor would be nonpositive (or whose previous close is missing) lands in the excluded list.
    """
    close = close.sort_index()
    dividends = dividends.reindex(close.index).fillna(0.0)
    prev_close = close.shift(1)
    factor = pd.Series(1.0, index=close.index)
    excluded: list = []
    for ex_date in close.index[dividends > 0]:
        event_factor = 1 - dividends[ex_date] / prev_close[ex_date]
        if pd.notna(event_factor) and event_factor > 0:
            factor[ex_date] = event_factor
        else:
            excluded.append(ex_date)
    after = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * after, excluded


def _changed_rows(con: duckdb.DuckDBPyConnection, fix: Fix, ticker: str) -> pd.DataFrame:
    """(Ticker, Date, NewAC) for the rows whose rebuilt AC differs from the stored one."""
    prices = con.execute(f'SELECT Date, C, AC FROM {fix.table} WHERE Ticker = ? ORDER BY Date', [ticker]).df()
    if prices.empty:
        return pd.DataFrame(columns=['Ticker', 'Date', 'NewAC'])
    divs = con.execute('SELECT Date, Dividend FROM dividends WHERE Ticker = ? ORDER BY Date', [ticker]).df()
    close = prices.set_index('Date')['C']
    dividends = divs.set_index('Date')['Dividend'] if len(divs) else pd.Series(dtype=float)
    rebuilt, _excluded = rebuild_series(close, dividends)
    changed = ~np.isclose(rebuilt.to_numpy(), prices.set_index('Date')['AC'].to_numpy(), rtol=1e-6, atol=1e-9)
    out = pd.DataFrame({'Ticker': ticker, 'Date': rebuilt.index[changed], 'NewAC': rebuilt.to_numpy()[changed]})
    return out


def detect(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Count of rows whose adjusted close would change."""
    return sum(len(_changed_rows(con, fix, ticker)) for ticker in fix.tickers)


def apply(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Write the rebuilt adjusted closes, returning the number of rows changed."""
    changed = pd.concat([_changed_rows(con, fix, ticker) for ticker in fix.tickers], ignore_index=True)
    if changed.empty:
        return 0
    con.register('rebuilt_ac', changed)
    try:
        con.execute(
            f'UPDATE {fix.table} SET AC = r.NewAC FROM rebuilt_ac r '
            f'WHERE {fix.table}.Ticker = r.Ticker AND {fix.table}.Date = r.Date'
        )
    finally:
        con.unregister('rebuilt_ac')
    return len(changed)
