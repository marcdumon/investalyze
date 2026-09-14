"""Point-in-time readers: statement line items and identity on a (Ticker, as-of date) grid.

Two rules keep every value knowable on its as-of date. Fundamentals are filtered to
`IsRestated = FALSE`, the numbers as first reported, and every quarterly window carries the publish date
of its newest member, so a window becomes visible only once the last filing behind it was out.

The join is a validity-interval table rather than a loop over dates: each quarterly window is computed
once and stamped with the span over which it was the current one, then a range join attaches it to
every as-of date inside that span.
"""

from enum import StrEnum

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import panels

AS_REPORTED = 'as_reported'
RESTATED = 'restated'
DATA_VERSIONS = (AS_REPORTED, RESTATED)

IDENTITY_COLS = ['name', 'sector', 'industry', 'last_close']

FAR_FUTURE = '9999-12-31'  # closes the last validity interval so a range join needs no NULL branch
CLOSE_LOOKBACK_DAYS = 10   # calendar days searched back for the last close, covering long weekends
HISTORY_DAYS = 600         # calendar days of prices loaded before the first as-of date, for 253-session windows
MIN_PRICE_COVERAGE = 0.01  # a ticker joins a price panel once it trades on this share of the calendar


def _restated_clause(version: str) -> str:
    """SQL predicate selecting one fundamentals version."""
    if version == AS_REPORTED:
        return 'NOT IsRestated'
    if version == RESTATED:
        return 'IsRestated'
    raise ValueError(f'version must be one of {DATA_VERSIONS}, got {version!r}')


def _ticker_clause(tickers: list[str] | None) -> str:
    """SQL predicate narrowing to a ticker list, or an always-true predicate when none is given.

    None means no filter; an empty list means no names, and must not collapse into the same thing.
    A universe that resolved to nothing would otherwise silently build over the whole database.
    """
    return 'TRUE' if tickers is None else 'Ticker IN (SELECT unnest(?::VARCHAR[]))'


def _ticker_params(tickers: list[str] | None) -> list:
    """Parameters matching `_ticker_clause`."""
    return [] if tickers is None else [tickers]


def _grid_params(as_of: pd.DatetimeIndex) -> list:
    """The as-of grid as a DuckDB DATE array parameter."""
    return [list(pd.DatetimeIndex(as_of).date)]


def _to_float(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast the named columns to float64, since DuckDB returns nullable Int64 when a sum is NULL."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype('float64')
    return df


def quarterly_ttm(con: duckdb.DuckDBPyConnection, table: str, columns: dict[str, str], *,
                  as_of: pd.DatetimeIndex, version: str = AS_REPORTED,
                  tickers: list[str] | None = None, extras: dict[str, str] | None = None) -> pd.DataFrame:
    """Per (as-of date, ticker): TTM and prior-year sums from the quarters published by that date.

    Output columns are `<alias>_ttm` and `<alias>_prior`, plus any `extras` aliases. A window holding
    fewer than 4 quarters reads NaN, matching `factors._ttm`. `extras` maps an alias to a SQL
    expression evaluated on the window's newest quarter, which may use the `win_seq` window.
    """
    extras = extras or {}
    sums = ',\n           '.join(
        f'sum("{source}") OVER win4 AS {alias}_ttm, sum("{source}") OVER win_prior AS {alias}_prior'
        for source, alias in columns.items()
    )
    extra_terms = ''.join(f',\n           {expression} AS {alias}' for alias, expression in extras.items())
    aliases = [f'{alias}_{suffix}' for alias in columns.values() for suffix in ('ttm', 'prior')]
    selected = ', '.join(f'p.{alias}' for alias in aliases + list(extras))

    df = con.execute(f"""
        WITH q AS (
            SELECT "Report Date"::DATE AS report_date, "Publish Date"::DATE AS publish_date, *
            FROM {table}
            WHERE Period = 'Q' AND {_restated_clause(version)} AND {_ticker_clause(tickers)}
        ),
        w AS (
            SELECT Ticker, report_date,
                   {sums},
                   count(*) OVER win4 AS n_recent,
                   count(*) OVER win_prior AS n_prior,
                   max(publish_date) OVER win4 AS avail{extra_terms}
            FROM q
            WINDOW win4 AS (PARTITION BY Ticker ORDER BY report_date, publish_date
                            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW),
                   win_prior AS (PARTITION BY Ticker ORDER BY report_date, publish_date
                                 ROWS BETWEEN 7 PRECEDING AND 4 PRECEDING),
                   win_seq AS (PARTITION BY Ticker ORDER BY report_date, publish_date)
        ),
        v AS (
            SELECT *, max(avail) OVER (PARTITION BY Ticker ORDER BY report_date
                                       ROWS UNBOUNDED PRECEDING) AS valid_from
            FROM w
        ),
        pit AS (
            SELECT *, COALESCE(lead(valid_from) OVER (PARTITION BY Ticker ORDER BY report_date),
                               DATE '{FAR_FUTURE}') AS valid_to
            FROM v
        )
        SELECT g.as_of AS Date, p.Ticker, p.n_recent, p.n_prior, {selected}
        FROM (SELECT unnest(?::DATE[]) AS as_of) g
        JOIN pit p ON g.as_of >= p.valid_from AND g.as_of < p.valid_to
    """, _ticker_params(tickers) + _grid_params(as_of)).df()

    df = _to_float(df, aliases + list(extras))
    recent = [f'{alias}_ttm' for alias in columns.values()]
    prior = [f'{alias}_prior' for alias in columns.values()]
    df.loc[df['n_recent'] < 4, recent] = np.nan
    df.loc[df['n_prior'] < 4, prior] = np.nan
    return df.drop(columns=['n_recent', 'n_prior'])


def balance_panel(con: duckdb.DuckDBPyConnection, *, as_of: pd.DatetimeIndex, version: str = AS_REPORTED,
                  tickers: list[str] | None = None, extras: dict[str, str] | None = None) -> pd.DataFrame:
    """Per (as-of date, ticker): equity, assets, total debt and cash from the latest published quarter.

    `extras` maps additional balance columns to aliases, which is how a user-defined ratio factor
    reaches a line item the built-in factors do not use.
    """
    extras = extras or {}
    extra_selects = ''.join(f',\n                   "{source}" AS {alias}' for source, alias in extras.items())
    extra_outputs = ''.join(f', p.{alias}' for alias in extras.values())
    df = con.execute(f"""
        WITH b AS (
            SELECT Ticker, "Report Date"::DATE AS report_date, "Publish Date"::DATE AS publish_date,
                   "Total Equity" AS equity, "Total Assets" AS assets,
                   COALESCE("Short Term Debt", 0) + COALESCE("Long Term Debt", 0) AS debt,
                   COALESCE("Cash & Cash Equivalents", 0) AS cash{extra_selects}
            FROM balance
            WHERE Period = 'Q' AND {_restated_clause(version)} AND {_ticker_clause(tickers)}
        ),
        v AS (
            SELECT *, max(publish_date) OVER (PARTITION BY Ticker ORDER BY report_date, publish_date
                                              ROWS UNBOUNDED PRECEDING) AS valid_from
            FROM b
        ),
        pit AS (
            SELECT *, COALESCE(lead(valid_from) OVER (PARTITION BY Ticker ORDER BY report_date, publish_date),
                               DATE '{FAR_FUTURE}') AS valid_to
            FROM v
        )
        SELECT g.as_of AS Date, p.Ticker, p.equity, p.assets, p.debt, p.cash{extra_outputs}
        FROM (SELECT unnest(?::DATE[]) AS as_of) g
        JOIN pit p ON g.as_of >= p.valid_from AND g.as_of < p.valid_to
    """, _ticker_params(tickers) + _grid_params(as_of)).df()
    return _to_float(df, ['equity', 'assets', 'debt', 'cash'] + list(extras.values()))


def identity_panel(con: duckdb.DuckDBPyConnection, *, as_of: pd.DatetimeIndex,
                   tickers: list[str] | None = None) -> pd.DataFrame:
    """Per (as-of date, ticker): company identity and the last close on or before that date.

    Company name, sector and industry come from `companies`, which holds one current row per ticker,
    so they are the same on every as-of date.
    """
    closes = con.execute(f"""
        SELECT g.as_of AS Date, p.Ticker, arg_max(p.C, p.Date) AS last_close
        FROM (SELECT unnest(?::DATE[]) AS as_of) g
        JOIN prices p ON p.Date <= g.as_of AND p.Date > g.as_of - INTERVAL {CLOSE_LOOKBACK_DAYS} DAY
        WHERE p.C IS NOT NULL AND p.C > 0 AND {_ticker_clause(tickers).replace('Ticker', 'p.Ticker')}
        GROUP BY g.as_of, p.Ticker
    """, _grid_params(as_of) + _ticker_params(tickers)).df()
    companies = con.execute(
        'SELECT Ticker, CompanyName AS name, Sector AS sector, Industry AS industry FROM companies'
    ).df()
    df = closes.merge(companies, on='Ticker', how='left')
    df['name'] = df['name'].fillna('')
    for column in ('sector', 'industry'):
        df[column] = df[column].fillna('unknown')
    return _to_float(df, ['last_close'])


class Rebalance(StrEnum):
    """How often a run rebalances, as the pandas period code of the sessions it keeps."""

    DAILY = 'D'
    MONTHLY = 'M'
    QUARTERLY = 'Q'

    @property
    def label(self) -> str:
        """The frequency in words."""
        return self.name.lower()


def rebalance_dates(con: duckdb.DuckDBPyConnection, *, start: str, end: str,
                    freq: Rebalance = Rebalance.MONTHLY) -> pd.DatetimeIndex:
    """Trading sessions on which a run rebalances: the last session of each period in the range.

    Taken from `panels.trading_calendar`, so every returned date is a session the whole universe
    traded on rather than a calendar month end that may fall on a weekend.
    """
    calendar = panels.trading_calendar(con, start=start, end=end)
    if freq == Rebalance.DAILY:
        return calendar
    period = pd.Series(calendar, index=calendar).groupby(calendar.to_period(freq)).max()
    return pd.DatetimeIndex(period.to_numpy(), name='Date')
