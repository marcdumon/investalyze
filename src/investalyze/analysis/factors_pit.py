"""Point-in-time factor panel: the factors of `factors.py` on a (Ticker, as-of date) grid.

`factors.py` answers "what do these factors read today"; this module answers "what did they read on
each of these dates, using only what a reader could have known then". Two rules make that true.
Fundamentals are filtered to `IsRestated = FALSE`, the numbers as first reported, and every quarterly
window carries the publish date of its newest member, so a window becomes visible only once the last
filing behind it was out. Prices use the session grid, so an offset lands on a calendar position
rather than on a ticker's own row order.

The join is a validity-interval table rather than a loop over dates: each quarterly window is
computed once and stamped with the span over which it was the current one, then a range join attaches
it to every as-of date inside that span.

`version='restated'` reads the same rows `factors.py` reads, which is what makes the two comparable
on the latest date.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import custom_factors, panels
from investalyze.analysis.factors import (
    BUILTIN_FACTORS,
    BUILTIN_HIGHER_IS_BETTER,
    CASHFLOW_TTM,
    CUSTOM,
    FACTORS,
    INCOME_TTM,
    MIN_RISK_OBS,
    growth,
)
from investalyze.ingest import storage

AS_REPORTED = 'as_reported'
RESTATED = 'restated'
DATA_VERSIONS = (AS_REPORTED, RESTATED)

IDENTITY_COLS = ['name', 'sector', 'industry', 'last_close', 'mcap', 'mcap_bn']
PANEL_KEY = ['Ticker', 'Date', 'Version']

# a rank-based factor is a property of a ticker, a date and the names it was ranked against,
# so it cannot be stored per ticker and date. Everything else is an absolute quantity.
POPULATION_DEPENDENT = custom_factors.population_dependent(CUSTOM)
STORED_FACTORS = [factor for factor in FACTORS if factor not in POPULATION_DEPENDENT]

# balance columns the built-in factors already pull, so a ratio term naming one reuses it
BALANCE_LATEST = {'Total Equity': 'equity', 'Total Assets': 'assets'}

FAR_FUTURE = '9999-12-31'  # closes the last validity interval so a range join needs no NULL branch
CLOSE_LOOKBACK_DAYS = 10   # calendar days searched back for the last close, covering long weekends
HISTORY_DAYS = 600         # calendar days of prices loaded before the first as-of date, for 253-session windows
MIN_PRICE_COVERAGE = 0.01  # a ticker joins the price panel once it trades on this share of the calendar

# income columns the panel needs beyond the TTM sums: the current window's share count, the
# count four quarters earlier, and the basic-or-diluted count market cap is built from
_INCOME_EXTRAS = {
    'shares_latest': '"Shares (Diluted)"',
    'shares_prior': 'lag("Shares (Diluted)", 4) OVER win_seq',
    'shares_mcap': 'COALESCE("Shares (Basic)", "Shares (Diluted)")',
}


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


def annual_revenue_panel(con: duckdb.DuckDBPyConnection, *, as_of: pd.DatetimeIndex, version: str = AS_REPORTED,
                         tickers: list[str] | None = None) -> pd.DataFrame:
    """Per (as-of date, ticker): the 3-year revenue CAGR from published fiscal-year rows."""
    df = con.execute(f"""
        WITH a AS (
            SELECT Ticker, "Report Date"::DATE AS report_date, "Publish Date"::DATE AS publish_date, Revenue,
                   lag(Revenue, 3) OVER (PARTITION BY Ticker ORDER BY "Report Date", "Publish Date") AS rev_fy3
            FROM income
            WHERE Period = 'A' AND {_restated_clause(version)} AND Revenue IS NOT NULL
              AND {_ticker_clause(tickers)}
        ),
        v AS (
            SELECT *, max(publish_date) OVER (PARTITION BY Ticker ORDER BY report_date, publish_date
                                              ROWS UNBOUNDED PRECEDING) AS valid_from
            FROM a
        ),
        pit AS (
            SELECT *, COALESCE(lead(valid_from) OVER (PARTITION BY Ticker ORDER BY report_date, publish_date),
                               DATE '{FAR_FUTURE}') AS valid_to
            FROM v
        )
        SELECT g.as_of AS Date, p.Ticker, p.Revenue AS rev_fy0, p.rev_fy3
        FROM (SELECT unnest(?::DATE[]) AS as_of) g
        JOIN pit p ON g.as_of >= p.valid_from AND g.as_of < p.valid_to
    """, _ticker_params(tickers) + _grid_params(as_of)).df()
    df = _to_float(df, ['rev_fy0', 'rev_fy3'])
    valid = (df['rev_fy0'] > 0) & (df['rev_fy3'] > 0)
    df['revenue_3y'] = np.where(valid, (df['rev_fy0'] / df['rev_fy3']) ** (1 / 3) - 1, np.nan)
    return df[['Date', 'Ticker', 'revenue_3y']]


def dividend_panel(con: duckdb.DuckDBPyConnection, *, as_of: pd.DatetimeIndex,
                   tickers: list[str] | None = None) -> pd.DataFrame:
    """Per (as-of date, ticker): dividends over the trailing 365 days and the 365 days before that."""
    df = con.execute(f"""
        SELECT g.as_of AS Date, dv.Ticker,
               sum(CASE WHEN dv.Date > g.as_of - INTERVAL 365 DAY THEN dv.Dividend END) AS div_ttm,
               sum(CASE WHEN dv.Date <= g.as_of - INTERVAL 365 DAY THEN dv.Dividend END) AS div_prior
        FROM (SELECT unnest(?::DATE[]) AS as_of) g
        JOIN dividends dv ON dv.Date <= g.as_of AND dv.Date > g.as_of - INTERVAL 730 DAY
        WHERE {_ticker_clause(tickers).replace('Ticker', 'dv.Ticker')}
        GROUP BY g.as_of, dv.Ticker
    """, _grid_params(as_of) + _ticker_params(tickers)).df()
    return _to_float(df, ['div_ttm', 'div_prior'])


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


def price_factor_panel(con: duckdb.DuckDBPyConnection, *, as_of: pd.DatetimeIndex,
                       tickers: list[str], benchmark: str = panels.MARKET_TICKER) -> pd.DataFrame:
    """Per (as-of date, ticker): momentum and risk factors from the shared session grid.

    Offsets count calendar positions on the trading calendar, so a ticker missing the session an
    offset lands on reads NaN rather than reaching past the gap for a neighbouring price.
    """
    grid = pd.DatetimeIndex(as_of)
    start = (grid.min() - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat()
    end = grid.max().date().isoformat()
    calendar = panels.trading_calendar(con, start=start, end=end)
    closes = panels.close_panel(con, tickers, calendar, min_obs_frac=MIN_PRICE_COVERAGE)

    returns = closes.pct_change()
    market = panels.benchmark_returns(con, benchmark, start=start, end=end, kind='simple').reindex(closes.index)

    factor_frames = {
        'ret_12_1': closes.shift(21) / closes.shift(252) - 1.0,
        'ret_6m': closes / closes.shift(126) - 1.0,
        'high_52w': closes / closes.rolling(253, min_periods=MIN_RISK_OBS).max(),
        'vol_252': returns.rolling(252, min_periods=MIN_RISK_OBS).std() * np.sqrt(252),
        'beta_252': (returns.rolling(252, min_periods=MIN_RISK_OBS).cov(market)
                     .div(market.rolling(252, min_periods=MIN_RISK_OBS).var(), axis=0)),
    }

    on_grid = grid.intersection(closes.index)
    stacked = []
    for name, frame in factor_frames.items():
        series = frame.reindex(on_grid).stack(future_stack=True).rename(name)
        stacked.append(series)
    panel = pd.concat(stacked, axis=1).reset_index()
    panel.columns = ['Date', 'Ticker'] + list(factor_frames)
    return panel


def rebalance_dates(con: duckdb.DuckDBPyConnection, *, start: str, end: str, freq: str = 'M') -> pd.DatetimeIndex:
    """Trading sessions on which a run rebalances: the last session of each period in the range.

    Taken from `panels.trading_calendar`, so every returned date is a session the whole universe
    traded on rather than a calendar month end that may fall on a weekend.
    """
    calendar = panels.trading_calendar(con, start=start, end=end)
    if freq == 'D':
        return calendar
    period = pd.Series(calendar, index=calendar).groupby(calendar.to_period(freq)).max()
    return pd.DatetimeIndex(period.to_numpy(), name='Date')


def build_factor_panel(con: duckdb.DuckDBPyConnection, as_of: pd.DatetimeIndex, *,
                       tickers: list[str] | None = None, version: str = AS_REPORTED,
                       benchmark: str = panels.MARKET_TICKER) -> pd.DataFrame:
    """One row per (Ticker, as-of date) with every factor in FACTORS, from data published by that date.

    Rows come from the price panel, so a ticker appears on a date once it has a close there; the
    fundamentals join is left, so a name with no filings yet carries NaN factors rather than dropping
    out. The `Version` column records which fundamentals version produced the row.
    """
    _restated_clause(version)
    grid = pd.DatetimeIndex(as_of)
    if grid.empty:
        raise ValueError('as_of is empty')

    identity = identity_panel(con, as_of=grid, tickers=tickers)
    if identity.empty:
        raise ValueError('no priced tickers on the requested as-of dates')
    universe = tickers if tickers else sorted(identity['Ticker'].unique())

    existing = {'income': INCOME_TTM, 'cashflow': CASHFLOW_TTM, 'balance': BALANCE_LATEST}
    resolved = custom_factors.resolve_terms(CUSTOM, custom_factors.statement_schema(con))
    aliases = custom_factors.term_aliases(CUSTOM, resolved, existing)
    needed = custom_factors.extra_columns(CUSTOM, resolved, aliases, existing)

    df = identity.merge(price_factor_panel(con, as_of=grid, tickers=universe, benchmark=benchmark),
                        on=['Date', 'Ticker'], how='left')
    df = df.merge(quarterly_ttm(con, 'income', INCOME_TTM | needed['income'], as_of=grid, version=version,
                                tickers=universe, extras=_INCOME_EXTRAS), on=['Date', 'Ticker'], how='left')
    df = df.merge(quarterly_ttm(con, 'cashflow', CASHFLOW_TTM | needed['cashflow'], as_of=grid,
                                version=version, tickers=universe), on=['Date', 'Ticker'], how='left')
    df = df.merge(balance_panel(con, as_of=grid, version=version, tickers=universe,
                                extras=needed['balance']), on=['Date', 'Ticker'], how='left')
    df = df.merge(annual_revenue_panel(con, as_of=grid, version=version, tickers=universe),
                  on=['Date', 'Ticker'], how='left')
    df = df.merge(dividend_panel(con, as_of=grid, tickers=universe), on=['Date', 'Ticker'], how='left')

    df['mcap'] = df['shares_mcap'] * df['last_close']
    df['mcap_bn'] = (df['mcap'] / 1e9).round(3)
    df = _derive_factors(df)
    df = custom_factors.apply_ratios(df, CUSTOM, resolved, aliases)
    df = apply_derived(df)
    df['Version'] = version
    return df[PANEL_KEY + IDENTITY_COLS + STORED_FACTORS].sort_values(['Date', 'Ticker'],
                                                                     ignore_index=True)


def apply_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the factors computed from the frame's own columns, ranked within each as-of date.

    Called once on a build for the expressions that hold whatever population they are given, and
    again by every reader for the ones that do not, so a rank is always taken over the names the
    caller is actually working with rather than over whoever happened to be in the last build.
    """
    return custom_factors.apply_expressions(frame, CUSTOM, BUILTIN_FACTORS, BUILTIN_HIGHER_IS_BETTER,
                                            by='Date')


def _derive_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the joined raw aggregates into the factor columns, matching `factors.fundamental_factors`."""
    mcap = df['mcap']
    fcf_ttm = df['cfo_ttm'] + df['capex_ttm']            # capex is stored negative
    fcf_prior = df['cfo_prior'] + df['capex_prior']
    ev = mcap + df['debt'] - df['cash']
    ebitda = df['op_income_ttm'] - df['dep_amort_ttm']   # D&A is stored negative; subtract to add it back
    positive_equity = df['equity'] > 0

    df['earnings_yield'] = df['ni_common_ttm'] / mcap
    df['fcf_yield'] = fcf_ttm / mcap
    df['sales_yield'] = df['revenue_ttm'] / mcap
    df['book_to_market'] = df['equity'] / mcap
    df['ebitda_ev'] = np.where((ev > 0) & (ebitda > 0), ebitda / ev, np.nan)

    df['roe'] = np.where(positive_equity, df['net_income_ttm'] / df['equity'], np.nan)
    df['roa'] = np.where(df['assets'] > 0, df['net_income_ttm'] / df['assets'], np.nan)
    df['gross_margin'] = np.where(df['revenue_ttm'] > 0, df['gross_profit_ttm'] / df['revenue_ttm'], np.nan)
    df['op_margin'] = np.where(df['revenue_ttm'] > 0, df['op_income_ttm'] / df['revenue_ttm'], np.nan)
    op_margin_prior = np.where(df['revenue_prior'] > 0, df['op_income_prior'] / df['revenue_prior'], np.nan)
    df['margin_trend'] = df['op_margin'] - op_margin_prior
    df['accruals'] = np.where(df['assets'] > 0, (df['net_income_ttm'] - df['cfo_ttm']) / df['assets'], np.nan)
    df['debt_to_equity'] = np.where(positive_equity, df['debt'] / df['equity'], np.nan)
    df['interest_coverage'] = np.where(df['interest_expense_ttm'] != 0,   # interest expense is stored negative
                                       df['op_income_ttm'] / df['interest_expense_ttm'].abs(), np.nan)

    df['revenue_yoy'] = growth(df['revenue_ttm'], df['revenue_prior'])
    eps = pd.Series(np.where(df['shares_latest'] > 0, df['ni_common_ttm'] / df['shares_latest'], np.nan))
    eps_prior = pd.Series(np.where(df['shares_prior'] > 0, df['ni_common_prior'] / df['shares_prior'], np.nan))
    df['eps_yoy'] = growth(eps, eps_prior)
    df['fcf_yoy'] = growth(fcf_ttm, fcf_prior)
    df['share_change_1y'] = growth(df['shares_latest'], df['shares_prior'])
    df['div_growth'] = growth(df['div_ttm'], df['div_prior'])
    return df


FREQUENCY_DAYS = {'D': 1, 'W': 7, 'M': 30, 'Q': 91}   # typical gap between panel dates


def infer_frequency(dates: pd.DatetimeIndex) -> str | None:
    """The rebalance frequency a set of panel dates was built on, from the usual gap between them.

    The stored panel keeps no record of the frequency it was asked for, so it is read back off the
    dates themselves: the closest of the known frequencies to the median gap. Fewer than two dates
    says nothing, so nothing is what it returns.
    """
    if len(dates) < 2:
        return None
    gap = float(pd.Series(dates.sort_values()).diff().dt.days.median())
    return min(FREQUENCY_DAYS, key=lambda freq: abs(FREQUENCY_DAYS[freq] - gap))


def stored_dates(con: duckdb.DuckDBPyConnection, version: str,
                 tickers: list[str] | None = None) -> set[pd.Timestamp]:
    """The as-of dates the panel already holds, narrowed to a ticker list when one is given.

    Narrowed rather than global on purpose: a date built for other names is no use to these ones, so
    what counts as already built depends on who is asking.
    """
    if not storage.table_exists(con, 'factor_panel'):
        return set()
    clauses, params = ['Version = ?'], [version]
    if tickers is not None:
        clauses.append('Ticker IN (SELECT unnest(?::VARCHAR[]))')
        params.append(sorted(tickers))
    rows = con.execute(f'SELECT DISTINCT Date FROM factor_panel WHERE {" AND ".join(clauses)}',
                       params).fetchall()
    return {pd.Timestamp(row[0]) for row in rows}


def buildable_dates(con: duckdb.DuckDBPyConnection, grid: pd.DatetimeIndex,
                    tickers: list[str] | None = None) -> pd.DatetimeIndex:
    """The as-of dates that can produce rows at all: those on or after these names first traded.

    A panel row comes from a close, so a date before every requested name listed has nothing to
    hold. Seven solar companies asked for from 2000 have no row until FSLR listed in 2006, and
    treating those dates as missing would report a gap that no build can ever close.
    """
    if not storage.table_exists(con, 'prices'):
        return grid
    clause = 'WHERE Ticker IN (SELECT unnest(?::VARCHAR[]))' if tickers is not None else ''
    params = [sorted(tickers)] if tickers is not None else []
    first = con.execute(f'SELECT min(Date) FROM prices {clause}', params).fetchone()[0]
    if first is None:
        return pd.DatetimeIndex([], name=grid.name)
    return grid[grid >= pd.Timestamp(first)]


def empty_factors(con: duckdb.DuckDBPyConnection, version: str,
                  tickers: list[str] | None = None) -> list[str]:
    """Stored factors with no value at all in the rows this version holds for these names.

    Scoped to the tickers asked about, because a factor can be well populated across the database
    and empty for one corner of it: none of a handful of solar companies has ever paid a dividend,
    so no dividend growth exists for them and no amount of building will produce one.
    """
    if not storage.table_exists(con, 'factor_panel'):
        return []
    stored = {row[0] for row in con.execute('DESCRIBE factor_panel').fetchall()}
    present = [factor for factor in STORED_FACTORS if factor in stored]
    if not present:
        return []
    clauses, params = ['Version = ?'], [version]
    if tickers is not None:
        clauses.append('Ticker IN (SELECT unnest(?::VARCHAR[]))')
        params.append(sorted(tickers))
    counts = ', '.join(f'count("{factor}")' for factor in present)
    row = con.execute(f'SELECT {counts} FROM factor_panel WHERE {" AND ".join(clauses)}',
                      params).fetchone()
    return [factor for factor, filled in zip(present, row) if not filled]


def panel_coverage(con: duckdb.DuckDBPyConnection, version: str = AS_REPORTED) -> dict | None:
    """What the stored panel actually holds for one version, or None when it holds nothing.

    Read from the rows rather than from any record of what was asked for, so a panel built in two
    passes reports the span of both, and a factor added to the library after the last build reports
    as having no values.
    """
    if not storage.table_exists(con, 'factor_panel'):
        return None
    stored = {row[0] for row in con.execute('DESCRIBE factor_panel').fetchall()}
    rows = con.execute("""
        SELECT count(DISTINCT Date), count(DISTINCT Ticker), min(Date), max(Date)
        FROM factor_panel WHERE Version = ?
    """, [version]).fetchone()
    if not rows[0]:
        return None
    dates = con.execute('SELECT DISTINCT Date FROM factor_panel WHERE Version = ? ORDER BY Date',
                        [version]).df()['Date']

    populated = []
    for factor in STORED_FACTORS:
        if factor not in stored:
            continue
        filled = con.execute(f'SELECT count("{factor}") FROM factor_panel WHERE Version = ?',
                             [version]).fetchone()[0]
        if filled:
            populated.append(factor)
    return {'version': version, 'n_dates': rows[0], 'n_tickers': rows[1],
            'start': pd.Timestamp(rows[2]), 'end': pd.Timestamp(rows[3]),
            'frequency': infer_frequency(pd.DatetimeIndex(dates)),
            'factors': populated,
            'missing_factors': [factor for factor in STORED_FACTORS if factor not in populated]}


def _widen_stored_panel(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame) -> None:
    """Bring the stored table's columns into line with what a build now writes.

    Defining a new factor grows the panel by a column, and the merge writes into a table created by
    an earlier run, so a missing column is added rather than requiring a rebuild.

    A column for a rank-based factor is dropped instead. Those are no longer stored, because their
    value depends on which names were in the cross-section, and a table keyed by ticker and date has
    nowhere to record that. Rows written by an earlier build hold a number ranked against whatever
    universe that build covered, which is not comparable with the rest of its own column.

    The table is described rather than looked up by name, since a session with another database
    attached can see a `factor_panel` that the write target does not hold.

    A stored table may also still carry the column under its old name, `Vintage`: it read as
    unexplained finance jargon and was renamed to `Version`, so a column present under either name
    is renamed rather than treated as missing.
    """
    try:
        existing = {row[0] for row in con.execute('DESCRIBE factor_panel').fetchall()}
    except duckdb.CatalogException:
        return
    if 'Vintage' in existing and 'Version' not in existing:
        con.execute('ALTER TABLE factor_panel RENAME COLUMN Vintage TO Version')
        existing.discard('Vintage')
        existing.add('Version')
    for column in panel.columns:
        if column not in existing:
            con.execute(f'ALTER TABLE factor_panel ADD COLUMN "{column}" DOUBLE')
    for column in POPULATION_DEPENDENT:
        if column in existing:
            con.execute(f'ALTER TABLE factor_panel DROP COLUMN "{column}"')


def store_factor_panel(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame) -> int:
    """Merge the panel into `factor_panel`, keyed on ticker, date and version. Returns the table's row count."""
    if list(panel.columns) != PANEL_KEY + IDENTITY_COLS + STORED_FACTORS:
        raise ValueError('panel columns must be PANEL_KEY + IDENTITY_COLS + STORED_FACTORS')
    _widen_stored_panel(con, panel)
    return storage.store(con, 'factor_panel', panel, PANEL_KEY)
