"""Aligned return panels for cross-sectional work: a shared trading calendar, wide price/return matrices, z-scores.

Everything that touches DuckDB for the cross-sectional studies lives here; `notebooks/eigen.py` holds the math and takes only
matrices. Splitting them keeps the numerics testable without a database.

The calendar is derived from a minimum share of the busiest session's ticker count, never from the raw
union of dates. In the live DB, 2026-06-19 (Juneteenth) carries exactly 3 rows out of ~6,000, and taking
the union would let that one date reduce every ticker in the panel to incomplete.
"""

from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd

MIN_DATE_COVERAGE = 0.5  # a date joins the calendar once this share of nearby sessions' typical ticker count trades
LOCAL_WINDOW = 21        # sessions averaged to form that local reference count
MARKET_TICKER = '^SPX'


class Standardized(NamedTuple):
    """Z-scored returns alongside the per-asset mean and volatility used to produce them.

    `vol` is what converts a loading back into a tradable weight, so it is returned rather than
    discarded inside the scaling step.
    """

    z: pd.DataFrame
    mean: pd.Series
    vol: pd.Series


def trading_calendar(con: duckdb.DuckDBPyConnection, *, start: str | None = None, end: str | None = None,
                     min_coverage: float = MIN_DATE_COVERAGE,
                     local_window: int = LOCAL_WINDOW) -> pd.DatetimeIndex:
    """Sessions whose priced-ticker count reaches `min_coverage` of the count typical of nearby sessions.

    Holidays that carry a handful of stray rows are the reason this exists: they look like real sessions
    to a `GROUP BY Date` but no ticker has data for them.

    The reference count is a centred rolling median over `local_window` sessions, not the busiest session
    in the window. The number of listed securities grows over time, from 2,658 priced on a typical 2010
    session to 6,369 in 2026, so measuring against a single global maximum discards years of complete
    early history along with the holidays it is meant to remove. A median is used rather than a mean so
    that the holidays themselves do not drag the reference down.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError(f'min_coverage must be in (0, 1], got {min_coverage}')
    if local_window < 1:
        raise ValueError(f'local_window must be at least 1 session, got {local_window}')

    counts = con.execute(f"""
        SELECT Date, count(*) AS n
        FROM prices
        WHERE AC IS NOT NULL AND AC > 0 {_date_clause(start, end)}
        GROUP BY Date
        ORDER BY Date
    """, _date_params(start, end)).df()
    if counts.empty:
        raise ValueError('no priced sessions in the requested window')

    reference = counts['n'].rolling(local_window, center=True, min_periods=1).median()
    kept = counts.loc[counts['n'] >= min_coverage * reference, 'Date']
    return pd.DatetimeIndex(kept, name='Date')


RANKINGS = ('liquidity', 'mcap', 'ticker')


def complete_universe(con: duckdb.DuckDBPyConnection, calendar: pd.DatetimeIndex, *, n_top: int | None = None,
                      rank_by: str = 'liquidity', require_sector: bool = True,
                      candidates: list[str] | None = None) -> list[str]:
    """Tickers priced on every session of `calendar`, cut to the best `n_top` by the chosen ranking.

    `candidates` restricts the search to a supplied list, which is how a hand-picked selection or a
    saved universe from `data/universes/` enters; leaving it out searches every equity in the DB.

    Rankings:
        liquidity   median daily dollar volume over the window. The default, because a stock that does
                    not trade every day carries its previous close forward, and that flat close enters
                    the panel as a return of exactly zero. Fake zeros pull correlations toward zero,
                    which is the one quantity this study is built on. Measured over the last five
                    years, the top 400 names by volume show a flat close on 0.24% of days against
                    5.03% for the bottom third.
        mcap        latest reported share count times latest close. The convention for index work, and
                    it overlaps the liquidity ranking heavily; where the two differ it favours large
                    companies with a small tradable float.
        ticker      alphabetical, i.e. no preference at all, for when `candidates` is already the
                    answer and only the completeness filter is wanted.
    """
    return _ranked_universe(con, calendar, min_sessions=len(calendar), n_top=n_top, rank_by=rank_by,
                            require_sector=require_sector, candidates=candidates)


def traded_universe(con: duckdb.DuckDBPyConnection, calendar: pd.DatetimeIndex, *,
                    n_top: int | None = None, rank_by: str = 'liquidity', require_sector: bool = True,
                    candidates: list[str] | None = None,
                    min_sessions_frac: float = 0.0) -> list[str]:
    """Tickers that traded at some point in `calendar`, whether or not they traded throughout.

    This is the universe a backtest wants. Requiring a name on every session, as
    `complete_universe` does for the correlation studies it was written for, drops every company
    that listed partway through and every one that stopped: exactly the names whose absence is
    survivorship bias. The engine already declines to trade a name on a session it has no price
    for, and liquidates a holding whose name stops trading, so neither case needs excluding here.

    `min_sessions_frac` raises the bar if a run wants some minimum tenure; the default asks only
    that a name traded at all, since a tenure rule is itself a filter on survival.
    """
    minimum = max(1, round(min_sessions_frac * len(calendar)))
    return _ranked_universe(con, calendar, min_sessions=minimum, n_top=n_top, rank_by=rank_by,
                            require_sector=require_sector, candidates=candidates)


def _ranked_universe(con: duckdb.DuckDBPyConnection, calendar: pd.DatetimeIndex, *, min_sessions: int,
                     n_top: int | None, rank_by: str, require_sector: bool,
                     candidates: list[str] | None) -> list[str]:
    """Names trading on at least `min_sessions` of the calendar, ordered by the chosen ranking.

    The liquidity median is taken over the sessions a name actually traded, so a name present for
    part of the window is ranked on how it traded while it existed rather than penalised for the
    sessions it did not.
    """
    if rank_by not in RANKINGS:
        raise ValueError(f'rank_by must be one of {RANKINGS}, got {rank_by!r}')

    sector_join = 'JOIN companies s ON s.Ticker = p.Ticker AND s.Sector IS NOT NULL' if require_sector else ''
    candidate_filter = 'AND p.Ticker IN (SELECT unnest(?::VARCHAR[]))' if candidates is not None else ''
    params: list = [list(calendar.date)]
    if candidates is not None:
        params.append(candidates)
    params.append(min_sessions)

    ranked = con.execute(f"""
        WITH complete AS (
            SELECT p.Ticker, median(p.C * p.V) AS liquidity
            FROM prices p
            {sector_join}
            WHERE p.AC IS NOT NULL AND p.AC > 0 AND p.Date IN (SELECT unnest(?::DATE[])) {candidate_filter}
            GROUP BY p.Ticker
            HAVING count(*) >= ?
        )
        SELECT c.Ticker, c.liquidity, shares.n * close.last * 1.0 AS mcap
        FROM complete c
        LEFT JOIN (SELECT Ticker, arg_max(C, Date) AS last FROM prices GROUP BY Ticker) close USING (Ticker)
        LEFT JOIN (SELECT Ticker, arg_max(COALESCE("Shares (Basic)", "Shares (Diluted)"), ("Report Date", IsRestated)) AS n
                   FROM income WHERE Period = 'Q' GROUP BY Ticker) shares USING (Ticker)
    """, params).df()

    if rank_by == 'ticker':
        ordered = ranked.sort_values('Ticker')
    else:
        ordered = ranked.sort_values(rank_by, ascending=False, na_position='last')
    tickers = ordered['Ticker'].tolist()
    return tickers if n_top is None else tickers[:n_top]


def close_panel(con: duckdb.DuckDBPyConnection, tickers: list[str], calendar: pd.DatetimeIndex, *,
                min_obs_frac: float = 1.0) -> pd.DataFrame:
    """Wide adjusted-close matrix, dates down the rows and tickers across, reindexed onto `calendar`.

    Rows with a NULL or non-positive adjusted close are excluded at the source, so they surface as gaps
    rather than as prices a return could be taken from. Tickers observed on less than `min_obs_frac` of
    the calendar are dropped; the default keeps only complete series, which needs no fill policy.
    """
    if not tickers:
        raise ValueError('tickers is empty')
    if not 0 < min_obs_frac <= 1:
        raise ValueError(f'min_obs_frac must be in (0, 1], got {min_obs_frac}')

    long = con.execute("""
        SELECT Ticker, Date, AC
        FROM prices
        WHERE AC IS NOT NULL AND AC > 0
          AND Ticker IN (SELECT unnest(?::VARCHAR[]))
          AND Date IN (SELECT unnest(?::DATE[]))
    """, [tickers, list(calendar.date)]).df()
    if long.empty:
        raise ValueError('no priced rows for the requested tickers on this calendar')

    wide = long.pivot(index='Date', columns='Ticker', values='AC')
    wide.index = pd.DatetimeIndex(wide.index, name='Date')
    wide = wide.reindex(calendar)

    coverage = wide.notna().mean()
    kept = coverage[coverage >= min_obs_frac].index
    if len(kept) == 0:
        raise ValueError(f'no ticker reaches min_obs_frac={min_obs_frac} on this calendar')
    return wide[sorted(kept)]


def returns_panel(closes: pd.DataFrame, *, kind: str = 'log') -> pd.DataFrame:
    """Daily returns of a wide close panel, differenced on the calendar grid.

    Differencing on the grid rather than on each ticker's own observed sequence means a one-day hole
    yields two missing returns instead of quietly splicing a two-day move into a single bar.

    `kind='log'` suits estimation: additive across time, symmetric, thinner-tailed. `kind='simple'` is
    what a portfolio return is linear in, so eigenportfolio performance must be rebuilt from it.
    """
    if kind == 'log':
        returns = np.log(closes).diff()
    elif kind == 'simple':
        returns = closes.pct_change()
    else:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")
    return returns.iloc[1:]


def standardize(returns: pd.DataFrame) -> Standardized:
    """Z-score each ticker's returns, so PCA reads the correlation structure rather than the loudest names.

    Without it the first component tracks whichever assets happen to be most volatile.
    """
    vol = returns.std(ddof=1)
    flat = vol[(vol == 0) | vol.isna()].index.tolist()
    if flat:
        raise ValueError(f'{len(flat)} ticker(s) have no return variation and would produce NaN: {flat[:5]}')

    mean = returns.mean()
    return Standardized(z=(returns - mean) / vol, mean=mean, vol=vol)


def benchmark_returns(con: duckdb.DuckDBPyConnection, ticker: str = MARKET_TICKER, *, start: str | None = None,
                      end: str | None = None, kind: str = 'log') -> pd.Series:
    """Daily returns of one `market_data` series, for checking a component against a known index.

    Index coverage runs to different dates than `prices`: as of 2026-08, `^SPX` stops at 2026-07-02 and
    `^DJI` at 2026-07-10, while `^NDQ` and `^NDX` are current. Callers should join on dates and report
    the overlap rather than assume the benchmark spans the panel.
    """
    rows = con.execute(f"""
        SELECT Date, C
        FROM market_data
        WHERE Ticker = ? AND C IS NOT NULL AND C > 0 {_date_clause(start, end)}
        ORDER BY Date
    """, [ticker, *_date_params(start, end)]).df()
    if rows.empty:
        raise ValueError(f'no market_data rows for {ticker!r} in the requested window')

    close = pd.Series(rows['C'].to_numpy(dtype=float), index=pd.DatetimeIndex(rows['Date'], name='Date'), name=ticker)
    returns = np.log(close).diff() if kind == 'log' else close.pct_change()
    return returns.dropna()


def sector_labels(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> pd.Series:
    """Sector per ticker from the `companies` table, for attributing loadings to industry groups."""
    rows = con.execute("""
        SELECT Ticker, Sector
        FROM companies
        WHERE Ticker IN (SELECT unnest(?::VARCHAR[])) AND Sector IS NOT NULL
    """, [tickers]).df()
    return pd.Series(rows['Sector'].to_numpy(), index=pd.Index(rows['Ticker'], name='Ticker'), name='sector')


def rolling_windows(calendar: pd.DatetimeIndex, *, window: int, step: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start and end dates of each rolling window of `window` sessions, advanced `step` sessions at a time.

    Returned as dates rather than slices so the caller can label a decomposition by the window it came from.
    """
    if window < 2:
        raise ValueError(f'window must be at least 2 sessions, got {window}')
    if step < 1:
        raise ValueError(f'step must be at least 1 session, got {step}')
    if window > len(calendar):
        raise ValueError(f'window of {window} sessions exceeds the {len(calendar)}-session calendar')

    windows = []
    for stop in range(window, len(calendar) + 1, step):
        windows.append((calendar[stop - window], calendar[stop - 1]))
    return windows


def _date_clause(start: str | None, end: str | None) -> str:
    """SQL fragment bounding a Date column, matching the parameters `_date_params` supplies."""
    clause = ''
    if start is not None:
        clause += ' AND Date >= ?'
    if end is not None:
        clause += ' AND Date <= ?'
    return clause


def _date_params(start: str | None, end: str | None) -> list[str]:
    """Bound-parameter values for `_date_clause`, in the order it writes its placeholders."""
    return [value for value in (start, end) if value is not None]
