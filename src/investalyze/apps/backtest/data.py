"""Data access for the backtest page: stored runs, their curves, and their blotters.

Each read opens a short-lived read-only connection, matching the other pages. Nothing is cached: a
run finishing while the page is open should show up on the next refresh rather than after a restart.
"""

from pathlib import Path

import duckdb
import pandas as pd

from investalyze.analysis import custom_factors, factors_pit
from investalyze.analysis.factors import CUSTOM
from investalyze.apps import universes
from investalyze.backtest import writer
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'


def _connect() -> duckdb.DuckDBPyConnection:
    """A read-only connection to the project database."""
    return storage.connect(DATA_ROOT, read_only=True)


def _has_column(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Whether a stored table carries a column yet.

    The evaluation tables gained columns after they were first written, and the writer only widens
    them when a command next opens the database for writing. Reading has to cope until then.
    """
    return any(row[0] == column for row in con.execute(f'DESCRIBE {table}').fetchall())


def get_runs() -> pd.DataFrame:
    """Every stored run, newest first, with the headline statistics the table shows."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'backtest_runs'):
            return pd.DataFrame()
        return con.execute("""
            SELECT RunId, Signal, Kind, StartDate::VARCHAR AS StartDate, EndDate::VARCHAR AS EndDate,
                   NTickers, Benchmark, Cagr, Vol, Sharpe,
                   Sortino, MaxDrawdown, Calmar, Turnover, HitRate, NTrades, FinalEquity, TotalCosts,
                   SurvivorshipNote, strftime(CreatedAt, '%Y-%m-%d %H:%M') AS Created
            FROM backtest_runs ORDER BY CreatedAt DESC
        """).df()
    finally:
        con.close()


def get_equity(run_id: str) -> pd.DataFrame:
    """One run's equity curve, with the benchmark alongside it where the run recorded one."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'backtest_equity'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Date, Equity, Gross, Net, Cash, BenchmarkEquity
            FROM backtest_equity WHERE RunId = ? ORDER BY Date
        """, [run_id]).df()
    finally:
        con.close()


def get_trades(run_id: str, limit: int = 500) -> pd.DataFrame:
    """The most recent fills of one run."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'backtest_trades'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Date::VARCHAR AS Date, Ticker, Side, Shares, Price, Notional, Cost, Kind
            FROM backtest_trades WHERE RunId = ? ORDER BY Date DESC, Ticker LIMIT ?
        """, [run_id, limit]).df()
    finally:
        con.close()


def get_evaluated_universes() -> list[str]:
    """The universes an evaluation has been stored for, so one of them can be chosen to read."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_ic') or not _has_column(con, 'factor_ic', 'Universe'):
            return []
        rows = con.execute('SELECT DISTINCT Universe FROM factor_ic ORDER BY Universe').fetchall()
        return [row[0] for row in rows]
    finally:
        con.close()


def get_factor_ic(vintage: str = 'as_reported', universe: str | None = None) -> pd.DataFrame:
    """Mean information coefficient and its t-statistic per factor and horizon, for one universe."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_ic'):
            return pd.DataFrame()
        clauses = ['Vintage = ?', 'Ic IS NOT NULL']
        params: list = [vintage]
        if universe and _has_column(con, 'factor_ic', 'Universe'):
            clauses.append('Universe = ?')
            params.append(universe)
        return con.execute(f"""
            SELECT Factor, Horizon, count(*) AS Periods, avg(Ic) AS MeanIc, stddev_samp(Ic) AS StdIc,
                   avg(CASE WHEN Ic > 0 THEN 1.0 ELSE 0.0 END) AS HitRate, avg(NNames) AS MeanNames
            FROM factor_ic WHERE {' AND '.join(clauses)}
            GROUP BY Factor, Horizon ORDER BY Factor, Horizon
        """, params).df()
    finally:
        con.close()


def get_quantile_profile(factor: str, vintage: str = 'as_reported',
                         universe: str | None = None) -> pd.DataFrame:
    """Forward return of each score bucket for one factor, as a mean and as a median.

    Both are returned because a bucket's mean answers to a few extreme names while its median does
    not, and the two part company on a universe that was never filtered for price or liquidity.
    """
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_quantiles'):
            return pd.DataFrame()
        clauses = ['Vintage = ?', 'Factor = ?']
        params: list = [vintage, factor]
        if universe and _has_column(con, 'factor_quantiles', 'Universe'):
            clauses.append('Universe = ?')
            params.append(universe)
        median = ('avg(MedianReturn)' if _has_column(con, 'factor_quantiles', 'MedianReturn')
                  else 'CAST(NULL AS DOUBLE)')
        return con.execute(f"""
            SELECT Quantile, Horizon, avg(Return) AS MeanReturn, {median} AS MedianReturn,
                   count(*) AS Periods
            FROM factor_quantiles WHERE {' AND '.join(clauses)}
            GROUP BY Quantile, Horizon ORDER BY Horizon, Quantile
        """, params).df()
    finally:
        con.close()


def get_panel_coverage(vintage: str = 'as_reported') -> tuple[dict | None, dict | None]:
    """What the stored factor panel holds for a vintage, and the request behind its last build.

    Coverage is read from the rows, so it is true whatever produced them; the build record adds the
    universe and the time, which the rows do not carry.
    """
    con = _connect()
    try:
        coverage = factors_pit.panel_coverage(con, vintage)
        build = None
        if storage.table_exists(con, 'factor_panel_builds'):
            build = writer.latest_panel_build(con, vintage)
        return coverage, build
    finally:
        con.close()


def get_last_session() -> pd.Timestamp | None:
    """The last session prices exist for, which is as far as any run can actually reach."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'market_data'):
            return None
        row = con.execute('SELECT max(Date) FROM market_data').fetchone()
        return pd.Timestamp(row[0]) if row and row[0] is not None else None
    finally:
        con.close()


def panel_gaps(vintage: str, universe: str | None, start: str, end: str, freq: str) -> dict:
    """Everything about the stored panel that stands between a request and a result.

    Dates are counted for the requested names rather than for the panel as a whole: a date built for
    other tickers is no use to these ones. Factor definitions are compared against the fingerprint
    the last build recorded, since a redefined factor leaves the dates in place and the values wrong.

    The estimate comes from how long the last build took per row. Without a recorded build there is
    nothing to estimate from, and None says so rather than a number pulled from the air.
    """
    tickers = universes.load_universe(universe) if universe else None
    con = _connect()
    try:
        coverage = factors_pit.panel_coverage(con, vintage)
        # checked at the panel's own cadence, not the requested one: a session between panel
        # dates reads the most recent one before it, so a coarser panel is a choice, not a hole
        cadence = (coverage['frequency'] if coverage else None) or freq
        grid = factors_pit.rebalance_dates(con, start=start, end=end, freq=cadence)
        build = writer.latest_panel_build(con, vintage) \
            if storage.table_exists(con, 'factor_panel_builds') else None
        held = _panel_dates(con, vintage, tickers)
    finally:
        con.close()

    missing = [date for date in grid if date not in held]
    n_tickers = len(tickers) if tickers is not None else (coverage['n_tickers'] if coverage else 0)
    return {
        'requested_dates': len(grid),
        'missing_dates': len(missing),
        'n_tickers': n_tickers,
        'missing_factors': coverage['missing_factors'] if coverage else [],
        'changed_factors': custom_factors.changed_since(build['definitions'] if build else None, CUSTOM),
        'seconds_estimate': _build_estimate(build, len(missing) * n_tickers),
        'has_panel': coverage is not None,
        'panel_frequency': cadence,
        'requested_frequency': freq,
    }


def _panel_dates(con: duckdb.DuckDBPyConnection, vintage: str, tickers: list[str] | None) -> set:
    """The distinct panel dates stored for one vintage, narrowed to a ticker list when given."""
    if not storage.table_exists(con, 'factor_panel'):
        return set()
    clauses, params = ['Vintage = ?'], [vintage]
    if tickers is not None:
        clauses.append('Ticker IN (SELECT unnest(?::VARCHAR[]))')
        params.append(sorted(tickers))
    rows = con.execute(f'SELECT DISTINCT Date FROM factor_panel WHERE {" AND ".join(clauses)}',
                       params).fetchall()
    return {pd.Timestamp(row[0]) for row in rows}


def _build_estimate(build: dict | None, rows: int) -> float | None:
    """Seconds this build should take, from the rate the last one managed."""
    if build is None or not build.get('seconds') or not build.get('n_rows'):
        return None
    return rows * (build['seconds'] / build['n_rows'])
