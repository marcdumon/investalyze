"""Data access for the backtest page: stored runs, their curves, and their blotters.

Each read opens a short-lived read-only connection, matching the other pages. Nothing is cached: a
run finishing while the page is open should show up on the next refresh rather than after a restart.
"""

from pathlib import Path

import duckdb
import pandas as pd

from investalyze.analysis import custom_factors, factors_pit, panels
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
                   strftime(CreatedAt, '%Y-%m-%d %H:%M') AS Created
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


def get_evaluated_universes(version: str | None = None) -> list[str]:
    """The distinct universes an evaluation has been stored for, so one can be chosen to read.

    Distinct because a scope is a version and a universe together: evaluating the same universe on
    both versions gives two scopes and one name, and a picker offered the same name twice is a
    picker that cannot say which it means.
    """
    seen = []
    for scope in get_evaluation_scopes(version):
        if scope['universe'] not in seen:
            seen.append(scope['universe'])
    return seen


def get_evaluation_scopes(version: str | None = None) -> list[dict]:
    """Every stored evaluation, one row per version and universe, with what it holds.

    The counts are what a delete would remove, so the same read answers both the picker and the
    confirmation it leads to.
    """
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_ic') or not _has_column(con, 'factor_ic', 'Universe'):
            return []
        clause, params = ('WHERE Version = ?', [version]) if version else ('', [])
        rows = con.execute(f"""
            SELECT ic.Version, ic.Universe, count(DISTINCT ic.Factor) AS factors, count(*) AS ic_rows,
                   min(ic.Date)::VARCHAR AS first_date, max(ic.Date)::VARCHAR AS last_date
            FROM factor_ic ic {clause}
            GROUP BY ic.Version, ic.Universe ORDER BY ic.Version, ic.Universe
        """, params).df()
        quantiles = dict(con.execute(
            'SELECT Version || \'|\' || Universe, count(*) FROM factor_quantiles GROUP BY 1').fetchall())
    finally:
        con.close()

    scopes = []
    for row in rows.itertuples():
        scopes.append({'version': row.Version, 'universe': row.Universe, 'factors': int(row.factors),
                       'ic_rows': int(row.ic_rows), 'first': row.first_date, 'last': row.last_date,
                       'quantile_rows': int(quantiles.get(f'{row.Version}|{row.Universe}', 0))})
    return scopes


def get_factor_ic(version: str = 'as_reported', universe: str | None = None) -> pd.DataFrame:
    """Mean information coefficient and its t-statistic per factor and horizon, for one universe."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_ic'):
            return pd.DataFrame()
        clauses = ['Version = ?', 'Ic IS NOT NULL']
        params: list = [version]
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


def get_quantile_profile(factor: str, version: str = 'as_reported',
                         universe: str | None = None) -> pd.DataFrame:
    """Forward return of each score bucket for one factor, as a mean and as a median.

    Both are returned because a bucket's mean answers to a few extreme names while its median does
    not, and the two part company on a universe that was never filtered for price or liquidity.
    """
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_quantiles'):
            return pd.DataFrame()
        clauses = ['Version = ?', 'Factor = ?']
        params: list = [version, factor]
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


def get_panel_coverage(version: str = 'as_reported') -> tuple[dict | None, dict | None]:
    """What the stored factor panel holds for a version, and the request behind its last build.

    Coverage is read from the rows, so it is true whatever produced them; the build record adds the
    universe and the time, which the rows do not carry.
    """
    con = _connect()
    try:
        coverage = factors_pit.panel_coverage(con, version)
        build = None
        if storage.table_exists(con, 'factor_panel_builds'):
            build = writer.latest_panel_build(con, version)
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


def panel_gaps(version: str, universe: str | None, start: str, end: str, panel_freq: str,
               rebalance: str | None = None) -> dict:
    """Everything about the stored panel that stands between a request and a result.

    Dates are counted for the requested names rather than for the panel as a whole: a date built for
    other tickers is no use to these ones. Factor definitions are compared against the fingerprint
    the last build recorded, since a redefined factor leaves the dates in place and the values wrong.

    `empty_factors` is the separate case of a factor with no value in the rows that do exist for
    these names, which building cannot change: a company that has never paid a dividend has no
    dividend growth to compute.

    `panel_freq` is the cadence a build would use, which is the card's own field rather than
    anything read off what is stored. `rebalance` is the separate question of how often a run
    trades, and is only carried through so the caller can say what a finer one would trade on.

    The estimate comes from how long the last build took per row. Without a recorded build there is
    nothing to estimate from, and None says so rather than a number pulled from the air.
    """
    tickers = universes.load_universe(universe) if universe else None
    con = _connect()
    try:
        coverage = factors_pit.panel_coverage(con, version)
        grid = factors_pit.buildable_dates(
            con, factors_pit.rebalance_dates(con, start=start, end=end, freq=panel_freq), tickers)
        build = writer.latest_panel_build(con, version) \
            if storage.table_exists(con, 'factor_panel_builds') else None
        held = factors_pit.stored_dates(con, version, tickers)
        empty = factors_pit.empty_factors(con, version, tickers)
        n_tickers = len(tickers) if tickers is not None else _priced_tickers(con)
    finally:
        con.close()

    missing = [date for date in grid if date not in held]
    return {
        'requested_dates': len(grid),
        'missing_dates': len(missing),
        'n_tickers': n_tickers,
        'empty_factors': empty,
        'changed_factors': custom_factors.changed_since(build['definitions'] if build else None, CUSTOM),
        'seconds_estimate': _build_estimate(build, len(missing) * n_tickers),
        'has_panel': coverage is not None,
        'build_frequency': panel_freq,
        'panel_frequency': coverage['frequency'] if coverage else None,
        'rebalance': rebalance,
    }


def _priced_tickers(con: duckdb.DuckDBPyConnection) -> int | None:
    """How many names a build over the whole database would cover.

    Counted from the prices rather than from the stored panel: a version with no panel yet would
    otherwise report no names, when what it means is that nothing has been built for it.
    """
    if not storage.table_exists(con, 'prices'):
        return None
    return con.execute('SELECT count(DISTINCT Ticker) FROM prices').fetchone()[0]


def _build_estimate(build: dict | None, rows: int) -> float | None:
    """Seconds this build should take, from the rate the last one managed."""
    if build is None or not build.get('seconds') or not build.get('n_rows'):
        return None
    return rows * (build['seconds'] / build['n_rows'])


def tradable_names(universe: str | None, start: str, end: str, n_top: int,
                   rank_by: str) -> tuple[int, int | None]:
    """How many names a run would actually trade, and how many the universe holds.

    A name counts if it traded at any point in the range, since one that listed later cannot be
    bought before it exists and one that stopped is liquidated at its last price. What is reported
    here is therefore how many of a saved universe have any history in the window at all.
    """
    candidates = universes.load_universe(universe) if universe else None
    con = _connect()
    try:
        calendar = panels.trading_calendar(con, start=start, end=end)
        kept = panels.traded_universe(con, calendar, n_top=n_top, rank_by=rank_by,
                                      require_sector=True, candidates=candidates)
    finally:
        con.close()
    return len(kept), len(candidates) if candidates is not None else None
