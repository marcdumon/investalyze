"""Data access for the backtest page: stored runs, their curves, and their blotters.

Each read opens a short-lived read-only connection, matching the other pages. Nothing is cached: a
run finishing while the page is open should show up on the next refresh rather than after a restart.
"""

from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'

RUN_TABLES = ('backtest_runs', 'backtest_equity', 'backtest_trades')


def _connect() -> duckdb.DuckDBPyConnection:
    """A read-only connection to the project database."""
    return storage.connect(DATA_ROOT, read_only=True)


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
                   SurvivorshipNote, CreatedAt
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


def get_factor_ic(vintage: str = 'as_reported') -> pd.DataFrame:
    """Mean information coefficient and its t-statistic per factor and horizon."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_ic'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Factor, Horizon, count(*) AS Periods, avg(Ic) AS MeanIc, stddev_samp(Ic) AS StdIc,
                   avg(CASE WHEN Ic > 0 THEN 1.0 ELSE 0.0 END) AS HitRate, avg(NNames) AS MeanNames
            FROM factor_ic WHERE Vintage = ? AND Ic IS NOT NULL
            GROUP BY Factor, Horizon ORDER BY Factor, Horizon
        """, [vintage]).df()
    finally:
        con.close()


def get_quantile_profile(factor: str, vintage: str = 'as_reported') -> pd.DataFrame:
    """Mean forward return of each score bucket for one factor."""
    con = _connect()
    try:
        if not storage.table_exists(con, 'factor_quantiles'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Quantile, Horizon, avg(Return) AS MeanReturn, count(*) AS Periods
            FROM factor_quantiles WHERE Vintage = ? AND Factor = ?
            GROUP BY Quantile, Horizon ORDER BY Horizon, Quantile
        """, [vintage, factor]).df()
    finally:
        con.close()
