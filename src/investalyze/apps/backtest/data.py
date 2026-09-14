"""Data access for the backtest page: stored runs, their curves, and their blotters.

Each read opens a short-lived read-only connection, matching the other pages. Nothing is cached: a
run finishing while the page is open should show up on the next refresh rather than after a restart.
"""

import pandas as pd

from investalyze.apps.research import data as research_data
from investalyze.apps.research import logic as research_logic
from investalyze.ingest import storage


def get_runs() -> pd.DataFrame:
    """Every stored run, newest first, with the headline statistics the table shows."""
    with storage.connect(read_only=True) as con:
        if not storage.table_exists(con, 'backtest_runs'):
            return pd.DataFrame()
        return con.execute("""
            SELECT RunId, Signal, Kind, StartDate::VARCHAR AS StartDate, EndDate::VARCHAR AS EndDate,
                   NTickers, Benchmark, ConfigJson, Cagr, Vol, Sharpe,
                   Sortino, MaxDrawdown, Calmar, Turnover, HitRate, NTrades, FinalEquity, TotalCosts,
                   strftime(CreatedAt, '%Y-%m-%d %H:%M') AS Created
            FROM backtest_runs ORDER BY CreatedAt DESC
        """).df()


def get_equity(run_id: str) -> pd.DataFrame:
    """One run's equity curve, with the benchmark alongside it where the run recorded one."""
    with storage.connect(read_only=True) as con:
        if not storage.table_exists(con, 'backtest_equity'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Date, Equity, Gross, Net, Cash, BenchmarkEquity
            FROM backtest_equity WHERE RunId = ? ORDER BY Date
        """, [run_id]).df()


def get_trades(run_id: str, limit: int = 500) -> pd.DataFrame:
    """The most recent fills of one run."""
    with storage.connect(read_only=True) as con:
        if not storage.table_exists(con, 'backtest_trades'):
            return pd.DataFrame()
        return con.execute("""
            SELECT Date::VARCHAR AS Date, Ticker, Side, Shares, Price, Notional, Cost, Kind
            FROM backtest_trades WHERE RunId = ? ORDER BY Date DESC, Ticker LIMIT ?
        """, [run_id, limit]).df()


def factor_rows() -> list[dict]:
    """The factor library as grid rows, with each factor's latest IC analysis and backtest results."""
    return research_logic.library_rows(research_data.library(), research_data.summaries())
