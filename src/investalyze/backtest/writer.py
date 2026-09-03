"""The one path backtest and evaluation results reach the DB.

Each run owns its rows wholesale, so writing replaces them by `RunId` (delete-then-insert), keeping
re-runs idempotent even when a re-run covers a shorter date range than the one before it. Evaluation
rows work the same way, keyed by vintage and factor. `factor_panel` is the exception: it is
update-shaped, so `ingest.storage.store` merge-upserts it.
"""

import hashlib
import json

import duckdb
import pandas as pd

RUN_SCHEMA_VERSION = 1  # bump when engine semantics change, so old runs do not collide with new

RUN_COLS = ['RunId', 'Signal', 'Kind', 'ConfigJson', 'StartDate', 'EndDate', 'NTickers', 'Benchmark',
            'SurvivorshipNote', 'Cagr', 'Vol', 'Sharpe', 'Sortino', 'MaxDrawdown', 'Calmar',
            'Turnover', 'HitRate', 'NTrades', 'FinalEquity', 'TotalCosts']
EQUITY_COLS = ['RunId', 'Date', 'Equity', 'Gross', 'Net', 'Cash', 'BenchmarkEquity']
TRADE_COLS = ['RunId', 'Date', 'Ticker', 'Side', 'Shares', 'Price', 'Notional', 'Cost', 'Kind']
IC_COLS = ['Vintage', 'Factor', 'Date', 'Horizon', 'Ic', 'NNames']
QUANTILE_COLS = ['Vintage', 'Factor', 'Date', 'Horizon', 'Quantile', 'Return', 'NNames']

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        RunId            VARCHAR NOT NULL,
        Signal           VARCHAR NOT NULL,
        Kind             VARCHAR NOT NULL,
        ConfigJson       VARCHAR NOT NULL,
        StartDate        DATE NOT NULL,
        EndDate          DATE NOT NULL,
        NTickers         BIGINT NOT NULL,
        Benchmark        VARCHAR,
        SurvivorshipNote VARCHAR NOT NULL,
        Cagr             DOUBLE,
        Vol              DOUBLE,
        Sharpe           DOUBLE,
        Sortino          DOUBLE,
        MaxDrawdown      DOUBLE,
        Calmar           DOUBLE,
        Turnover         DOUBLE,
        HitRate          DOUBLE,
        NTrades          BIGINT,
        FinalEquity      DOUBLE,
        TotalCosts       DOUBLE,
        CreatedAt        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_equity (
        RunId           VARCHAR NOT NULL,
        Date            DATE NOT NULL,
        Equity          DOUBLE NOT NULL,
        Gross           DOUBLE,
        Net             DOUBLE,
        Cash            DOUBLE,
        BenchmarkEquity DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_trades (
        RunId    VARCHAR NOT NULL,
        Date     DATE NOT NULL,
        Ticker   VARCHAR NOT NULL,
        Side     VARCHAR NOT NULL,
        Shares   DOUBLE NOT NULL,
        Price    DOUBLE NOT NULL,
        Notional DOUBLE NOT NULL,
        Cost     DOUBLE NOT NULL,
        Kind     VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_ic (
        Vintage VARCHAR NOT NULL,
        Factor  VARCHAR NOT NULL,
        Date    DATE NOT NULL,
        Horizon BIGINT NOT NULL,
        Ic      DOUBLE,
        NNames  BIGINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_quantiles (
        Vintage  VARCHAR NOT NULL,
        Factor   VARCHAR NOT NULL,
        Date     DATE NOT NULL,
        Horizon  BIGINT NOT NULL,
        Quantile BIGINT NOT NULL,
        Return   DOUBLE,
        NNames   BIGINT NOT NULL
    )
    """,
]


def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create every backtest output table that does not exist yet."""
    for statement in _DDL:
        con.execute(statement)


def run_id(cfg: dict) -> str:
    """Stable 12-hex identifier for a run, derived from its resolved config and the schema version."""
    payload = json.dumps({'schema': RUN_SCHEMA_VERSION, **cfg}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _replace(con: duckdb.DuckDBPyConnection, table: str, columns: list[str], rows: pd.DataFrame,
             where: str, params: list) -> int:
    """Delete the rows `where` selects, insert `rows` in their place, and return how many landed."""
    if list(rows.columns) != columns:
        raise ValueError(f'{table} columns must be {columns}, got {list(rows.columns)}')
    con.execute(f'DELETE FROM {table} WHERE {where}', params)
    con.register('_rows', rows)
    try:
        con.execute(f'INSERT INTO {table} SELECT * FROM _rows')
    finally:
        con.unregister('_rows')
    return len(rows)


def replace_run(con: duckdb.DuckDBPyConnection, identifier: str, summary: pd.DataFrame,
                equity: pd.DataFrame, trades: pd.DataFrame) -> int:
    """Replace one run's summary, equity curve and blotter, returning the number of equity rows."""
    if list(summary.columns) != RUN_COLS:
        raise ValueError(f'summary columns must be {RUN_COLS}, got {list(summary.columns)}')
    con.execute('DELETE FROM backtest_runs WHERE RunId = ?', [identifier])
    con.register('_summary', summary)
    try:
        con.execute('INSERT INTO backtest_runs SELECT *, now() FROM _summary')
    finally:
        con.unregister('_summary')
    _replace(con, 'backtest_trades', TRADE_COLS, trades, 'RunId = ?', [identifier])
    return _replace(con, 'backtest_equity', EQUITY_COLS, equity, 'RunId = ?', [identifier])


def replace_evaluation(con: duckdb.DuckDBPyConnection, vintage: str, factor: str,
                       ic: pd.DataFrame, quantiles: pd.DataFrame) -> int:
    """Replace one factor's IC and quantile rows for a vintage, returning the number of IC rows."""
    _replace(con, 'factor_quantiles', QUANTILE_COLS, quantiles,
             'Vintage = ? AND Factor = ?', [vintage, factor])
    return _replace(con, 'factor_ic', IC_COLS, ic, 'Vintage = ? AND Factor = ?', [vintage, factor])
