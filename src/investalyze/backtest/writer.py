"""The one path backtest runs reach the DB.

Each run owns its rows wholesale, so writing replaces them by `RunId` (delete-then-insert), keeping
re-runs idempotent even when a re-run covers a shorter date range than the one before it.
"""

import hashlib
import json

import duckdb
import pandas as pd

RUN_SCHEMA_VERSION = 1  # bump when engine semantics change, so old runs do not collide with new

RUN_COLS = ['RunId', 'Signal', 'Kind', 'ConfigJson', 'StartDate', 'EndDate', 'NTickers', 'Benchmark',
            'Cagr', 'Vol', 'Sharpe', 'Sortino', 'MaxDrawdown', 'Calmar',
            'Turnover', 'HitRate', 'NTrades', 'FinalEquity', 'TotalCosts']
EQUITY_COLS = ['RunId', 'Date', 'Equity', 'Gross', 'Net', 'Cash', 'BenchmarkEquity']
TRADE_COLS = ['RunId', 'Date', 'Ticker', 'Side', 'Shares', 'Price', 'Notional', 'Cost', 'Kind']
RUN_TABLES = ('backtest_runs', 'backtest_equity', 'backtest_trades')

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
]

# columns a table used to carry that no longer belong: an older database still has them
_DROPPED_COLUMNS = {
    'backtest_runs': ['SurvivorshipNote'],
}


def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create every backtest run table that does not exist yet, and drop columns that no longer belong."""
    for statement in _DDL:
        con.execute(statement)
    for table, columns in _DROPPED_COLUMNS.items():
        existing = {row[0] for row in con.execute(f'DESCRIBE {table}').fetchall()}
        for column in columns:
            if column in existing:
                con.execute(f'ALTER TABLE {table} DROP COLUMN {column}')


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
    named = ', '.join(f'"{column}"' for column in columns)
    try:
        con.execute(f'INSERT INTO {table} ({named}) SELECT {named} FROM _rows')
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
    named = ', '.join(f'"{column}"' for column in RUN_COLS)
    try:
        con.execute(f'INSERT INTO backtest_runs ({named}, CreatedAt) SELECT {named}, now() FROM _summary')
    finally:
        con.unregister('_summary')
    _replace(con, 'backtest_trades', TRADE_COLS, trades, 'RunId = ?', [identifier])
    return _replace(con, 'backtest_equity', EQUITY_COLS, equity, 'RunId = ?', [identifier])


def delete_run(con: duckdb.DuckDBPyConnection, identifier: str) -> dict[str, int]:
    """Delete one run everywhere it was stored, returning how many rows went per table.

    Counting before the delete lets the caller tell a run that was removed from a run id that was
    never stored, and clears rows a half-written run left behind in the curve or blotter tables.
    """
    removed = {}
    for table in RUN_TABLES:
        removed[table] = con.execute(f'SELECT count(*) FROM {table} WHERE RunId = ?', [identifier]).fetchone()[0]
        con.execute(f'DELETE FROM {table} WHERE RunId = ?', [identifier])
    return removed
