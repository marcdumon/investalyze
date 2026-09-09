"""The one path backtest and evaluation results reach the DB.

Each run owns its rows wholesale, so writing replaces them by `RunId` (delete-then-insert), keeping
re-runs idempotent even when a re-run covers a shorter date range than the one before it. Evaluation
rows work the same way, keyed by version, universe, factor, horizon and period, so evaluating the same
factor over a different universe, horizon or date range sits alongside the earlier reading rather than
replacing it, while evaluating the same one again is what replaces it. `factor_panel` is the exception:
it is update-shaped, so `ingest.storage.store` merge-upserts it.
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

IC_COLS = ['Version', 'Universe', 'Factor', 'Date', 'Horizon', 'Ic', 'NNames']
QUANTILE_COLS = ['Version', 'Universe', 'Factor', 'Date', 'Horizon', 'Quantile', 'Return',
                 'MedianReturn', 'NNames']
WHOLE_DATABASE = 'all'          # the Universe label for an evaluation that named no universe
UNRECORDED_UNIVERSE = 'unrecorded'   # rows written before the Universe column existed

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
    """
    CREATE TABLE IF NOT EXISTS factor_panel_builds (
        Version   VARCHAR NOT NULL,
        Universe  VARCHAR NOT NULL,
        StartDate DATE NOT NULL,
        EndDate   DATE NOT NULL,
        Frequency VARCHAR NOT NULL,
        NTickers  BIGINT NOT NULL,
        NDates    BIGINT NOT NULL,
        NRows       BIGINT NOT NULL,
        Seconds     DOUBLE,
        Definitions VARCHAR,
        BuiltAt     TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_ic (
        Version  VARCHAR NOT NULL,
        Universe VARCHAR NOT NULL,
        Factor   VARCHAR NOT NULL,
        Date     DATE NOT NULL,
        Horizon  BIGINT NOT NULL,
        Ic       DOUBLE,
        NNames   BIGINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_quantiles (
        Version      VARCHAR NOT NULL,
        Universe     VARCHAR NOT NULL,
        Factor       VARCHAR NOT NULL,
        Date         DATE NOT NULL,
        Horizon      BIGINT NOT NULL,
        Quantile     BIGINT NOT NULL,
        Return       DOUBLE,
        MedianReturn DOUBLE,
        NNames       BIGINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS factor_ic_decay (
        Version   VARCHAR NOT NULL,
        Universe  VARCHAR NOT NULL,
        Factor    VARCHAR NOT NULL,
        StartDate DATE NOT NULL,
        EndDate   DATE NOT NULL,
        Horizon   BIGINT NOT NULL,
        MeanIc    DOUBLE,
        TStat     DOUBLE,
        NPeriods  BIGINT NOT NULL
    )
    """,
]

DECAY_HORIZONS = (5, 21, 63, 126)  # the rebalance-relevant horizons the backlog's own analysis used
DECAY_COLS = ['Version', 'Universe', 'Factor', 'StartDate', 'EndDate', 'Horizon', 'MeanIc', 'TStat', 'NPeriods']


# columns added after the tables were first written. Rows already there are labelled
# unrecorded rather than all: what they were measured over was never stored.
_ADDED_COLUMNS = {
    'factor_ic': [('Universe', f"VARCHAR DEFAULT '{UNRECORDED_UNIVERSE}'"),
                 ('StartDate', 'DATE'), ('EndDate', 'DATE')],
    'factor_quantiles': [('Universe', f"VARCHAR DEFAULT '{UNRECORDED_UNIVERSE}'"),
                         ('MedianReturn', 'DOUBLE'), ('StartDate', 'DATE'), ('EndDate', 'DATE')],
    'factor_panel_builds': [('Seconds', 'DOUBLE'), ('Definitions', 'VARCHAR')],
}

# tables whose StartDate/EndDate a row written before those columns existed can recover from
# its own Date column: the period a reading covered was always implied by which dates it has
_PERIOD_BACKFILL_TABLES = ('factor_ic', 'factor_quantiles')

# columns a table used to carry that no longer belong: an older database still has them
_DROPPED_COLUMNS = {
    'backtest_runs': ['SurvivorshipNote'],
}

# a column kept but renamed: 'vintage' read as unexplained finance jargon. factor_panel is
# included even though it has no fixed DDL (its schema comes from whatever DataFrame first
# wrote it): it is otherwise touched only by a build, so a command that only reads it, like
# evaluate, would never trigger the rename and would fail looking for the new column name.
_RENAMED_COLUMNS = {
    'factor_panel': [('Vintage', 'Version')],
    'factor_ic': [('Vintage', 'Version')],
    'factor_quantiles': [('Vintage', 'Version')],
    'factor_panel_builds': [('Vintage', 'Version')],
}


def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create every backtest output table that does not exist yet, and bring the older ones in line."""
    for statement in _DDL:
        con.execute(statement)
    for table, renames in _RENAMED_COLUMNS.items():
        try:
            existing = {row[0] for row in con.execute(f'DESCRIBE {table}').fetchall()}
        except duckdb.CatalogException:
            continue   # factor_panel has no fixed DDL; a fresh DB may not have it yet
        for old, new in renames:
            if old in existing and new not in existing:
                con.execute(f'ALTER TABLE {table} RENAME COLUMN {old} TO {new}')
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[0] for row in con.execute(f'DESCRIBE {table}').fetchall()}
        for column, definition in columns:
            if column not in existing:
                con.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
    for table, columns in _DROPPED_COLUMNS.items():
        existing = {row[0] for row in con.execute(f'DESCRIBE {table}').fetchall()}
        for column in columns:
            if column in existing:
                con.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
    for table in _PERIOD_BACKFILL_TABLES:
        con.execute(f"""
            UPDATE {table} t SET StartDate = g.min_date, EndDate = g.max_date
            FROM (SELECT Version, Universe, Factor, Horizon, min(Date) AS min_date, max(Date) AS max_date
                  FROM {table} GROUP BY Version, Universe, Factor, Horizon) g
            WHERE t.Version = g.Version AND t.Universe = g.Universe AND t.Factor = g.Factor
              AND t.Horizon = g.Horizon AND t.StartDate IS NULL
        """)


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


def record_panel_build(con: duckdb.DuckDBPyConnection, *, version: str, universe: str,
                       start: str, end: str, frequency: str, n_tickers: int, n_dates: int,
                       n_rows: int, seconds: float | None = None,
                       definitions: str | None = None) -> None:
    """Append what a factor panel build was asked for, which the panel rows themselves do not say.

    The rows record what the panel holds; this records the request behind them, so a page can say
    which universe a panel came from and when it was last built. `seconds` is what the next build
    is estimated from, and `definitions` fingerprints the factor formulas used, which is the only
    way to notice that a redefined factor left the panel holding values computed from the old one.
    """
    con.execute("""
        INSERT INTO factor_panel_builds (Version, Universe, StartDate, EndDate, Frequency,
                                         NTickers, NDates, NRows, Seconds, Definitions, BuiltAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
    """, [version, universe, start, end, frequency, n_tickers, n_dates, n_rows, seconds, definitions])


def latest_panel_build(con: duckdb.DuckDBPyConnection, version: str) -> dict | None:
    """The most recent build recorded for a version, or None when none was."""
    cursor = con.execute('SELECT * FROM factor_panel_builds WHERE Version = ? ORDER BY BuiltAt DESC LIMIT 1',
                         [version])
    row = cursor.fetchone()
    if row is None:
        return None
    # read by name: a record written before a column was added simply has None for it
    stored = dict(zip([column[0] for column in cursor.description], row))
    aliases = {'Universe': 'universe', 'StartDate': 'start', 'EndDate': 'end',
               'Frequency': 'frequency', 'NTickers': 'n_tickers', 'NDates': 'n_dates',
               'NRows': 'n_rows', 'Seconds': 'seconds', 'Definitions': 'definitions',
               'BuiltAt': 'built_at'}
    return {alias: stored.get(column) for column, alias in aliases.items()}


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


EVALUATION_TABLES = ('factor_ic', 'factor_quantiles')


def delete_evaluation(con: duckdb.DuckDBPyConnection, version: str, universe: str) -> dict[str, int]:
    """Delete one version and universe's evaluation, returning how many rows went per table.

    Re-running an evaluation replaces the factors it covers; this is how a scope stops existing,
    which is what a universe that was renamed or never recorded needs.
    """
    removed = {}
    for table in EVALUATION_TABLES:
        removed[table] = con.execute(
            f'SELECT count(*) FROM {table} WHERE Version = ? AND Universe = ?',
            [version, universe]).fetchone()[0]
        con.execute(f'DELETE FROM {table} WHERE Version = ? AND Universe = ?', [version, universe])
    return removed


_STORED_IC_COLS = IC_COLS + ['StartDate', 'EndDate']
_STORED_QUANTILE_COLS = QUANTILE_COLS + ['StartDate', 'EndDate']


def replace_evaluation(con: duckdb.DuckDBPyConnection, version: str, universe: str, factor: str,
                       ic: pd.DataFrame, quantiles: pd.DataFrame, start, end) -> int:
    """Replace one factor's IC and quantile rows for a version, universe, horizon and period.

    `start` and `end` are what the whole evaluate run covered, not derived from this factor's own
    rows: a factor needing warm-up data reads over a narrower span than the rest of the run, and
    stamping from its own rows would fragment one run into several near-duplicate periods instead of
    the one it actually was.
    """
    horizon = int(ic['Horizon'].iloc[0])
    stamped_ic = ic.assign(StartDate=start, EndDate=end)[_STORED_IC_COLS]
    stamped_quantiles = quantiles.assign(StartDate=start, EndDate=end)[_STORED_QUANTILE_COLS]
    where = 'Version = ? AND Universe = ? AND Factor = ? AND Horizon = ? AND StartDate = ? AND EndDate = ?'
    params = [version, universe, factor, horizon, start, end]
    _replace(con, 'factor_quantiles', _STORED_QUANTILE_COLS, stamped_quantiles, where, params)
    return _replace(con, 'factor_ic', _STORED_IC_COLS, stamped_ic, where, params)


def replace_ic_decay(con: duckdb.DuckDBPyConnection, version: str, universe: str, factor: str,
                     start, end, decay: pd.DataFrame) -> int:
    """Replace one factor's IC-decay row set for a version, universe and period.

    `decay` is `evaluation.ic_decay`'s own frame (`horizon, mean_ic, t_stat, n_periods`), one row per
    horizon it was asked for; the period is not read off `decay` itself, since it carries no dates, so
    the caller passes the same start and end the headline evaluation for this run was stamped with.
    """
    stamped = decay.rename(columns={'horizon': 'Horizon', 'mean_ic': 'MeanIc', 't_stat': 'TStat',
                                    'n_periods': 'NPeriods'}).assign(
        Version=version, Universe=universe, Factor=factor, StartDate=start, EndDate=end)[DECAY_COLS]
    where = 'Version = ? AND Universe = ? AND Factor = ? AND StartDate = ? AND EndDate = ?'
    return _replace(con, 'factor_ic_decay', DECAY_COLS, stamped, where, [version, universe, factor, start, end])
