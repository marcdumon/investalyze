"""Shared plumbing every provider + the orchestrator use.

Pure: takes plain paths/values, knows nothing about config (the orchestrator
owns that). A few small jobs, deliberately so:
  1. connect       — DuckDB connection (one place, one policy)
  2. introspection — table_exists / count_rows
  3. store         — the single helper providers store through (DRY DB writes)

Split this module only if one of these grows enough to earn its own file.
"""

from pathlib import Path

import duckdb
import pandas as pd

_DB = 'investalyze.duckdb'


# --- db -----------------------------------------------------------------------
def connect(data_root: Path, db: str = _DB, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the shared DB at the data root (writable by default).

    `read_only=True` for consumers (e.g. notebooks) that only query.
    """
    return duckdb.connect(str(data_root / db), read_only=read_only)


# --- introspection ------------------------------------------------------------
def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """True if `table` exists in the connected DB."""
    return con.execute('SELECT 1 FROM information_schema.tables WHERE table_name = ?', [table]).fetchone() is not None


def count_rows(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Row count of `table`, or 0 if it doesn't exist."""
    if not table_exists(con, table):
        return 0
    return int(con.execute(f'SELECT count(*) FROM {table}').fetchone()[0])  # type: ignore[index]


# --- store --------------------------------------------------------------------
def store(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key: list[str]) -> int:
    """Merge-upsert `df` into `table` on `key`. The one path output reaches the DB.

    Creates the table from the frame's schema on first write, then upserts:
    matched keys update their non-key columns, new keys insert. Idempotent —
    re-ingesting full history or a daily update both converge. Providers
    decide WHAT/WHEN to store; this owns HOW. Returns the table's row count.
    """
    cols = list(df.columns)
    non_key = [c for c in cols if c not in key]
    con.register('_incoming', df)
    try:
        con.execute(f'CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming WHERE FALSE')
        on = ' AND '.join(f't."{c}" = s."{c}"' for c in key)
        updates = ', '.join(f'"{c}" = s."{c}"' for c in non_key)
        insert_cols = ', '.join(f'"{c}"' for c in cols)
        insert_vals = ', '.join(f's."{c}"' for c in cols)
        distinct_on = ', '.join(f'"{c}"' for c in key)
        con.execute(f"""
            MERGE INTO {table} t
            USING (SELECT DISTINCT ON ({distinct_on}) * FROM _incoming) s
            ON {on}
            WHEN MATCHED THEN UPDATE SET {updates}
            WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """)
    finally:
        con.unregister('_incoming')
    count: int = con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]  # type: ignore[index]
    return count
