"""Shared plumbing every provider + the orchestrator use.

Pure: takes plain paths/values, knows nothing about config (the orchestrator
owns that). Two jobs, deliberately small:
  1. db      — DuckDB connection (one place, one policy)
  2. write   — the single helper providers save through (DRY DB writes)

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


# --- write --------------------------------------------------------------------
def write(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key: list[str]) -> int:
    """Merge-upsert `df` into `table` on `key`. The one path output reaches the DB.

    Creates the table from the frame's schema on first write, then upserts:
    matched keys update their non-key columns, new keys insert. Idempotent —
    re-ingesting full history or a daily update both converge. Providers
    decide WHAT/WHEN to save; this owns HOW. Returns the table's row count.
    """
    cols = list(df.columns)
    non_key = [c for c in cols if c not in key]
    con.register('_incoming', df)
    try:
        con.execute(f'CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming WHERE FALSE')
        on = ' AND '.join(f't.{c} = s.{c}' for c in key)
        updates = ', '.join(f'{c} = s.{c}' for c in non_key)
        insert_cols = ', '.join(cols)
        insert_vals = ', '.join(f's.{c}' for c in cols)
        con.execute(f"""
            MERGE INTO {table} t
            USING (SELECT DISTINCT ON ({', '.join(key)}) * FROM _incoming) s
            ON {on}
            WHEN MATCHED THEN UPDATE SET {updates}
            WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """)
    finally:
        con.unregister('_incoming')
    count: int = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]  # type: ignore[index]
    return count
