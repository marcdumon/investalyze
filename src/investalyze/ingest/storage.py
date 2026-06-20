"""Shared plumbing every provider + the orchestrator use.

Three jobs, deliberately small:
  1. paths   — the data-dir contract (provider-first layout, dirs created by code)
  2. db      — DuckDB connection (one place, one policy)
  3. write   — the single helper providers save through (DRY DB writes)

Split this module only if one of these grows enough to earn its own file.
"""
from pathlib import Path

import duckdb
import pandas as pd

# --- paths --------------------------------------------------------------------
# Provider-first: each provider owns data/<provider>/{raw,processed,state};
# the shared DB lives at the data root. Dirs are created by code, idempotently.

SUBDIRS: tuple[str, ...] = ('raw', 'processed', 'state')
DB_FILENAME = 'investalyze.duckdb'


def _db_path(data_root: Path, db: str = DB_FILENAME) -> Path:
    """Path to the shared DuckDB file at the data root."""
    return data_root / db


def setup_provider(data_root: Path, provider: str) -> dict[str, Path]:
    """Create data/<provider>/{raw,processed,state} if missing. Idempotent.

    Returns the subdir paths keyed by name. A provider's first action; it
    touches nothing outside its own subtree.
    """
    root = data_root / provider
    paths = {name: root / name for name in SUBDIRS}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# --- db -----------------------------------------------------------------------
def connect(data_root: Path, db: str = DB_FILENAME) -> duckdb.DuckDBPyConnection:
    """Open a writable connection to the shared DB at the data root."""
    return duckdb.connect(str(_db_path(data_root, db)))


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
