"""One-off migration: rename metadata-table columns to canonical PascalCase.

Renames the columns the rename maps cover, in place, on the live DuckDB — no refetch.
Idempotent: each rename runs only when the old column is present and the new one is not, so
re-running (or running against an already-migrated or empty DB) is a safe no-op. Imports the
provider rename maps directly so it can never drift from the write-time renames.

Run once:  uv run python -m scripts.migrate_pascalcase_columns
"""
import logging
from pathlib import Path

import duckdb

from investalyze.ingest import config, storage
from investalyze.ingest.providers.simfin import columns as simfin_cols
from investalyze.ingest.providers.yahoo import columns as yahoo_cols

log = logging.getLogger('investalyze.migrate')

# table -> {old column name: new column name}
_RENAMES: dict[str, dict[str, str]] = {
    'company_profile': yahoo_cols.COMPANY_PROFILE,
    'company_officers': yahoo_cols.COMPANY_OFFICERS,
    'companies': simfin_cols.COMPANIES,
}


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Current column names of `table` (empty set if the table does not exist)."""
    rows = con.execute(
        'SELECT column_name FROM information_schema.columns WHERE table_name = ?', [table]
    ).fetchall()
    return {r[0] for r in rows}


def migrate(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Rename old metadata columns to canonical PascalCase. Returns the renames performed.

    Idempotent: skips a table that doesn't exist and a column already renamed/absent.
    """
    done: list[str] = []
    for table, mapping in _RENAMES.items():
        cols = _columns(con, table)
        if not cols:
            continue
        for old, new in mapping.items():
            if old in cols and new not in cols:
                con.execute(f'ALTER TABLE {table} RENAME COLUMN "{old}" TO "{new}"')
                done.append(f'{table}.{old} -> {new}')
                log.info(f'renamed {table}.{old} -> {new}')
    return done


def main() -> None:
    """Resolve the DB from ingest.toml and migrate it in place."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(name)s: %(message)s')
    cfg = config.load(Path('ingest.toml'))
    con = storage.connect(cfg.data_root, cfg.db)
    try:
        done = migrate(con)
        log.info(f'migration complete — {len(done)} column(s) renamed')
    finally:
        con.close()


if __name__ == '__main__':
    main()
