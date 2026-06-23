"""One-off migration: rename the two raw company tables, then build the combined `companies` table.

Renames `company_profile` -> `_yahoo_companies` and `companies` -> `_simfin_companies` in place, then
calls the housekeeping `rebuild_companies` to (re)create the combined `companies` table. Idempotent:
a table already renamed is skipped; the rebuild is `CREATE OR REPLACE`, so re-running is safe.

Run once:  uv run python -m scripts.migrate_company_tables
"""
import logging
from pathlib import Path

import duckdb

from investalyze.ingest import config, storage
from investalyze.ingest.housekeeping import rebuild_companies

log = logging.getLogger('investalyze.migrate')

# old DB table name -> new name
_RENAMES: dict[str, str] = {
    'company_profile': '_yahoo_companies',
    'companies': '_simfin_companies',
}


def migrate(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Rename the raw company tables then rebuild combined `companies`. Returns the renames performed."""
    done: list[str] = []
    tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
    for old, new in _RENAMES.items():
        if old in tables and new not in tables:
            con.execute(f'ALTER TABLE {old} RENAME TO {new}')
            done.append(f'{old} -> {new}')
            log.info(f'renamed table {old} -> {new}')
    rebuild_companies(con, None, {})
    return done


def main() -> None:
    """Resolve the DB from ingest.toml and migrate it in place."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(name)s: %(message)s')
    cfg = config.load(Path('ingest.toml'))
    con = storage.connect(cfg.data_root, cfg.db)
    try:
        done = migrate(con)
        log.info(f'migration complete — {len(done)} table(s) renamed, companies rebuilt')
    finally:
        con.close()


if __name__ == '__main__':
    main()
