"""CLI entry: `python -m investalyze.cleaning`. Check or apply the fixes in `cleaning.toml`.

`check` reports what each fix would touch (read-only); `apply` executes the fixes.
Both are safe to re-run: a clean fix matches 0 rows and is skipped. `detect` confirms the
target rows *exist*, not that the underlying quirk still *holds*: if a vendor ever replaces
bogus rows with real data, re-evaluate the entry by hand (each fix's `reason` points at the
evidence, usually notebooks/9999_data_quirks.ipynb).
"""

import argparse
import logging
from pathlib import Path

from investalyze.cleaning import registry
from investalyze.ingest import config, storage
from investalyze.ingest.logging import configure_logging

log = logging.getLogger('investalyze.cleaning')


def main() -> None:
    """Parse CLI args and run check/apply over all configured fixes."""
    parser = argparse.ArgumentParser(
        prog='python -m investalyze.cleaning',
        description='Apply persistent manual data corrections to the DuckDB.',
    )
    parser.add_argument('command', choices=('check', 'apply'),
                        help="'check' reports what each fix would touch, 'apply' executes the fixes")
    parser.add_argument('--config', type=Path, default=Path('cleaning.toml'),
                        help='fixes TOML (default: ./cleaning.toml)')
    parser.add_argument('--ingest-config', type=Path, default=Path('ingest.toml'),
                        help='ingest TOML giving the DB location (default: ./ingest.toml)')
    args = parser.parse_args()

    cfg = config.read(args.ingest_config)
    configure_logging(cfg.log_level)
    fixes = registry.read_fixes(args.config)

    con = storage.connect(cfg.data_root, cfg.db, read_only=args.command == 'check')
    for fix in fixes:
        module = registry.FIX_TYPES[fix.fix_type]
        span = f'{fix.start or "..."} .. {fix.end or "..."}'
        label = f'{fix.fix_type} {fix.table} {fix.tickers} [{span}]' + (f' {fix.column}' if fix.column else '')
        if args.command == 'check':
            n = module.detect(con, fix)
            state = 'clean' if n == 0 else 'pending'
            log.info(f'{label}: {n} rows ({state})')
        else:
            changed = module.apply(con, fix)
            if changed == 0:
                log.info(f'{label}: clean, skipped')
            else:
                log.info(f'{label}: changed {changed} rows')


if __name__ == '__main__':
    main()
