"""CLI entry: `python -m investalyze.ingest`. Parse args, run the orchestrator."""

import argparse
import json
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from investalyze.apps import universes
from investalyze.ingest import config, housekeeping, listings, orchestrator, storage
from investalyze.ingest.logging import configure_logging

log = logging.getLogger('investalyze.ingest')


def _table(title: str, descriptions: dict[str, str]) -> list[str]:
    """Render one `title:` section listing each name and its one-line description, sorted."""
    width = max(len(name) for name in descriptions)
    lines = [f'{title}:']
    for name in sorted(descriptions):
        lines.append(f'  {name:<{width}}  {descriptions[name]}')
    return lines


def _epilog() -> str:
    """Render the provider and housekeeping-task catalogs shown at the end of `--help`."""
    lines = _table('providers (-p, default: all)', orchestrator.PROVIDER_DESCRIPTIONS)
    lines.append('')
    lines += _table('housekeeping tasks (-t, default: all)', housekeeping.TASK_DESCRIPTIONS)
    return '\n'.join(lines)


def main() -> None:
    """Parse CLI args and run the selected providers."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog='python -m investalyze.ingest',
        description='Ingest market data from providers into the DuckDB.',
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'command',
        nargs='?',
        choices=('setup', 'housekeeping', 'scan-listings', 'blacklist-ticker', 'add-listings'),
        help=("'setup' scaffolds the data dirs, 'housekeeping' runs maintenance tasks, "
              "'scan-listings' checks for new/reactivated tickers, 'blacklist-ticker' deletes a "
              "ticker (or every ticker in a saved universe) and blocks future scans from "
              "re-adding it, 'add-listings' adds reviewed candidates to ticker.csv, renaming a "
              'collision first (all exit without ingesting); omit to run the ingest'),
    )
    parser.add_argument('--config', type=Path, default=Path('ingest.toml'),
                        help='TOML config file (default: ./ingest.toml; missing is fine)')
    parser.add_argument('--data-root', type=Path, default=None, help='override the data dir from config')
    parser.add_argument('--ticker', help='ticker to blacklist (with blacklist-ticker, alternative to --universe)')
    parser.add_argument('--universe', help='saved universe whose tickers to blacklist (alternative to --ticker)')
    parser.add_argument('--reason', help='reason for blacklisting (required with blacklist-ticker)')
    parser.add_argument('--rows', help='JSON list of {ticker, market, classification, new_ticker} (required with add-listings)')
    parser.add_argument(
        '-p', '--provider', action='append', dest='providers', choices=sorted(orchestrator.PROVIDERS),
        help='provider to run; repeatable (default: all)',
    )
    parser.add_argument(
        '-t',
        '--task',
        action='append',
        dest='tasks',
        choices=sorted(housekeeping.HOUSEKEEPING_TASKS),
        help='housekeeping task to run; repeatable (default: all)',
    )
    parser.add_argument('--update', action='store_true', help='apply the daily update instead of a full load')
    args = parser.parse_args()

    cfg = config.read(args.config)
    if args.data_root is not None:
        cfg = replace(cfg, data_root=args.data_root)
    configure_logging(cfg.log_level)

    if args.command == 'setup':
        orchestrator.create_data_dirs(cfg)
        log.info(f'data dirs ready under {cfg.data_root}')
        return

    if args.command == 'housekeeping':
        summary = housekeeping.run_housekeeping(cfg, args.tasks)
        for name, result in summary.items():
            log.info(f'{name}: {result}')
        return

    if args.command == 'scan-listings':
        con = storage.connect(cfg.data_root, cfg.db, read_only=True)
        try:
            result = listings.scan(con, cfg.data_root)
        finally:
            con.close()
        counts = Counter(c.classification for c in result.candidates)
        log.info(f'scan complete: {len(result.candidates)} candidate(s) -> {result.saved_to}')
        for classification, n in sorted(counts.items()):
            log.info(f'  {classification}: {n}')
        new_tickers = sorted(c.ticker for c in result.candidates if c.classification == 'new')
        if new_tickers:
            log.info(f'  added: {", ".join(new_tickers)}')
        return

    if args.command == 'blacklist-ticker':
        if not args.reason or (not args.ticker and not args.universe) or (args.ticker and args.universe):
            parser.error('blacklist-ticker requires --reason and exactly one of --ticker or --universe')
        if args.universe:
            tickers = universes.load_universe(args.universe, universe_dir=cfg.data_root / 'universes')
        else:
            tickers = [args.ticker]
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            deleted = listings.blacklist_tickers(con, cfg.data_root, tickers, args.reason)
        finally:
            con.close()
        log.info(f'blacklisted {len(tickers)} ticker(s) ({", ".join(tickers)}): {deleted} row(s) deleted')
        return

    if args.command == 'add-listings':
        if not args.rows:
            parser.error('add-listings requires --rows')
        rows = json.loads(args.rows)
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            added, renamed, skipped = listings.add_listings(con, cfg.data_root, rows)
        finally:
            con.close()
        log.info(f'added {added} ticker(s), {renamed} renamed first, {skipped} skipped (no new ticker given)')
        return

    summary = orchestrator.run(cfg, args.providers, update=args.update)
    for name, rows in summary.items():
        log.info(f'{name}: {rows} rows')


if __name__ == '__main__':
    main()
