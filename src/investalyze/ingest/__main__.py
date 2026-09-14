"""CLI entry: `python -m investalyze.ingest`. Parse args, run the orchestrator."""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable
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


def _parser() -> argparse.ArgumentParser:
    """The `python -m investalyze.ingest` argument parser."""
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
    parser.add_argument('--ticker', action='append', dest='tickers',
                        help='ticker to blacklist (repeatable), alternative to --universe')
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
    return parser


def _argument_problem(args: argparse.Namespace) -> str | None:
    """Why the parsed arguments cannot run their command, or None when they can."""
    if args.command == 'blacklist-ticker':
        if not args.reason or (not args.tickers and not args.universe) or (args.tickers and args.universe):
            return 'blacklist-ticker requires --reason and exactly one of --ticker (repeatable) or --universe'
    if args.command == 'add-listings' and not args.rows:
        return 'add-listings requires --rows'
    return None


def _housekeeping(cfg: config.Config, args: argparse.Namespace) -> None:
    """Run the chosen housekeeping tasks and log each result."""
    summary = housekeeping.run_housekeeping(cfg, args.tasks)
    for name, result in summary.items():
        log.info(f'{name}: {result}')


def _scan_listings(cfg: config.Config, args: argparse.Namespace) -> None:
    """Scan the exchange listings for new and reactivated tickers and log the counts."""
    with storage.connect(cfg.data_root, cfg.db, read_only=True) as con:
        result = listings.scan(con, cfg.data_root)
    counts = Counter(c.classification for c in result.candidates)
    log.info(f'scan complete: {len(result.candidates)} candidate(s) -> {result.saved_to}')
    for classification, n in sorted(counts.items()):
        log.info(f'  {classification}: {n}')
    new_tickers = sorted(c.ticker for c in result.candidates if c.classification == 'new')
    if new_tickers:
        log.info(f'  added: {", ".join(new_tickers)}')


def _blacklist_tickers(cfg: config.Config, args: argparse.Namespace) -> None:
    """Delete the given tickers, or a saved universe's tickers, and block future scans from re-adding them."""
    if args.universe:
        tickers = universes.load_universe(args.universe, universe_dir=cfg.data_root / 'universes')
    else:
        tickers = args.tickers
    with storage.connect(cfg.data_root, cfg.db) as con:
        deleted = listings.blacklist_tickers(con, cfg.data_root, tickers, args.reason)
    log.info(f'blacklisted {len(tickers)} ticker(s) ({", ".join(tickers)}): {deleted} row(s) deleted')


def _add_listings(cfg: config.Config, args: argparse.Namespace) -> None:
    """Add the reviewed listing candidates to ticker.csv."""
    rows = json.loads(args.rows)
    with storage.connect(cfg.data_root, cfg.db) as con:
        added, renamed, skipped = listings.add_listings(con, cfg.data_root, rows)
    log.info(f'added {len(added)} ticker(s), {renamed} renamed first, {skipped} skipped (no new ticker given)')
    if added:
        log.info(f'  added: {", ".join(added)}')


def _ingest(cfg: config.Config, args: argparse.Namespace) -> None:
    """Run the chosen providers and log the row count each one stored."""
    summary = orchestrator.run(cfg, args.providers, update=args.update)
    for name, rows in summary.items():
        log.info(f'{name}: {rows} rows')


def _setup(cfg: config.Config, args: argparse.Namespace) -> None:
    """Create the data directories."""
    orchestrator.create_data_dirs(cfg)
    log.info(f'data dirs ready under {cfg.data_root}')


_COMMANDS: dict[str | None, Callable[[config.Config, argparse.Namespace], None]] = {
    'setup': _setup,
    'housekeeping': _housekeeping,
    'scan-listings': _scan_listings,
    'blacklist-ticker': _blacklist_tickers,
    'add-listings': _add_listings,
    None: _ingest,
}


def main() -> None:
    """Parse CLI args and run the selected command, or the ingest when none is given."""
    load_dotenv()
    parser = _parser()
    args = parser.parse_args()
    problem = _argument_problem(args)
    if problem:
        parser.error(problem)

    cfg = config.read(args.config)
    if args.data_root is not None:
        cfg = replace(cfg, data_root=args.data_root)
    configure_logging(cfg.log_level)
    _COMMANDS[args.command](cfg, args)


if __name__ == '__main__':
    main()
