"""CLI entry: `python -m investalyze.backtest`. Report and delete stored backtest runs.

Runs are made by `python -m investalyze.research backtest` (factor portfolios) and
`python -m investalyze.research timing` (timing rules), both built from the factor library under
`factors/`. `report` prints what a finished run wrote and `delete` removes stored runs.
"""

import argparse
import logging
from pathlib import Path

import duckdb

from investalyze.backtest import report, writer
from investalyze.ingest import config, storage
from investalyze.ingest.logging import configure_logging

log = logging.getLogger('investalyze.backtest')

COMMANDS = ('report', 'delete')


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser, kept separate so tests can inspect it without running a command."""
    parser = argparse.ArgumentParser(prog='python -m investalyze.backtest',
                                     description='Report and delete stored backtest runs.')
    parser.add_argument('command', choices=COMMANDS, metavar='command', help=f'one of: {", ".join(COMMANDS)}')
    parser.add_argument('--run-id', nargs='+', metavar='run_id',
                        help='runs to delete, or the one run to print, for the delete and report commands')
    parser.add_argument('--ingest-config', type=Path, default=Path('ingest.toml'),
                        help='ingest TOML giving the DB location (default: ./ingest.toml)')
    return parser


def _report(args: argparse.Namespace, data_root: Path) -> None:
    """Print the report a run wrote when it finished."""
    if args.run_id is None or len(args.run_id) != 1:
        raise SystemExit('report needs one --run-id')
    path = report.report_dir(args.run_id[0], data_root) / 'report.md'
    if not path.is_file():
        raise SystemExit(f'no report at {path}; the run may have been deleted')
    print(path.read_text())


def _delete(con: duckdb.DuckDBPyConnection, args: argparse.Namespace, data_root: Path) -> None:
    """Remove stored runs: their summary rows, equity curves, blotters and rendered reports.

    Every run named is dealt with before an unknown one is reported, so one bad id in a selection
    does not leave the rest of it half deleted.
    """
    if not args.run_id:
        raise SystemExit('delete needs --run-id')
    missing = []
    for identifier in args.run_id:
        removed = writer.delete_run(con, identifier)
        if not any(removed.values()):
            missing.append(identifier)
            continue
        written = 'report removed' if report.delete_report(identifier, data_root) else 'no report on disk'
        log.info(f'deleted run {identifier}: {removed["backtest_equity"]:,} equity rows, '
                 f'{removed["backtest_trades"]:,} fills, {written}')
    if missing:
        raise SystemExit(f'no stored run: {", ".join(missing)}')


def main() -> None:
    """Parse CLI args and dispatch to the selected command."""
    args = _parser().parse_args()
    cfg = config.read(args.ingest_config)
    configure_logging(cfg.log_level)

    if args.command == 'report':
        _report(args, cfg.data_root)
        return

    with storage.connect(cfg.data_root, cfg.db) as con:
        writer.ensure_tables(con)
        _delete(con, args, cfg.data_root)


if __name__ == '__main__':
    main()
