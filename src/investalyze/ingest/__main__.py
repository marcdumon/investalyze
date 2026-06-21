"""CLI entry: `python -m investalyze.ingest`. Parse args, run the orchestrator."""
import argparse
from dataclasses import replace
from pathlib import Path

from investalyze.ingest import config, orchestrator


def main() -> None:
    """Parse CLI args and run the selected providers."""
    parser = argparse.ArgumentParser(
        prog='python -m investalyze.ingest',
        description='Ingest market data from providers into the DuckDB.',
    )
    parser.add_argument('command', nargs='?', choices=('setup',),
                        help="'setup' scaffolds the data dirs and exits; omit to run the ingest")
    parser.add_argument('--config', type=Path, default=Path('ingest.toml'),
                        help='TOML config file (default: ./ingest.toml; missing is fine)')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='override the data dir from config')
    parser.add_argument('-p', '--provider', action='append', dest='providers',
                        choices=sorted(orchestrator.PROVIDERS),
                        help='provider to run; repeatable (default: all)')
    parser.add_argument('--update', action='store_true',
                        help='apply the daily update instead of a full load')
    args = parser.parse_args()

    cfg = config.load(args.config)
    if args.data_root is not None:
        cfg = replace(cfg, data_root=args.data_root)

    if args.command == 'setup':
        orchestrator.create_data_dirs(cfg)
        print(f'data dirs ready under {cfg.data_root}')
        return

    summary = orchestrator.run(cfg, args.providers, update=args.update)
    for name, rows in summary.items():
        print(f'{name}: {rows} rows')


if __name__ == '__main__':
    main()
