"""CLI entry: `python -m investalyze.quality`. Run anomaly checks and store the findings.

Each run replaces the selected checks' rows in the `anomalies` table (delete-then-insert),
so the table always reflects the latest run. Review findings in
notebooks/3_data_quality.ipynb; fixes stay manual (quirks log -> cleaning.toml -> apply).
"""

import argparse
import logging
from pathlib import Path

from investalyze.ingest import config, storage
from investalyze.ingest.logging import configure_logging
from investalyze.quality import registry, writer

log = logging.getLogger('investalyze.quality')


def _epilog() -> str:
    """Render the check catalog (name, severity, description) shown at the end of `--help`."""
    width = max(len(name) for name in registry.CHECKS)
    lines = ['checks (default: all):']
    for name in sorted(registry.CHECKS):
        severity, _ = registry.CHECKS[name]
        lines.append(f'  {name:<{width}}  {severity:<5}  {registry.CHECK_DESCRIPTIONS[name]}')
    return '\n'.join(lines)


def main() -> None:
    """Parse CLI args and run the selected checks (default: all)."""
    parser = argparse.ArgumentParser(
        prog='python -m investalyze.quality',
        description='Detect data anomalies and store findings in the anomalies table.',
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('checks', nargs='*', choices=sorted(registry.CHECKS), metavar='check',
                        help='check to run (default: all); see the list below')
    parser.add_argument('--ingest-config', type=Path, default=Path('ingest.toml'),
                        help='ingest TOML giving the DB location (default: ./ingest.toml)')
    args = parser.parse_args()

    cfg = config.read(args.ingest_config)
    configure_logging(cfg.log_level)
    selected = args.checks or sorted(registry.CHECKS)

    con = storage.connect(cfg.data_root, cfg.db)
    try:
        writer.ensure_table(con)
        for name in selected:
            severity, check = registry.CHECKS[name]
            findings = check(con)
            n = writer.replace_findings(con, name, severity, findings)
            log.info(f'{name}: {n} findings ({severity})')
    finally:
        con.close()


if __name__ == '__main__':
    main()
