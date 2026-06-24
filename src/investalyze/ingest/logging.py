"""Console logging for the ingest: one colored StreamHandler, noisy libraries tamed.

`configure_logging` is called once at the CLI edge; everything else just uses
`logging.getLogger(...)`. The module name is safe — `import logging` here resolves
to the stdlib (absolute import), same as the reference repo.
"""

import logging

LOG_FMT = '%(asctime)s|%(levelname)s|%(name)s: %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

LEVEL_COLORS_ANSI = {
    logging.DEBUG: '\033[37m',
    logging.INFO: '\033[32m',
    logging.WARNING: '\033[33m',
    logging.ERROR: '\033[31m',
    logging.CRITICAL: '\033[91m',
}
_RESET = '\033[0m'


class _ColorFormatter(logging.Formatter):
    """Wrap each formatted line in its level's ANSI color."""

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS_ANSI.get(record.levelno, _RESET)
        return f'{color}{super().format(record)}{_RESET}'


def configure_logging(level: int | str = 'INFO') -> None:
    """Install one colored console handler on the root logger and quiet noisy libraries."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter(fmt=LOG_FMT, datefmt=LOG_DATEFMT))
    root.addHandler(handler)

    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
