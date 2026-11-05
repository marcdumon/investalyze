"""Locations: the config folder with the hand-edited settings, and the data folder with the data, the factor library and the
quality log.

The config folder ~/.config/md_apps/investalyze (the repo in development) holds `investalyze.toml` (not in git: the data folder, the port, the debug mode), `ingest.toml`, `backtest.toml`,
`cleaning.toml`, `stooq_tickers.toml` and `.env`. The data folder is `data_dir` of `investalyze.toml`."""

import sys
import tomllib
from pathlib import Path

APP_DIR = Path(sys.prefix).parent
# an installed app keeps its settings in ~/.config; a development copy (a repo) keeps its own, next to the code
CONFIG_DIR = APP_DIR if (APP_DIR / 'pyproject.toml').is_file() else Path.home() / '.config' / 'md_apps' / 'investalyze'
SETTINGS_FILE = CONFIG_DIR / 'investalyze.toml'
INGEST_CONFIG = CONFIG_DIR / 'ingest.toml'
CLEANING_CONFIG = CONFIG_DIR / 'cleaning.toml'


def settings() -> dict:
    """The contents of investalyze.toml; empty when the file is missing."""
    return tomllib.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.is_file() else {}


def data_dir() -> Path:
    """The data folder: data_dir of investalyze.toml, or the folder data in the app folder when it is not set."""
    value = settings().get('data_dir', '')
    return Path(value).expanduser() if value else APP_DIR / 'data'


def server() -> tuple[int, bool]:
    """The port of the app and whether it runs in debug mode: [server] of investalyze.toml, else 8050 and no debug."""
    section = settings().get('server', {})
    return int(section.get('port', 8050)), bool(section.get('debug', False))


DATA_ROOT = data_dir()
CACHE_DIR = DATA_ROOT / 'cache'
UNIVERSE_DIR = DATA_ROOT / 'universes'
FACTORS_DIR = DATA_ROOT / 'factors'
