"""Shared named-universe files (data/universes/<name>.csv) used by every app page."""

import re
from pathlib import Path

import pandas as pd

UNIVERSE_DIR = Path(__file__).resolve().parents[3] / 'data' / 'universes'


def list_universes(universe_dir: Path = UNIVERSE_DIR) -> list[str]:
    """Names of every saved universe file."""
    if not universe_dir.exists():
        return []
    return sorted(p.stem for p in universe_dir.glob('*.csv'))


def save_universe(name: str, tickers: list[str], universe_dir: Path = UNIVERSE_DIR) -> str:
    """Write the selection to <universe_dir>/<name>.csv and return the cleaned name."""
    clean = re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_')
    if not clean:
        raise ValueError('universe name is empty')
    universe_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Ticker': sorted(tickers)}).to_csv(universe_dir / f'{clean}.csv', index=False)
    return clean


def load_universe(name: str, universe_dir: Path = UNIVERSE_DIR) -> list[str]:
    """Read a saved universe back as a ticker list."""
    return pd.read_csv(universe_dir / f'{name}.csv', keep_default_na=False)['Ticker'].tolist()
