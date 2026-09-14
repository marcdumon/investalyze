"""Repository locations: the data directory, the factor library and the files beside them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'data'
CACHE_DIR = DATA_ROOT / 'cache'
UNIVERSE_DIR = DATA_ROOT / 'universes'
FACTORS_DIR = REPO_ROOT / 'factors'
