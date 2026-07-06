"""Blacklist/dead-ticker CSV schema and readers, shared by the yahoo price and meta providers.

Both `price_data.py` and `meta_data.py` track failed tickers in their own blacklist/dead CSVs
under `data/yahoo/state/`; this module owns the shared shape and reading. Writing stays with
each provider — their retry/promotion logic differs enough to not be worth unifying.
"""

from pathlib import Path

import pandas as pd

BLACKLIST_COLS = ['ticker', 'market', 'attempts', 'first_blacklisted', 'last_checked']
DEAD_COLS = ['ticker', 'attempts', 'first_blacklisted', 'died_on']


def read_blacklist(path: Path) -> pd.DataFrame:
    """Blacklist records (empty frame with the right columns if the file is absent)."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=BLACKLIST_COLS)


def read_dead(path: Path) -> pd.DataFrame:
    """Permanently-dead records (empty frame with the right columns if the file is absent)."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=DEAD_COLS)
