"""The one runner: open the shared DB via storage, then run each provider.

Each provider fetches + saves itself (saving through storage.write). Adding a
provider = one new folder under providers/ + one entry in PROVIDERS here. This is
also the one place that touches config — storage stays config-free plumbing.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import duckdb

from investalyze.ingest import storage
from investalyze.ingest.config import Config
from investalyze.ingest.providers.simfin import fundamental_data as simfin
from investalyze.ingest.providers.stooq import market_data as stooq
from investalyze.ingest.providers.yahoo import meta_data as yahoo_meta
from investalyze.ingest.providers.yahoo import price_data as yahoo

log = logging.getLogger('investalyze.ingest')

# name -> the provider's run(con, data_root, settings, *, update) -> rows loaded
ProviderRun = Callable[..., int]
PROVIDERS: dict[str, ProviderRun] = {
    'stooq': stooq.run,
    'yahoo': yahoo.run,
    'yahoo-meta': yahoo_meta.fetch_meta,
    'simfin': simfin.run,
}
SUBDIRS: tuple[str, ...] = ('raw', 'processed', 'state')
# Providers that reuse another provider's data tree instead of owning one. yahoo-meta
# reads yahoo's ticker.csv + blacklist and writes its own state alongside them under
# data/yahoo/state/ (meta_blacklist.csv / meta_dead.csv) — it has no tree of its own.
NO_DIR_PROVIDERS: frozenset[str] = frozenset({'yahoo-meta'})


def create_data_dirs(config: Config) -> None:
    """Create data/<provider>/{raw,processed,state} for every registered provider.

    Run this ONCE, up front, so source files can be dropped into each provider's
    `raw/` before the first load. Idempotent. Providers in `NO_DIR_PROVIDERS` are
    skipped — they live under another provider's tree.
    """
    for provider in PROVIDERS:
        if provider in NO_DIR_PROVIDERS:
            continue
        for sub in SUBDIRS:
            (Path(config.data_root) / provider / sub).mkdir(parents=True, exist_ok=True)


def run(config: Config, providers: Sequence[str] | None = None, *, update: bool = False) -> dict[str, int]:
    """Run the selected providers against the shared DB. Returns {provider: rows}.

    `providers=None` runs every registered provider. Opens one connection (at
    `config.data_root`/`config.db`), passes each provider its own settings, and
    closes the connection when done.
    """
    selected = list(providers) if providers is not None else list(PROVIDERS)
    con: duckdb.DuckDBPyConnection = storage.connect(config.data_root, config.db)
    try:
        results: dict[str, int] = {}
        for name in selected:
            log.info(f'running {name}')
            results[name] = PROVIDERS[name](con, config.data_root, config.provider(name), update=update)
        return results
    finally:
        con.close()
