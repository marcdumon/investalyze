"""The one runner: open the shared DB via storage, then run each provider.

Each provider fetches + saves itself (saving through storage.write). Adding a
provider = one new folder under providers/ + one entry in PROVIDERS here.
"""
from collections.abc import Callable, Sequence

import duckdb

from investalyze.ingest import storage
from investalyze.ingest.config import Config
from investalyze.ingest.providers.stooq import provider as stooq

# name -> the provider's run(con, data_root, settings, *, update) -> rows loaded
ProviderRun = Callable[..., int]
PROVIDERS: dict[str, ProviderRun] = {
    'stooq': stooq.run,
}


def run(config: Config, providers: Sequence[str] | None = None, *, update: bool = False) -> dict[str, int]:
    """Run the selected providers against the shared DB. Returns {provider: rows}.

    `providers=None` runs every registered provider. Opens one connection (at
    `config.data_root`/`config.db`), passes each provider its own settings, and
    closes the connection when done.
    """
    selected = list(providers) if providers is not None else list(PROVIDERS)
    con: duckdb.DuckDBPyConnection = storage.connect(config.data_root, config.db)
    try:
        return {
            name: PROVIDERS[name](con, config.data_root, config.provider(name), update=update)
            for name in selected
        }
    finally:
        con.close()
