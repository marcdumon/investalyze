"""The housekeeping runner: open the shared DB, then run each selected maintenance task.

Mirrors orchestrator.py's shape — a registry of name -> (provider name, task callable), one
place that opens the DB connection. Adding a task = one new function + one registry entry.
"""
import logging
from collections.abc import Callable, Sequence

from investalyze.ingest import storage
from investalyze.ingest.config import Config
from investalyze.ingest.providers.yahoo import meta as yahoo_meta
from investalyze.ingest.providers.yahoo import provider as yahoo

log = logging.getLogger('investalyze.ingest')

# name -> (settings section to use, the task's (con, data_root, settings) -> dict result)
HousekeepingTask = Callable[..., dict]
HOUSEKEEPING_TASKS: dict[str, tuple[str, HousekeepingTask]] = {
    'yahoo-blacklist': ('yahoo', yahoo.recheck_blacklist),
    'yahoo-meta-blacklist': ('yahoo-meta', yahoo_meta.recheck_meta_blacklist),
}


def run_housekeeping(config: Config, tasks: Sequence[str] | None = None) -> dict[str, dict]:
    """Run the selected housekeeping tasks against the shared DB. Returns {task: result}.

    `tasks=None` runs every registered task. Opens one connection (at `config.data_root`/
    `config.db`), passes each task its provider's settings, and closes the connection when done.
    """
    selected = list(tasks) if tasks is not None else list(HOUSEKEEPING_TASKS)
    con = storage.connect(config.data_root, config.db)
    try:
        results: dict[str, dict] = {}
        for name in selected:
            provider_name, task = HOUSEKEEPING_TASKS[name]
            log.info(f'housekeeping: {name}')
            results[name] = task(con, config.data_root, config.provider(provider_name))
        return results
    finally:
        con.close()
