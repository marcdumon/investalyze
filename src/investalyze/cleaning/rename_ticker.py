"""Fix type: rename a ticker across every ticker-keyed table.

Used when a delisted ticker symbol is later reused by an unrelated company: the old company's
historical rows are moved to a disambiguated key (e.g. SPCX -> SPCX.1) so the new company can be
ingested under the bare symbol without merging with the old company's data. Unlike every other fix
type, this one is not scoped to one `table` in cleaning.toml — it always touches every table in
`storage.TICKER_KEYED_TABLES`, skipping any that don't exist in the connected DB.
"""

import duckdb

from investalyze.cleaning.fix import Fix
from investalyze.ingest.storage import TICKER_KEYED_TABLES, table_exists


def detect(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Row count still under the old ticker, summed across every ticker-keyed table that exists."""
    old = fix.tickers[0]
    total = 0
    for table in TICKER_KEYED_TABLES:
        if table_exists(con, table):
            total += int(con.execute(f'SELECT count(*) FROM {table} WHERE Ticker = ?', [old]).fetchone()[0])  # type: ignore[index]
    return total


def apply(con: duckdb.DuckDBPyConnection, fix: Fix) -> int:
    """Rename the old ticker to `fix.new_ticker` in every ticker-keyed table that exists."""
    old = fix.tickers[0]
    total = 0
    for table in TICKER_KEYED_TABLES:
        if table_exists(con, table):
            total += int(con.execute(
                f'UPDATE {table} SET Ticker = ? WHERE Ticker = ?', [fix.new_ticker, old],
            ).fetchone()[0])  # type: ignore[index]
    return total
