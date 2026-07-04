"""The housekeeping runner: open the shared DB, then run each selected maintenance task.

Mirrors orchestrator.py's shape — a registry of name -> (provider name, task callable), one
place that opens the DB connection. Adding a task = one new function + one registry entry.
"""

import logging
from collections.abc import Callable, Sequence

import duckdb

from investalyze.ingest import storage
from investalyze.ingest.config import Config
from investalyze.ingest.providers.stooq import instrument_names
from investalyze.ingest.providers.yahoo import meta_data as yahoo_meta
from investalyze.ingest.providers.yahoo import price_data as yahoo

log = logging.getLogger('investalyze.ingest')

HousekeepingTask = Callable[..., dict]


def ticker_diff(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Tickers held by one provider but not the other.

    Compares distinct tickers in the yahoo `prices` table against the simfin `_simfin_companies`
    table. Returns sorted lists under 'simfin_only' (in simfin, missing from yahoo) and 'yahoo_only'
    (in yahoo, missing from simfin).
    """
    simfin_only = [t for (t,) in con.execute(
        'SELECT Ticker FROM _simfin_companies EXCEPT SELECT Ticker FROM prices ORDER BY Ticker').fetchall()]
    yahoo_only = [t for (t,) in con.execute(
        'SELECT Ticker FROM prices EXCEPT SELECT Ticker FROM _simfin_companies ORDER BY Ticker').fetchall()]
    log.info(f'ticker diff: {len(simfin_only)} simfin-only, {len(yahoo_only)} yahoo-only')
    return {'simfin_only': simfin_only, 'yahoo_only': yahoo_only}


def rebuild_companies(con: duckdb.DuckDBPyConnection, data_root, settings: dict) -> dict[str, int]:
    """Rebuild the combined `companies` table from `_yahoo_companies` + `_simfin_companies`.

    Full outer join on Ticker; yahoo wins the overlapping columns (Industry, Sector, NrEmployees,
    BusinessSummary). `data_root`/`settings` are accepted to match the housekeeping task shape and
    are unused. Returns row counts: total, in_yahoo, in_simfin, both.
    """
    con.execute("""
        CREATE OR REPLACE TABLE companies AS
        SELECT
            COALESCE(y.Ticker, s.Ticker)                     AS Ticker,
            (y.Ticker IS NOT NULL)                           AS InYahoo,
            (s.Ticker IS NOT NULL)                           AS InSimfin,
            COALESCE(y.Industry, s.Industry)                 AS Industry,
            COALESCE(y.Sector, s.Sector)                     AS Sector,
            COALESCE(y.FullTimeEmployees, s.NumberEmployees) AS NrEmployees,
            s.CompanyName                                    AS CompanyName,
            y.Address1                                       AS Address,
            y.City                                           AS City,
            y.State                                          AS State,
            y.Zip                                            AS Zip,
            y.Country                                        AS Country,
            s.ISIN                                           AS ISIN,
            s.CIK                                            AS CIK,
            y.Website                                        AS Website,
            y.IRWebsite                                      AS IRWebsite,
            COALESCE(y.BusinessSummary, s.BusinessSummary)   AS BusinessSummary
        FROM _yahoo_companies y
        FULL OUTER JOIN _simfin_companies s ON y.Ticker = s.Ticker
    """)
    row = con.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE InYahoo),
               count(*) FILTER (WHERE InSimfin),
               count(*) FILTER (WHERE InYahoo AND InSimfin)
        FROM companies
    """).fetchone()
    result = {'rows': row[0], 'in_yahoo': row[1], 'in_simfin': row[2], 'both': row[3]}  # type: ignore [count() never returns None]
    log.info(f'rebuilt companies: {result}')
    return result


def rebuild_market_instruments(con: duckdb.DuckDBPyConnection, data_root, settings: dict) -> dict[str, int]:
    """Rebuild `market_instruments`: one row per distinct market_data ticker, name + country decoded.

    Indices come from the manually curated `[indices]` table in `stooq_tickers.toml`; bonds and
    currencies are decoded from their systematic ticker patterns (see
    investalyze.ingest.providers.stooq.instrument_names). `data_root`/`settings` are accepted to
    match the housekeeping task shape and are unused. Returns row counts: total, decoded, undecoded.
    """
    tickers = con.execute('SELECT DISTINCT Ticker, AssetClass FROM market_data').fetchall()
    rows = []
    for ticker, asset_class in tickers:
        described = instrument_names.describe(ticker)
        name, country = described if described is not None else (None, None)
        rows.append((ticker, name, country, asset_class))

    con.execute('CREATE OR REPLACE TABLE market_instruments (Ticker VARCHAR, Name VARCHAR, Country VARCHAR, AssetClass VARCHAR)')
    if rows:
        con.executemany('INSERT INTO market_instruments VALUES (?, ?, ?, ?)', rows)

    row = con.execute('SELECT count(*), count(*) FILTER (WHERE Name IS NOT NULL) FROM market_instruments').fetchone()
    result = {'rows': row[0], 'decoded': row[1], 'undecoded': row[0] - row[1]}  # type: ignore [count() never returns None]
    log.info(f'rebuilt market_instruments: {result}')
    return result


# name -> (settings section to use, the task's (con, data_root, settings) -> dict result)
HOUSEKEEPING_TASKS: dict[str, tuple[str, HousekeepingTask]] = {
    'yahoo-blacklist': ('yahoo', yahoo.recheck_blacklist),
    'yahoo-meta-blacklist': ('yahoo-meta', yahoo_meta.recheck_meta_blacklist),
    'companies': ('combined', rebuild_companies),
    'market-instruments': ('combined', rebuild_market_instruments),
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
