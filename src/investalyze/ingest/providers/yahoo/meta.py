"""Yahoo provider — company profile + officers, via `yf.Ticker(t).info`.

Independent of `provider.py` (prices/dividends/splits): one HTTP call per ticker (no
bulk endpoint for `.info`, unlike `yf.download`), so this fetches one ticker at a time
and paces itself with `sleep` between every call. Reuses `provider.py`'s ticker universe
and blacklist/dead state helpers — it has no ticker list or CSV-schema helpers of its own.
"""
import logging
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

from investalyze.ingest import storage
from investalyze.ingest.providers.yahoo import provider

log = logging.getLogger('investalyze.ingest.yahoo-meta')

_PROFILE, _OFFICERS = 'company_profile', 'company_officers'
_PROFILE_KEY = ['Ticker', 'Src']
_OFFICERS_KEY = ['Ticker', 'Src', 'name']
_PROFILE_COLS = ['address1', 'city', 'state', 'zip', 'country', 'website', 'industry', 'sector',
                 'longBusinessSummary', 'fullTimeEmployees', 'auditRisk', 'boardRisk', 'compensationRisk',
                 'shareHolderRightsRisk', 'overallRisk', 'irWebsite']
_OFFICER_COLS = ['name', 'title', 'age', 'yearBorn', 'fiscalYear', 'totalPay', 'exercisedValue', 'unexercisedValue']


def _fetch_info(symbol: str) -> dict:
    """One ticker's yfinance `.info` dict (empty dict on any failure).

    Broad except: yfinance raises a variety of unrelated exception types (HTTP errors,
    JSON decode errors, malformed-response KeyErrors) for a ticker with no profile data;
    all of them mean the same thing here — treat it like an empty result.
    """
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return {}
    return info if info else {}


def _to_profile(ticker: str, info: dict, fetched_on: date) -> pd.DataFrame:
    """One ticker's `.info` -> a single `company_profile` row."""
    row = {'Ticker': ticker, 'Src': 'yahoo'}
    for col in _PROFILE_COLS:
        row[col] = info.get(col)
    row['FetchedOn'] = fetched_on
    return pd.DataFrame([row])


def _to_officers(ticker: str, info: dict) -> pd.DataFrame:
    """One ticker's `companyOfficers` -> `company_officers` rows (empty if none)."""
    officers = info.get('companyOfficers') or []
    rows = []
    for officer in officers:
        row = {'Ticker': ticker, 'Src': 'yahoo'}
        for col in _OFFICER_COLS:
            row[col] = officer.get(col)
        rows.append(row)
    return pd.DataFrame(rows, columns=['Ticker', 'Src'] + _OFFICER_COLS)


def _load_existing_profile(con: duckdb.DuckDBPyConnection) -> dict[str, date]:
    """Each Yahoo ticker's stored `FetchedOn` (empty dict if the table is absent)."""
    if _PROFILE not in {t for (t,) in con.execute('SHOW TABLES').fetchall()}:
        return {}
    rows = con.execute(f"SELECT Ticker, FetchedOn FROM {_PROFILE} WHERE Src = 'yahoo'").fetchall()
    return dict(rows)


def _is_due(fetched_on: date | None, refresh_days: int, today: date) -> bool:
    """True if never fetched, or fetched more than `refresh_days` days before `today`."""
    return fetched_on is None or (today - fetched_on).days >= refresh_days


def fetch_meta(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict, *, update: bool = False) -> int:
    """Fetch + store Yahoo company profile + officers for due tickers. Returns the company_profile row count.

    `settings` is the `[yahoo-meta]` config (no fallback defaults — a missing key raises). `update`
    is accepted for signature parity with every other provider's `run` but unused — there is no
    incremental mode here, only "due" (no row yet, or `FetchedOn` stale) vs "not due".
    """
    price_raw_dir = data_root / 'yahoo' / 'raw'
    price_state_dir = data_root / 'yahoo' / 'state'
    state_dir = data_root / 'yahoo-meta' / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)

    ticker_df = pd.read_csv(price_raw_dir / 'ticker.csv')
    symbols = ticker_df['ticker'].tolist()
    market_by_ticker = dict(zip(ticker_df['ticker'], ticker_df['market']))

    price_blacklisted = set(provider._load_blacklist(price_state_dir / 'blacklist.csv')['ticker'])
    price_dead = set(provider._load_dead(price_state_dir / 'dead.csv')['ticker'])

    blacklist_file = state_dir / 'blacklist.csv'
    orig_blacklist_df = provider._load_blacklist(blacklist_file)
    meta_blacklisted = set(orig_blacklist_df['ticker'])
    meta_dead = set(provider._load_dead(state_dir / 'dead.csv')['ticker'])

    today = date.today()
    existing = _load_existing_profile(con)
    candidates = [s for s in symbols
                  if s not in price_blacklisted and s not in price_dead
                  and s not in meta_blacklisted and s not in meta_dead]
    todo = [s for s in candidates if _is_due(existing.get(s), settings['refresh_days_meta'], today)]

    newly_blacklisted: list[str] = []
    batches = provider._chunk(todo, settings['batch_size'])
    for i, batch in enumerate(batches):
        batch_profiles: list[pd.DataFrame] = []
        batch_officers: list[pd.DataFrame] = []
        for sym in batch:
            info = _fetch_info(sym)
            if not info:
                newly_blacklisted.append(sym)
                log.debug(f'{sym} no metadata — blacklisted')
            else:
                batch_profiles.append(_to_profile(sym, info, today))
                officers = _to_officers(sym, info)
                if not officers.empty:
                    batch_officers.append(officers)
            if settings['sleep']:
                time.sleep(settings['sleep'])
        if batch_profiles:
            storage.write(con, _PROFILE, pd.concat(batch_profiles, ignore_index=True), key=_PROFILE_KEY)
        if batch_officers:
            storage.write(con, _OFFICERS, pd.concat(batch_officers, ignore_index=True), key=_OFFICERS_KEY)
        if newly_blacklisted:
            today_iso = today.isoformat()
            new_rows = pd.DataFrame({
                'ticker': newly_blacklisted,
                'market': [market_by_ticker.get(s, '') for s in newly_blacklisted],
                'attempts': 1,
                'first_blacklisted': today_iso,
                'last_checked': today_iso,
            }).drop_duplicates('ticker', keep='first')
            pd.concat([orig_blacklist_df, new_rows], ignore_index=True).sort_values('ticker').to_csv(blacklist_file, index=False)
        log.info(f'batch {i + 1}/{len(batches)} — saved {len(batch_profiles)} blacklisted {len(batch) - len(batch_profiles)} (n={len(todo)})')

    log.info(f'done — {len(todo) - len(newly_blacklisted)} saved, {len(newly_blacklisted)} blacklisted')
    tables = {t for (t,) in con.execute('SHOW TABLES').fetchall()}
    row = con.execute(f'SELECT COUNT(*) FROM {_PROFILE}').fetchone() if _PROFILE in tables else None
    return int(row[0]) if row is not None else 0


def recheck_meta_blacklist(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict) -> dict:
    """Retry every yahoo-meta-blacklisted ticker; revive successes, age out chronic failures.

    `con` is unused — kept so this matches the `(con, data_root, settings)` shape every housekeeping
    task is dispatched with. `settings` is the `[yahoo-meta]` config: uses `batch_size`, `sleep`,
    `blacklist_max_attempts` (no fallback — missing raises `KeyError`). Unlike the price provider's
    `recheck_blacklist`, a revived ticker needs no further bookkeeping here — `yahoo-meta` has no
    ticker list of its own to prune; the next `fetch_meta` run picks a revived ticker up naturally
    once it's off this blacklist.
    """
    state_dir = data_root / 'yahoo-meta' / 'state'
    blacklist_file = state_dir / 'blacklist.csv'
    dead_file = state_dir / 'dead.csv'

    blacklist_df = provider._load_blacklist(blacklist_file)
    if blacklist_df.empty:
        return {'rechecked': 0, 'revived': 0, 'died': 0}

    max_attempts = settings['blacklist_max_attempts']
    today = date.today().isoformat()
    tickers = blacklist_df['ticker'].tolist()
    batches = provider._chunk(tickers, settings['batch_size'])

    revived_rows: list[dict] = []
    still_blacklisted: list[dict] = []
    died_rows: list[dict] = []
    for batch in batches:
        records = blacklist_df[blacklist_df['ticker'].isin(batch)].to_dict('records')
        for record in records:
            info = _fetch_info(record['ticker'])
            if info:
                revived_rows.append(record)
            else:
                record['attempts'] += 1
                record['last_checked'] = today
                if record['attempts'] >= max_attempts:
                    died_rows.append({'ticker': record['ticker'], 'attempts': record['attempts'],
                                      'first_blacklisted': record['first_blacklisted'], 'died_on': today})
                else:
                    still_blacklisted.append(record)
            if settings['sleep']:
                time.sleep(settings['sleep'])

    pd.DataFrame(still_blacklisted, columns=provider._BLACKLIST_COLS).to_csv(blacklist_file, index=False)
    if died_rows:
        dead_df = pd.concat([provider._load_dead(dead_file), pd.DataFrame(died_rows, columns=provider._DEAD_COLS)], ignore_index=True)
        dead_df.to_csv(dead_file, index=False)

    return {'rechecked': len(tickers), 'revived': len(revived_rows), 'died': len(died_rows)}
