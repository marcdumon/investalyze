"""Discover live US common-stock listings missing from ticker.csv, and tell apart a genuinely
new ticker from a delisted symbol reused by a different company.

Two free, unauthenticated sources: NASDAQ Trader's symbol directory (what's listed today, on
which exchange) and SEC's company_tickers.json (ticker -> CIK, used only to check identity on a
reactivated symbol). Network fetching stays in `fetch_nasdaq_listings`/`fetch_sec_cik_map`; the
parsing and classification functions are pure and take plain text/dicts, so they're testable
without HTTP calls.
"""

import io
import logging
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import requests

from investalyze.cleaning import registry as cleaning_registry
from investalyze.ingest import storage
from investalyze.ingest.providers.yahoo import blacklist

log = logging.getLogger('investalyze.ingest.listings')

NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
OTHER_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
HEADERS = {'User-Agent': 'investalyze ingest (7793950+marcdumon@users.noreply.github.com)'}

# security-name pattern that signals a warrant/right/preferred/unit rather than common stock
_DROP_PATTERN = re.compile(r'\b(warrants?|rights?|preferred|units?)\b')
_EXCHANGE_MARKET = {'N': 'nyse', 'A': 'nysemkt'}


def _strip_footer(text: str) -> str:
    """Drop the NASDAQ Trader footer line ('File Creation Time: ...'), if present."""
    lines = text.strip().splitlines()
    if lines and lines[-1].startswith('File Creation Time'):
        lines = lines[:-1]
    return '\n'.join(lines)


def _is_common_stock(security_name: str) -> bool:
    """False when the security name signals a warrant/right/preferred/unit rather than common stock."""
    # heuristic only: anything that slips through and later returns no price data is caught by
    # the existing yahoo blacklist/dead mechanism, the same as it is today
    return not _DROP_PATTERN.search(security_name.lower())


def _drop_blank_tickers(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a blank/missing ticker.

    Vendor data occasionally has a blank symbol on an otherwise-parseable row; a blank ticker can
    never be fetched or tracked, and would crash downstream code (e.g. yfinance) if it ever
    reached ticker.csv.
    """
    return frame[frame['ticker'].notna() & (frame['ticker'].str.strip() != '')].reset_index(drop=True)


def _parse_nasdaqlisted(text: str) -> pd.DataFrame:
    """NASDAQ-listed common stocks from nasdaqlisted.txt: columns ticker, market, company_name."""
    frame = pd.read_csv(io.StringIO(_strip_footer(text)), sep='|', dtype=str)
    frame = frame[(frame['Test Issue'] == 'N') & (frame['ETF'] == 'N')]
    frame = frame[frame['Security Name'].apply(_is_common_stock)]
    listings = pd.DataFrame({'ticker': frame['Symbol'], 'market': 'nasdaq', 'company_name': frame['Security Name']})
    return _drop_blank_tickers(listings)


def _parse_otherlisted(text: str) -> pd.DataFrame:
    """NYSE / NYSE American common stocks from otherlisted.txt: columns ticker, market, company_name.

    Other exchanges in the file (Arca, Cboe BZX, IEX, ...) fall outside today's tracked markets and
    are dropped, matching ticker.csv's existing nasdaq/nyse/nysemkt scope.
    """
    frame = pd.read_csv(io.StringIO(_strip_footer(text)), sep='|', dtype=str)
    frame = frame[(frame['Test Issue'] == 'N') & (frame['ETF'] == 'N')]
    frame = frame[frame['Security Name'].apply(_is_common_stock)]
    frame = frame[frame['Exchange'].isin(_EXCHANGE_MARKET)]
    listings = pd.DataFrame({'ticker': frame['ACT Symbol'], 'market': frame['Exchange'].map(_EXCHANGE_MARKET),
                             'company_name': frame['Security Name']})
    return _drop_blank_tickers(listings)


def fetch_nasdaq_listings() -> pd.DataFrame:
    """Live NASDAQ/NYSE/NYSE American common-stock listings: columns ticker, market, company_name."""
    nasdaq_response = requests.get(NASDAQ_LISTED_URL, headers=HEADERS, timeout=15)
    nasdaq_response.raise_for_status()
    other_response = requests.get(OTHER_LISTED_URL, headers=HEADERS, timeout=15)
    other_response.raise_for_status()
    return pd.concat([_parse_nasdaqlisted(nasdaq_response.text), _parse_otherlisted(other_response.text)], ignore_index=True)


def _parse_sec_cik_map(payload: dict) -> dict[str, int]:
    """SEC's company_tickers.json payload -> {ticker: cik}, tickers upper-cased."""
    return {str(entry['ticker']).upper(): int(entry['cik_str']) for entry in payload.values()}


def fetch_sec_cik_map() -> dict[str, int]:
    """Live SEC ticker -> CIK mapping, from company_tickers.json."""
    response = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return _parse_sec_cik_map(response.json())


@dataclass(frozen=True)
class Candidate:
    """One live ticker not currently tracked, with its universe-scan classification."""

    ticker: str
    market: str
    company_name: str
    classification: str   # 'new' | 'reactivated_safe' | 'reactivated_collision'
    suggested_new_ticker: str | None = None
    previous_company_name: str | None = None


def classify(listings: pd.DataFrame, tracked: set[str], dead_tickers: set[str], live_cik: dict[str, int],
            last_cik: dict[str, int | None], last_company_name: dict[str, str | None] | None = None) -> list[Candidate]:
    """Classify every live listing not already tracked as new, reactivated_safe, or reactivated_collision.

    A ticker not in `dead_tickers` is brand new. A ticker in `dead_tickers` (tracked before, now
    dead) is reactivated_safe only when its live SEC CIK matches the CIK on file from the last time
    it was tracked. A mismatch or missing CIK is reactivated_collision. An unknown CIK is treated
    as a collision. `last_company_name` (optional) fills in `previous_company_name` so a collision
    shows which earlier company held the symbol, not just the live listing's current name.
    """
    last_company_name = last_company_name or {}
    candidates = []
    for _, row in listings.iterrows():
        ticker = row['ticker']
        if ticker in tracked:
            continue
        if ticker not in dead_tickers:
            candidates.append(Candidate(ticker, row['market'], row['company_name'], 'new'))
            continue
        live = live_cik.get(ticker)
        known = last_cik.get(ticker)
        classification = 'reactivated_safe' if live is not None and known is not None and live == known \
            else 'reactivated_collision'
        candidates.append(Candidate(ticker, row['market'], row['company_name'], classification,
                                    previous_company_name=last_company_name.get(ticker)))
    return candidates


def next_free_suffix(ticker: str, existing: set[str]) -> str:
    """The first `{ticker}.N` key (N starting at 1) not already present in `existing`."""
    n = 1
    while f'{ticker}.{n}' in existing:
        n += 1
    return f'{ticker}.{n}'


CANDIDATES_FILE = 'listing_candidates.csv'


@dataclass(frozen=True)
class UniverseScanResult:
    """Everything one listings scan produced: the classified candidates and where they were saved."""

    candidates: list[Candidate]
    saved_to: Path


def _tickers_with_rows(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Every ticker holding at least one row in any ticker-keyed table."""
    known: set[str] = set()
    for table in storage.TICKER_KEYED_TABLES:
        if storage.table_exists(con, table):
            known |= {ticker for (ticker,) in con.execute(f'SELECT DISTINCT Ticker FROM {table}').fetchall()}
    return known


def _read_dead_ciks(con: duckdb.DuckDBPyConnection, data_root: Path, tracked: set[str],
                    ) -> tuple[set[str], dict[str, int | None], dict[str, str | None]]:
    """Every ticker needing an identity check before being offered as a candidate, with the CIK and
    company name on file for it.

    This is every ticker in price_dead.csv/meta_dead.csv (tracked before, now fully dead) plus any
    ticker holding rows in a ticker-keyed table but currently untracked (the state a ticker sits in
    while blacklisted but not yet aged out to dead) - both mean this symbol has history, and any
    live listing under it needs its identity verified before being treated as brand new.
    """
    state_dir = data_root / 'yahoo' / 'state'
    dead_tickers = set(blacklist.read_dead(state_dir / 'price_dead.csv')['ticker']) | \
        set(blacklist.read_dead(state_dir / 'meta_dead.csv')['ticker'])
    with_rows = _tickers_with_rows(con)
    needs_check = dead_tickers | (with_rows - tracked)
    if not needs_check or not storage.table_exists(con, 'companies'):
        return needs_check, {}, {}
    placeholders = ', '.join('?' for _ in needs_check)
    rows = con.execute(
        f'SELECT Ticker, CIK, CompanyName FROM companies WHERE Ticker IN ({placeholders})', list(needs_check),
    ).fetchall()
    last_cik = {ticker: (int(cik) if cik is not None else None) for ticker, cik, _ in rows}
    last_company_name = {ticker: name for ticker, _, name in rows}
    return needs_check, last_cik, last_company_name


def _ticker_row_count(con: duckdb.DuckDBPyConnection, ticker: str) -> int:
    """Row count for `ticker`, summed across every ticker-keyed table that exists.

    Deliberately duplicates rename_ticker.detect()'s loop rather than importing it, see this
    plan's "Design note: intentional duplication."
    """
    return sum(
        int(con.execute(f'SELECT count(*) FROM {table} WHERE Ticker = ?', [ticker]).fetchone()[0])  # type: ignore[index]
        for table in storage.TICKER_KEYED_TABLES if storage.table_exists(con, table)
    )


def _resolve_completed_renames(con: duckdb.DuckDBPyConnection, data_root: Path, cleaning_config: Path,
                               dead_tickers: set[str]) -> set[str]:
    """Drop dead-list tickers whose staged rename_ticker fix has fully moved their rows; returns the survivors."""
    if not cleaning_config.exists():
        return dead_tickers
    fixes = cleaning_registry.parse_fixes(cleaning_config.read_text())
    staged = {fix.tickers[0] for fix in fixes if fix.fix_type == 'rename_ticker'}
    resolved = {ticker for ticker in dead_tickers & staged if _ticker_row_count(con, ticker) == 0}
    if resolved:
        state_dir = data_root / 'yahoo' / 'state'
        for name in ('price_dead.csv', 'meta_dead.csv'):
            path = state_dir / name
            df = blacklist.read_dead(path)
            if not df.empty:
                df[~df['ticker'].isin(resolved)].to_csv(path, index=False)
    return dead_tickers - resolved


MANUAL_BLACKLIST_FILE = 'manual_blacklist.csv'
_MANUAL_BLACKLIST_COLS = ('ticker', 'reason', 'blacklisted_on')


def _read_manual_blacklist(data_root: Path) -> set[str]:
    """Tickers the user has explicitly blacklisted; empty set if none have been."""
    path = data_root / 'yahoo' / 'state' / MANUAL_BLACKLIST_FILE
    if not path.exists():
        return set()
    return set(pd.read_csv(path, keep_default_na=False)['ticker'])


_BLACKLIST_CHUNK_SIZE = 200


def blacklist_tickers(con: duckdb.DuckDBPyConnection, data_root: Path, tickers: list[str], reason: str) -> int:
    """Delete every row for each of `tickers`, drop them from ticker.csv, and record each one with
    the same `reason` so no future scan offers it again. Returns the number of rows deleted.

    Processed in chunks of `_BLACKLIST_CHUNK_SIZE`: each chunk's DB deletes and file rewrites
    complete before the next chunk starts, so progress is logged as it goes and an interruption
    only ever leaves the current chunk incomplete, not the whole batch.
    """
    if not tickers:
        return 0
    total = 0
    for start in range(0, len(tickers), _BLACKLIST_CHUNK_SIZE):
        chunk = tickers[start:start + _BLACKLIST_CHUNK_SIZE]
        total += _blacklist_chunk(con, data_root, chunk, reason)
        log.info(f'blacklisted {start + len(chunk)}/{len(tickers)} tickers')
    return total


def _blacklist_chunk(con: duckdb.DuckDBPyConnection, data_root: Path, tickers: list[str], reason: str) -> int:
    """Delete one chunk's rows from every ticker-keyed table and update ticker.csv/manual_blacklist.csv
    to match. Returns the number of rows deleted."""
    placeholders = ', '.join('?' for _ in tickers)
    total = 0
    for table in storage.TICKER_KEYED_TABLES:
        if storage.table_exists(con, table):
            total += int(con.execute(f'DELETE FROM {table} WHERE Ticker IN ({placeholders})', tickers).fetchone()[0])  # type: ignore[index]

    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    if ticker_csv.exists():
        existing_tracked = pd.read_csv(ticker_csv, keep_default_na=False)
        existing_tracked[~existing_tracked['ticker'].isin(tickers)].to_csv(ticker_csv, index=False)

    state_dir = data_root / 'yahoo' / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / MANUAL_BLACKLIST_FILE
    existing = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=_MANUAL_BLACKLIST_COLS)
    today = date.today().isoformat()
    new_rows = pd.DataFrame([{'ticker': ticker, 'reason': reason, 'blacklisted_on': today} for ticker in tickers])
    pd.concat([existing, new_rows], ignore_index=True).to_csv(path, index=False)
    return total


def unblacklist_tickers(data_root: Path, tickers: list[str]) -> int:
    """Remove `tickers` from manual_blacklist.csv so a future scan can offer them again.

    Only lifts the block on re-adding the ticker; data already deleted by blacklist_tickers stays
    deleted. A plain file write, not a DB write. Returns the number of entries removed.
    """
    path = data_root / 'yahoo' / 'state' / MANUAL_BLACKLIST_FILE
    if not path.exists():
        return 0
    existing = pd.read_csv(path, keep_default_na=False)
    remaining = existing[~existing['ticker'].isin(tickers)]
    removed = len(existing) - len(remaining)
    if removed:
        remaining.to_csv(path, index=False)
    return removed


def add_listings(con: duckdb.DuckDBPyConnection, data_root: Path, rows: list[dict]) -> tuple[int, int, int]:
    """Add reviewed candidates to ticker.csv in one step.

    A `reactivated_collision` row is renamed to its `new_ticker` key first, moving the old
    company's data aside, then added under the bare symbol. A collision row with a blank
    `new_ticker` is skipped entirely rather than added unrenamed, since that would merge two
    companies' data. Returns (tickers added, renames performed, rows skipped).
    """
    renamed = skipped = 0
    addable = []
    for row in rows:
        if row['classification'] == 'reactivated_collision':
            new_ticker = (row.get('new_ticker') or '').strip()
            if not new_ticker:
                skipped += 1
                continue
            old = row['ticker']
            for table in storage.TICKER_KEYED_TABLES:
                if storage.table_exists(con, table):
                    con.execute(f'UPDATE {table} SET Ticker = ? WHERE Ticker = ?', [new_ticker, old])
            renamed += 1
        addable.append(row)

    if addable:
        ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
        existing = pd.read_csv(ticker_csv, keep_default_na=False)
        new_rows = pd.DataFrame({'ticker': [r['ticker'] for r in addable], 'market': [r['market'] for r in addable]})
        combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates('ticker', keep='first')
        combined.to_csv(ticker_csv, index=False)
    return len(addable), renamed, skipped


def _add_new_tickers(candidates: list[Candidate], raw_dir: Path) -> None:
    """Append candidates classified 'new' to ticker.csv, so they're tracked without manual review."""
    additions = [c for c in candidates if c.classification == 'new']
    if not additions:
        return
    ticker_csv = raw_dir / 'ticker.csv'
    existing = pd.read_csv(ticker_csv, keep_default_na=False)
    new_rows = pd.DataFrame({'ticker': [c.ticker for c in additions], 'market': [c.market for c in additions]})
    combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates('ticker', keep='first')
    combined.to_csv(ticker_csv, index=False)


def scan(con: duckdb.DuckDBPyConnection, data_root: Path,
        cleaning_config: Path = Path('cleaning.toml')) -> UniverseScanResult:
    """Fetch live US common-stock listings, classify every untracked one, add the new ones to
    ticker.csv automatically, and save the ones still needing review to CSV.

    Compares against ticker.csv (currently tracked) and price_dead.csv/meta_dead.csv (tracked
    before, now dead). Resolves any dead ticker whose staged rename has completed before
    classifying, so a freshly cleared symbol is treated as brand new rather than stuck flagged as
    a collision. Skips anything in manual_blacklist.csv entirely. Candidates classified 'new' are
    added to ticker.csv directly and left out of the saved CSV, since there's nothing left to
    review; 'reactivated_safe'/'reactivated_collision' are saved for review. The returned
    `UniverseScanResult.candidates` still includes every classification, 'new' included.
    """
    raw_dir = data_root / 'yahoo' / 'raw'
    tracked = set(pd.read_csv(raw_dir / 'ticker.csv', keep_default_na=False)['ticker'])
    manual_blacklist = _read_manual_blacklist(data_root)
    dead_tickers, last_cik, last_company_name = _read_dead_ciks(con, data_root, tracked)
    dead_tickers = _resolve_completed_renames(con, data_root, cleaning_config, dead_tickers)

    live_listings = fetch_nasdaq_listings()
    live_listings = live_listings[~live_listings['ticker'].isin(manual_blacklist)]
    live_cik = fetch_sec_cik_map()
    candidates = classify(live_listings, tracked, dead_tickers, live_cik, last_cik, last_company_name)
    _add_new_tickers(candidates, raw_dir)

    known_tickers = set(tracked)
    if storage.table_exists(con, 'companies'):
        known_tickers |= set(con.execute('SELECT Ticker FROM companies').fetchdf()['Ticker'])
    resolved_candidates = []
    for candidate in candidates:
        if candidate.classification == 'reactivated_collision':
            suggestion = next_free_suffix(candidate.ticker, known_tickers)
            known_tickers.add(suggestion)
            candidate = replace(candidate, suggested_new_ticker=suggestion)
        resolved_candidates.append(candidate)

    reviewable = [c for c in resolved_candidates if c.classification != 'new']
    out_path = data_root / 'yahoo' / 'state' / CANDIDATES_FILE
    pd.DataFrame([vars(c) for c in reviewable]).to_csv(out_path, index=False)
    return UniverseScanResult(resolved_candidates, out_path)
