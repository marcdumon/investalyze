"""Discover live US common-stock listings missing from ticker.csv, and tell apart a genuinely
new ticker from a delisted symbol reused by a different company.

Two free, unauthenticated sources: NASDAQ Trader's symbol directory (what's listed today, on
which exchange) and SEC's company_tickers.json (ticker -> CIK, used only to check identity on a
reactivated symbol). Network fetching stays in `fetch_nasdaq_listings`/`fetch_sec_cik_map`; the
parsing and classification functions are pure and take plain text/dicts, so they're testable
without HTTP calls.
"""

import io
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pandas as pd
import requests

from investalyze.cleaning import registry as cleaning_registry
from investalyze.ingest import storage
from investalyze.ingest.providers.yahoo import blacklist

NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
OTHER_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
HEADERS = {'User-Agent': 'investalyze ingest (7793950+marcdumon@users.noreply.github.com)'}

# security-name keywords that signal a warrant/right/preferred/unit rather than common stock
_DROP_KEYWORDS = ('warrant', 'right', 'preferred', ' unit')
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
    lowered = security_name.lower()
    return not any(keyword in lowered for keyword in _DROP_KEYWORDS)


def _parse_nasdaqlisted(text: str) -> pd.DataFrame:
    """NASDAQ-listed common stocks from nasdaqlisted.txt: columns ticker, market, company_name."""
    frame = pd.read_csv(io.StringIO(_strip_footer(text)), sep='|', dtype=str)
    frame = frame[(frame['Test Issue'] == 'N') & (frame['ETF'] == 'N')]
    frame = frame[frame['Security Name'].apply(_is_common_stock)]
    return pd.DataFrame({'ticker': frame['Symbol'], 'market': 'nasdaq', 'company_name': frame['Security Name']})


def _parse_otherlisted(text: str) -> pd.DataFrame:
    """NYSE / NYSE American common stocks from otherlisted.txt: columns ticker, market, company_name.

    Other exchanges in the file (Arca, Cboe BZX, IEX, ...) fall outside today's tracked markets and
    are dropped, matching ticker.csv's existing nasdaq/nyse/nysemkt scope.
    """
    frame = pd.read_csv(io.StringIO(_strip_footer(text)), sep='|', dtype=str)
    frame = frame[(frame['Test Issue'] == 'N') & (frame['ETF'] == 'N')]
    frame = frame[frame['Security Name'].apply(_is_common_stock)]
    frame = frame[frame['Exchange'].isin(_EXCHANGE_MARKET)]
    return pd.DataFrame({'ticker': frame['ACT Symbol'], 'market': frame['Exchange'].map(_EXCHANGE_MARKET),
                         'company_name': frame['Security Name']})


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


def classify(listings: pd.DataFrame, tracked: set[str], dead_tickers: set[str],
            live_cik: dict[str, int], last_cik: dict[str, int | None]) -> list[Candidate]:
    """Classify every live listing not already tracked as new, reactivated_safe, or reactivated_collision.

    A ticker not in `dead_tickers` is brand new. A ticker in `dead_tickers` (tracked before, now
    dead) is reactivated_safe only when its live SEC CIK matches the CIK on file from the last time
    it was tracked. A mismatch or missing CIK is reactivated_collision. An unknown CIK is treated
    as a collision.
    """
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
        candidates.append(Candidate(ticker, row['market'], row['company_name'], classification))
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


def _read_dead_ciks(con: duckdb.DuckDBPyConnection, data_root: Path) -> tuple[set[str], dict[str, int | None]]:
    """Every ticker ever tracked but now dead, with the CIK on file from its last time being tracked."""
    state_dir = data_root / 'yahoo' / 'state'
    dead_tickers = set(blacklist.read_dead(state_dir / 'price_dead.csv')['ticker']) | \
        set(blacklist.read_dead(state_dir / 'meta_dead.csv')['ticker'])
    if not dead_tickers or not storage.table_exists(con, 'companies'):
        return dead_tickers, {}
    placeholders = ', '.join('?' for _ in dead_tickers)
    rows = con.execute(f'SELECT Ticker, CIK FROM companies WHERE Ticker IN ({placeholders})', list(dead_tickers)).fetchall()
    last_cik = {ticker: (int(cik) if cik is not None else None) for ticker, cik in rows}
    return dead_tickers, last_cik


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


def scan(con: duckdb.DuckDBPyConnection, data_root: Path,
        cleaning_config: Path = Path('cleaning.toml')) -> UniverseScanResult:
    """Fetch live US common-stock listings, classify every untracked one, and save the result to CSV.

    Compares against ticker.csv (currently tracked) and price_dead.csv/meta_dead.csv (tracked
    before, now dead). Resolves any dead ticker whose staged rename has completed before
    classifying, so a freshly cleared symbol is treated as brand new rather than stuck flagged as
    a collision.
    """
    raw_dir = data_root / 'yahoo' / 'raw'
    tracked = set(pd.read_csv(raw_dir / 'ticker.csv')['ticker'])
    dead_tickers, last_cik = _read_dead_ciks(con, data_root)
    dead_tickers = _resolve_completed_renames(con, data_root, cleaning_config, dead_tickers)

    live_listings = fetch_nasdaq_listings()
    live_cik = fetch_sec_cik_map()
    candidates = classify(live_listings, tracked, dead_tickers, live_cik, last_cik)

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

    out_path = data_root / 'yahoo' / 'state' / CANDIDATES_FILE
    pd.DataFrame([vars(c) for c in resolved_candidates]).to_csv(out_path, index=False)
    return UniverseScanResult(resolved_candidates, out_path)
