"""Discover live US common-stock listings missing from ticker.csv, and tell apart a genuinely
new ticker from a delisted symbol reused by a different company.

Two free, unauthenticated sources: NASDAQ Trader's symbol directory (what's listed today, on
which exchange) and SEC's company_tickers.json (ticker -> CIK, used only to check identity on a
reactivated symbol). Network fetching stays in `fetch_nasdaq_listings`/`fetch_sec_cik_map`; the
parsing and classification functions are pure and take plain text/dicts, so they're testable
without HTTP calls.
"""

import io
from dataclasses import dataclass

import pandas as pd
import requests

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
