"""Discover live US common-stock listings missing from ticker.csv, and tell apart a genuinely
new ticker from a delisted symbol reused by a different company.

Two free, unauthenticated sources: NASDAQ Trader's symbol directory (what's listed today, on
which exchange) and SEC's company_tickers.json (ticker -> CIK, used only to check identity on a
reactivated symbol). Network fetching stays in `fetch_nasdaq_listings`/`fetch_sec_cik_map`; the
parsing and classification functions are pure and take plain text/dicts, so they're testable
without HTTP calls.
"""

import io

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
