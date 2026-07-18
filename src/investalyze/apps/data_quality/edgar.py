"""On-demand SEC EDGAR filing lookup: resolves the exact document URL for one (CIK, report date, period).

Hits SEC's public submissions JSON API (data.sec.gov) and disk-caches the result, since a CIK's filing
history rarely changes and SEC rate-limits to 10 req/sec. The filing type is inferred from the fiscal
period ('FY'/'Q4' -> 10-K, else 10-Q, since many filers only report Q4 figures inside the annual report),
and the closest reportDate within a tolerance window is picked, since SimFin's Report Date can be a
normalised month-end while SEC's is the true period end.
"""

import json
import logging
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests

log = logging.getLogger('investalyze.apps.data_quality.edgar')

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / 'data' / 'sec' / 'submissions'
HEADERS = {'User-Agent': 'investalyze data-quality (dumon.marc@gmail.com)'}
EDGAR_DOC_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}'
CACHE_TTL_SECONDS = 86400 * 7


def _fetch_chunk(filename: str) -> dict:
    """Fetch one older-filings chunk listed in `filings.files[]`, disk-cached for `CACHE_TTL_SECONDS`.

    SEC partitions per-CIK filings: the master JSON's `filings.recent` holds the ~1000 most-recent
    filings, with everything older referenced in `filings.files[]` as chunk file names. Each chunk has
    the same parallel-arrays schema as `recent`.
    """
    cache = CACHE_DIR / filename
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS:
        return json.loads(cache.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f'GET https://data.sec.gov/submissions/{filename}')
    response = requests.get(f'https://data.sec.gov/submissions/{filename}', headers=HEADERS, timeout=10)
    response.raise_for_status()
    cache.write_text(response.text)
    time.sleep(0.11)
    return response.json()


def _fetch_submissions(cik: int) -> dict:
    """Fetch the full-history SEC submissions JSON for a CIK, disk-cached for `CACHE_TTL_SECONDS`.

    Hits `https://data.sec.gov/submissions/CIK{cik:010d}.json` for the master JSON, then merges every
    older chunk from `filings.files[]` into `filings.recent`, so the returned `filings.recent` spans the
    full filing history for the company.
    """
    cik_str = f'{cik:010d}'
    cache = CACHE_DIR / f'{cik_str}.json'
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS:
        data = json.loads(cache.read_text())
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f'GET https://data.sec.gov/submissions/CIK{cik_str}.json')
        response = requests.get(f'https://data.sec.gov/submissions/CIK{cik_str}.json', headers=HEADERS, timeout=10)
        response.raise_for_status()
        cache.write_text(response.text)
        time.sleep(0.11)
        data = response.json()

    filings = data.get('filings', {})
    recent = {key: list(value) for key, value in filings.get('recent', {}).items()}
    for file_info in filings.get('files', []):
        chunk = _fetch_chunk(file_info['name'])
        for key in recent:
            if key in chunk:
                recent[key].extend(chunk[key])
    data['filings']['recent'] = recent
    return data


def _parse_date(text: str) -> date | None:
    """Parse an ISO date string to `date`, or None on bad input."""
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _search_submissions(cik: int, target: date, forms: set[str], tol_days: int) -> tuple[str, str] | None:
    """Scan one CIK's filings for the closest match; return (accessionNumber, primaryDocument) or None.

    Matches on `reportDate` first. Some older, pre-XBRL filings carry a `reportDate` that SEC defaulted
    to the filing date instead of the true period end, so those never fall within `tol_days`; as a
    fallback, the filing whose `filingDate` is the earliest one on or after `target` (within a plausible
    reporting-lag window) is used instead.
    """
    try:
        data = _fetch_submissions(cik)
    except (requests.RequestException, ValueError) as error:
        log.debug(f'EDGAR fetch failed for CIK {cik}: {type(error).__name__}: {error}')
        return None
    recent = data.get('filings', {}).get('recent', {})
    rows = list(zip(recent.get('form', []), recent.get('accessionNumber', []), recent.get('primaryDocument', []),
                    recent.get('reportDate', []), recent.get('filingDate', []), strict=False))

    best, best_delta = None, tol_days + 1
    for form, accession, primary_doc, report_date, _ in rows:
        if form not in forms or not report_date or not primary_doc:
            continue
        parsed = _parse_date(report_date)
        if parsed is None:
            continue
        delta = abs((parsed - target).days)
        if delta <= tol_days and delta < best_delta:
            best, best_delta = (accession, primary_doc), delta
    if best is not None:
        return best

    lag_tol_days = 150 if any(form.startswith('10-K') for form in forms) else 60
    best, best_delta = None, lag_tol_days + 1
    for form, accession, primary_doc, _, filing_date in rows:
        if form not in forms or not filing_date or not primary_doc:
            continue
        parsed = _parse_date(filing_date)
        if parsed is None or parsed < target:
            continue
        delta = (parsed - target).days
        if delta <= lag_tol_days and delta < best_delta:
            best, best_delta = (accession, primary_doc), delta
    return best


@lru_cache(maxsize=10000)
def restated_filing_url(cik: int | None, restated_date: str | None, tol_days: int = 2) -> str | None:
    """Direct EDGAR document URL for the periodic filing filed on `restated_date`, or None if no match.

    SimFin's `Restated Date` on a restated fundamentals row is the filing date of the later 10-K/10-Q
    that reported the corrected figures for that period, so this matches on `filingDate` directly
    (nearest within `tol_days`) rather than the nearest-`reportDate` search `filing_url` does for
    as-reported rows. `restated_date` must be 'YYYY-MM-DD'.
    """
    target = _parse_date(restated_date) if restated_date else None
    if not cik or target is None:
        return None
    forms = {'10-K', '10-K405', '10-KSB', '10-Q', '10-QSB'}
    try:
        data = _fetch_submissions(int(cik))
    except (requests.RequestException, ValueError) as error:
        log.debug(f'EDGAR fetch failed for CIK {cik}: {type(error).__name__}: {error}')
        return None
    recent = data.get('filings', {}).get('recent', {})
    rows = zip(recent.get('form', []), recent.get('accessionNumber', []), recent.get('primaryDocument', []),
              recent.get('filingDate', []), strict=False)
    best, best_delta = None, tol_days + 1
    for form, accession, primary_doc, filing_date in rows:
        if form not in forms or not filing_date or not primary_doc:
            continue
        parsed = _parse_date(filing_date)
        if parsed is None:
            continue
        delta = abs((parsed - target).days)
        if delta <= tol_days and delta < best_delta:
            best, best_delta = (accession, primary_doc), delta
    if best is None:
        return None
    accession, primary_doc = best
    return EDGAR_DOC_URL.format(cik=int(cik), acc_no_dashes=accession.replace('-', ''), primary_doc=primary_doc)


@lru_cache(maxsize=10000)
def filing_url(cik: int | None, report_date: str | None, fiscal_period: str, tol_days: int = 10) -> str | None:
    """Direct EDGAR document URL for one (cik, report_date, fiscal_period), or None if no filing matches.

    Picks the form by fiscal period ('FY'/'Q4' -> 10-K, else 10-Q) and scans the SEC submissions JSON for
    the filing whose `reportDate` is closest to `report_date`, within `tol_days`. `report_date` must be
    'YYYY-MM-DD'. Memoised per (cik, report_date, fiscal_period, tol_days) for the process lifetime; the
    underlying submissions JSON is disk-cached separately by `_fetch_submissions`.
    """
    target = _parse_date(report_date) if report_date else None
    if not cik or target is None:
        return None
    forms = {'10-K', '10-K405', '10-KSB'} if fiscal_period in ('FY', 'Q4') else {'10-Q', '10-QSB'}
    match = _search_submissions(int(cik), target, forms, tol_days)
    if match is None:
        return None
    accession, primary_doc = match
    return EDGAR_DOC_URL.format(cik=int(cik), acc_no_dashes=accession.replace('-', ''), primary_doc=primary_doc)
