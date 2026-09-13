"""Read side of the Factors page.

Every read opens its own short-lived read-only connection, matching the other pages, so a running job
holding the write lock is never blocked. Diagnostics run on a sample: the most liquid names on a
monthly grid, which keeps a run around a second.
"""

from pathlib import Path

import pandas as pd

from investalyze.ingest import storage
from investalyze.research import cache, compute, diagnostics, factors, fields, tearsheet

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'
FACTORS_DIR = ROOT / 'factors'
CACHE_DIR = DATA_ROOT / 'cache'

SAMPLE_NAMES = 400
SAMPLE_START, SAMPLE_END = '2016-01-31', '2024-12-31'
TEARSHEET_NAMES = 1500


def _connect():
    return storage.connect(DATA_ROOT, read_only=True)


def library() -> dict:
    """Every stored factor, keyed by name."""
    return factors.load(FACTORS_DIR)


def catalog() -> dict:
    """Every data field a factor expression may read."""
    con = _connect()
    try:
        return fields.catalog(con)
    finally:
        con.close()


def known_names() -> set:
    """Every name an expression may reference: data fields, reference columns and other factors."""
    return set(catalog()) | set(fields.IDENTITY) | set(library())


def field_rows(search: str | None = None) -> list[dict]:
    """The data fields as grid rows, filtered by a search term."""
    available = catalog()
    rows = []
    for alias in sorted(available):
        if search and search.lower() not in alias.lower():
            continue
        field = available[alias]
        rows.append({'name': alias, 'statement': field.statement, 'period': field.window.upper(),
                     'line_item': field.source})
    return rows


def liquid_tickers(con, limit: int) -> list[str]:
    """The names with the highest average dollar volume over the last year."""
    rows = con.execute("""
        SELECT Ticker FROM (
            SELECT Ticker, avg(C * V) AS dollar_volume
            FROM prices WHERE Date > (SELECT max(Date) - INTERVAL 1 YEAR FROM prices) AND C IS NOT NULL
            GROUP BY Ticker
        ) ORDER BY dollar_volume DESC NULLS LAST LIMIT ?
    """, [limit]).fetchall()
    return [row[0] for row in rows]


def _grid(start: str, end: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(start, end, freq='ME'))


def run_diagnostics(name: str, expression: str, *, higher_is_better: bool = True,
                    names: int = SAMPLE_NAMES) -> dict:
    """Compute a factor on the sample and return its diagnostics.

    The factor does not have to be saved: it is added to a copy of the library for this run only, so
    an unsaved expression can reference saved factors.
    """
    con = _connect()
    try:
        available = fields.catalog(con)
        candidate = factors.Factor(name=name, expression=expression, higher_is_better=higher_is_better)
        working = library() | {name: candidate}
        factors.validate_library(working, set(available) | set(fields.IDENTITY))

        tickers = liquid_tickers(con, names)
        frame = compute.compute(con, [name], working, as_of=_grid(SAMPLE_START, SAMPLE_END), tickers=tickers)
        if frame.empty or name not in frame.columns:
            return {'error': 'No factor values over the sample.'}

        others = [other for other in working if other != name and other in frame.columns]
        return {
            'coverage': diagnostics.coverage(frame, name),
            'distribution': diagnostics.distribution(frame[name]),
            'top_bottom': diagnostics.top_bottom(frame, name, n=8, inputs=sorted(candidate.dependencies)),
            'correlations': diagnostics.correlations(frame, name, others),
            'autocorrelation': diagnostics.rank_autocorrelation(frame, name),
            'sample': {'names': int(frame['Ticker'].nunique()), 'dates': int(frame['Date'].nunique())},
        }
    finally:
        con.close()


def run_tearsheet(name: str, *, horizon: int = tearsheet.DEFAULT_HORIZON, names: int = TEARSHEET_NAMES) -> dict:
    """IC analysis for one saved factor, using the cache."""
    con = _connect()
    try:
        working = library()
        tickers = liquid_tickers(con, names)
        frame = compute.compute(con, [name], working, as_of=_grid(SAMPLE_START, SAMPLE_END), tickers=tickers)
        if frame.empty or name not in frame.columns:
            return {'error': 'No factor values, so there is nothing to analyse.'}
        return tearsheet.tearsheet(con, name, working, frame, horizon=horizon, store=cache.Cache(CACHE_DIR))
    finally:
        con.close()


def save(factor: factors.Factor) -> None:
    """Write one factor to the library and clear cached results for its previous definition."""
    factors.save(factor, FACTORS_DIR)
    cache.Cache(CACHE_DIR).forget(factor.name)


def delete(name: str) -> None:
    """Remove a factor and its cached results."""
    (FACTORS_DIR / f'{name}.toml').unlink(missing_ok=True)
    cache.Cache(CACHE_DIR).forget(name)
