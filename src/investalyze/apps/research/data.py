"""Read side of the workbench.

Every read opens its own short-lived read-only connection, matching the other pages, so nothing here
can block a job holding the write lock. Drafting is deliberately sampled: a draft is computed over a
few hundred liquid names and a coarse date grid, because the answer to "is this idea sane" does not
change with the last decimal place and a sub-second reply is what makes editing a loop.
"""

from pathlib import Path

import pandas as pd

from investalyze.ingest import storage
from investalyze.research import cache, compute, diagnostics, fields, ideas, predict

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'
IDEAS_DIR = ROOT / 'ideas'
CACHE_DIR = DATA_ROOT / 'cache'

DRAFT_NAMES = 400
DRAFT_START, DRAFT_END = '2016-01-31', '2024-12-31'
FULL_NAMES = 1500


def _connect():
    return storage.connect(DATA_ROOT, read_only=True)


def library() -> dict:
    """Every stored idea, keyed by name."""
    return ideas.load(IDEAS_DIR)


def catalog() -> dict:
    """Every raw field an idea may read."""
    con = _connect()
    try:
        return fields.catalog(con)
    finally:
        con.close()


def known_names() -> set:
    """Everything an expression may reference: raw fields, identity columns and other ideas."""
    return set(catalog()) | set(fields.IDENTITY) | set(library())


def field_rows(search: str | None = None) -> list[dict]:
    """The palette, as grid rows, narrowed by a search term."""
    available = catalog()
    rows = []
    for alias in sorted(available):
        if search and search.lower() not in alias.lower():
            continue
        field = available[alias]
        rows.append({'name': alias, 'statement': field.statement, 'window': field.window,
                     'source': field.source})
    return rows


def sample_tickers(con, limit: int) -> list[str]:
    """The most traded names, so a sampled answer describes the universe anyone would trade."""
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


def preview(name: str, expression: str, *, higher_is_better: bool = True,
            names: int = DRAFT_NAMES) -> dict:
    """Compute a draft over a sample and report everything level one knows about it.

    The draft is never stored: it is held in a library alongside the saved ideas just long enough to
    be evaluated, which is what lets an unsaved idea reference a saved one.
    """
    con = _connect()
    try:
        available = fields.catalog(con)
        draft = ideas.Idea(name=name, expression=expression, higher_is_better=higher_is_better)
        working = library() | {name: draft}
        ideas.validate_library(working, set(available) | set(fields.IDENTITY))

        tickers = sample_tickers(con, names)
        frame = compute.compute(con, [name], working, as_of=_grid(DRAFT_START, DRAFT_END), tickers=tickers)
        if frame.empty or name not in frame.columns:
            return {'error': 'this idea produced no values over the sample'}

        others = [other for other in working if other != name and other in frame.columns]
        inputs = sorted(draft.dependencies)
        return {
            'frame': frame,
            'coverage': diagnostics.coverage(frame, name),
            'distribution': diagnostics.distribution(frame[name]),
            'extremes': diagnostics.extremes(frame, name, n=8, inputs=inputs),
            'correlations': diagnostics.correlations(frame, name, others),
            'autocorrelation': diagnostics.rank_autocorrelation(frame, name),
            'sample': {'names': int(frame['Ticker'].nunique()), 'dates': int(frame['Date'].nunique())},
        }
    finally:
        con.close()


def verdict(name: str, *, horizon: int = predict.DEFAULT_HORIZON, names: int = FULL_NAMES) -> dict:
    """Level two for one saved idea, cached so asking again is cheap."""
    con = _connect()
    try:
        working = library()
        tickers = sample_tickers(con, names)
        frame = compute.compute(con, [name], working, as_of=_grid(DRAFT_START, DRAFT_END), tickers=tickers)
        if frame.empty or name not in frame.columns:
            return {'error': 'this idea produced no values, so there is nothing to score'}
        store = cache.Cache(CACHE_DIR)
        return predict.verdict(con, name, working, frame, horizon=horizon, store=store)
    finally:
        con.close()


def save(idea: ideas.Idea) -> None:
    """Write one idea to the library, and drop any answer the previous definition left behind."""
    ideas.save(idea, IDEAS_DIR)
    cache.Cache(CACHE_DIR).forget(idea.name)


def delete(name: str) -> None:
    """Remove an idea and everything computed from it."""
    path = IDEAS_DIR / f'{name}.toml'
    path.unlink(missing_ok=True)
    cache.Cache(CACHE_DIR).forget(name)
