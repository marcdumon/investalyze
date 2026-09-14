"""Read side of the Factors page.

Every read opens its own short-lived read-only connection, matching the other pages, so a running job
holding the write lock is never blocked. The universe on each rebalance date is the most liquid names
by trailing dollar volume known on that date. Diagnostics use a smaller universe than IC analysis,
which keeps a diagnostics run to a few seconds.
"""

import duckdb
import pandas as pd

from investalyze import paths
from investalyze.analysis import factors_pit
from investalyze.ingest import storage
from investalyze.research import cache, compute, diagnostics, factors, fields, tearsheet, universe

SAMPLE_START, SAMPLE_END = '2016-01-31', '2024-12-31'
HOLDOUT_START = pd.Timestamp('2022-01-01')
DIAGNOSTICS_SIZE = 400
RESEARCH_SIZE = 1000


def _store() -> cache.Cache:
    """The results cache under the data directory."""
    return cache.Cache(paths.CACHE_DIR)


def library() -> dict:
    """Every stored factor, keyed by name."""
    return factors.load(paths.FACTORS_DIR)


def catalog() -> dict:
    """Every data field a factor expression may read."""
    with storage.connect(read_only=True) as con:
        return fields.catalog(con)


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
                     'description': field.source})
    return rows


def summaries() -> dict:
    """Recorded results for the current definition of every factor, keyed by factor name."""
    working = library()
    store = _store()
    return {name: store.get_summary(name, factors.fingerprint(name, working)) for name in working}


def _panel(con: duckdb.DuckDBPyConnection, names: list[str], working: dict, size: int) -> pd.DataFrame:
    """The factor panel over the research period for a point-in-time universe of `size` names."""
    grid = factors_pit.rebalance_dates(con, start=SAMPLE_START, end=SAMPLE_END, freq='M')
    members = universe.liquid_universe(con, grid, size=size)
    return compute.compute(con, names, working, as_of=grid, members=members)


def _is_saved(name: str, expression: str, working: dict) -> bool:
    """True when `name` is in the library with exactly this expression."""
    return name in working and working[name].expression == expression


def run_diagnostics(name: str, expression: str, *, higher_is_better: bool = True) -> dict:
    """Compute a factor over the diagnostics universe and return its diagnostics.

    The factor does not have to be saved: it is added to a copy of the library for this run only, so
    an unsaved expression can reference saved factors.
    """
    with storage.connect(read_only=True) as con:
        available = fields.catalog(con)
        saved = library()
        candidate = factors.Factor(name=name, expression=expression, higher_is_better=higher_is_better)
        working = saved | {name: candidate}
        factors.validate_library(working, set(available) | set(fields.IDENTITY))

        # every library factor is computed alongside, so correlation covers the whole library
        frame = _panel(con, sorted(working), working, DIAGNOSTICS_SIZE)
        if frame.empty or name not in frame.columns:
            return {'error': 'No factor values over the sample.'}

        others = [other for other in working if other != name and other in frame.columns]
        coverage = diagnostics.coverage(frame, name)
        result = {
            'coverage': coverage,
            'distribution': diagnostics.distribution(frame[name]),
            'top_bottom': diagnostics.top_bottom(frame, name, n=8, inputs=sorted(candidate.dependencies)),
            'correlations': diagnostics.correlations(frame, name, others),
            'autocorrelation': diagnostics.rank_autocorrelation(frame, name),
            'sample': {'size': DIAGNOSTICS_SIZE, 'dates': int(frame['Date'].nunique())},
        }
        if _is_saved(name, expression, saved) and not coverage.empty:
            _store().put_summary(name, factors.fingerprint(name, saved), {'coverage': float(coverage.iloc[-1]['share'])})
        return result


def run_tearsheet(name: str, *, horizon: int = tearsheet.DEFAULT_HORIZON) -> dict:
    """IC analysis for one saved factor over the research universe, using the cache."""
    with storage.connect(read_only=True) as con:
        working = library()
        frame = _panel(con, [name], working, RESEARCH_SIZE)
        if frame.empty or name not in frame.columns:
            return {'error': 'No factor values, so there is nothing to analyse.'}
        store = _store()
        result = tearsheet.tearsheet(con, name, working, frame, horizon=horizon, store=store, holdout_start=HOLDOUT_START)
        split = result['split'].set_index('period') if not result['split'].empty else pd.DataFrame()
        store.put_summary(name, factors.fingerprint(name, working), {
            'ic': result['summary']['oriented_ic'], 't_stat': result['summary']['oriented_t'],
            'oos_t_stat': float(split.loc['Out-of-sample', 't_stat']) if 'Out-of-sample' in split.index else None,
        })
        return result


def save(factor: factors.Factor) -> None:
    """Write one factor to the library.

    Cached results are keyed by fingerprint, so a changed expression is never served an old result,
    and a change to status or rationale keeps the results it already has.
    """
    factors.save(factor, paths.FACTORS_DIR)


def delete(name: str) -> None:
    """Remove a factor and its cached results."""
    (paths.FACTORS_DIR / f'{name}.toml').unlink(missing_ok=True)
    _store().forget(name)
