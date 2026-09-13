"""Factor tear sheet: IC analysis, IC decay and quantile returns.

The information coefficient (IC) per date is the rank correlation between the factor values on that
date and the forward returns that followed. The mean IC, its t-statistic, IC decay and sub-period
readings are all summaries of that series, so the per-date series is what gets cached, once per
horizon. IC decay then reuses the cached series for each horizon.

Two numbers are reported beside every result. `names_median` and `names_min` give the size of the
cross-section each IC was computed on. `expected_false_positives` gives how many factors in a library
of this size would pass p<0.05 by chance (multiple testing).
"""

import numpy as np
import pandas as pd

from investalyze.analysis import panels
from investalyze.backtest import evaluation
from investalyze.research import cache, factors

HORIZONS = (5, 21, 63, 126)
DEFAULT_HORIZON = 21
DEFAULT_QUANTILES = 5


def _closes(con, tickers: list[str], first, last, horizon: int) -> pd.DataFrame:
    """Adjusted closes over the factor's dates, extended to cover the forward return window."""
    padded = pd.Timestamp(last) + pd.Timedelta(days=2 * horizon + 10)
    calendar = panels.trading_calendar(con, start=pd.Timestamp(first).date().isoformat(),
                                       end=padded.date().isoformat())
    return panels.close_panel(con, tickers, calendar)


def ic_series(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Per date: the rank IC against the `horizon`-day forward return, and the number of names used."""
    scores = evaluation.factor_scores(frame, name)
    tickers = sorted(scores.columns)
    closes = _closes(con, tickers, scores.index.min(), scores.index.max(), horizon)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    ic = evaluation.rank_ic(scores, forwards)

    shared = scores.index.intersection(forwards.index)
    paired = scores.loc[shared].notna() & forwards.loc[shared, scores.columns.intersection(forwards.columns)].notna()
    return pd.DataFrame({'Date': ic.index, 'ic': ic.to_numpy(),
                         'names': paired.sum(axis=1).reindex(ic.index).to_numpy()})


def ic_summary(ic_frame: pd.DataFrame, *, higher_is_better: bool = True) -> dict:
    """Mean IC, t-statistic, hit rate and cross-section size, with the IC signed by factor direction."""
    summary = evaluation.ic_summary(pd.Series(ic_frame['ic'].to_numpy(), index=ic_frame['Date']))
    counted = ic_frame.loc[ic_frame['ic'].notna(), 'names']
    direction = 1.0 if higher_is_better else -1.0
    summary['oriented_ic'] = summary['mean'] * direction
    summary['oriented_t'] = summary['t_stat'] * direction
    summary['names_median'] = float(counted.median()) if len(counted) else np.nan
    summary['names_min'] = float(counted.min()) if len(counted) else np.nan
    return summary


def ic_decay(con, name: str, frame: pd.DataFrame, *, horizons: tuple = HORIZONS,
             store: cache.Cache | None = None, fingerprint: str = '') -> pd.DataFrame:
    """Mean IC and t-statistic at each forward horizon."""
    rows = []
    for horizon in horizons:
        series = cached_ic(con, name, frame, horizon=horizon, store=store, fingerprint=fingerprint)
        summary = ic_summary(series)
        rows.append({'horizon': horizon, 'mean_ic': summary['mean'], 't_stat': summary['t_stat'],
                     'n_periods': summary['n_periods'], 'names_median': summary['names_median']})
    return pd.DataFrame(rows)


def quantile_returns(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
                     n_quantiles: int = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Mean and median forward return per factor quantile, with the average number of names in each."""
    scores = evaluation.factor_scores(frame, name)
    closes = _closes(con, sorted(scores.columns), scores.index.min(), scores.index.max(), horizon)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    table = evaluation.quantile_frame(scores, forwards, n_quantiles=n_quantiles)
    return table.groupby('Quantile', as_index=False).agg(
        mean_return=('Return', 'mean'), median_return=('MedianReturn', 'mean'), names=('NNames', 'mean'))


def is_monotonic(quantile_table: pd.DataFrame, *, higher_is_better: bool = True) -> bool:
    """Whether mean return rises from the bottom quantile to the top, in the factor's direction."""
    ordered = quantile_table.sort_values('Quantile')['mean_return'].to_numpy()
    if len(ordered) < 2 or np.isnan(ordered).any():
        return False
    steps = np.diff(ordered)
    return bool((steps > 0).all()) if higher_is_better else bool((steps < 0).all())


def expected_false_positives(tested: int, alpha: float = 0.05) -> float:
    """How many of `tested` factors would pass significance level `alpha` by chance."""
    return max(tested, 0) * alpha


def ic_params(frame: pd.DataFrame, horizon: int) -> str:
    """The cache key for an IC series: horizon, ticker universe and date range.

    The universe and dates are part of the key because the same factor and horizon give a different
    IC series on a different set of names or dates.
    """
    dates = pd.to_datetime(frame['Date'])
    return cache.params_hash(result='ic', horizon=horizon, tickers=sorted(frame['Ticker'].unique()),
                             start=dates.min().date().isoformat(), end=dates.max().date().isoformat())


def cached_ic(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
              store: cache.Cache | None = None, fingerprint: str = '') -> pd.DataFrame:
    """The per-date IC series, read from the cache when current and computed otherwise."""
    params = ic_params(frame, horizon)
    if store is not None and fingerprint:
        stored = store.get(cache.IC, name, params, fingerprint)
        if stored is not None:
            return stored
    computed = ic_series(con, name, frame, horizon=horizon)
    if store is not None and fingerprint:
        store.put(cache.IC, name, params, fingerprint, computed,
                  meta={'horizon': horizon, 'dates': int(computed['ic'].notna().sum())})
    return computed


def tearsheet(con, name: str, library: dict, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
              store: cache.Cache | None = None) -> dict:
    """The full tear sheet for one factor: IC summary, IC decay, quantile returns, multiple testing."""
    fingerprint = factors.fingerprint(name, library) if name in library else ''
    higher = library[name].higher_is_better if name in library else True
    series = cached_ic(con, name, frame, horizon=horizon, store=store, fingerprint=fingerprint)
    quantiles = quantile_returns(con, name, frame, horizon=horizon)
    return {
        'factor': name,
        'horizon': horizon,
        'higher_is_better': higher,
        'summary': ic_summary(series, higher_is_better=higher),
        'decay': ic_decay(con, name, frame, store=store, fingerprint=fingerprint),
        'quantiles': quantiles,
        'monotonic': is_monotonic(quantiles, higher_is_better=higher),
        'tested': len(library),
        'expected_false_positives': expected_false_positives(len(library)),
    }
