"""Level two: does an idea predict what happens next?

The primitive is the information coefficient per date, which is the rank correlation between how an
idea scored the names on a date and the return that followed. Everything else is a view over that
series: its mean and t-statistic, how the edge decays as the horizon lengthens, how it reads in one
year rather than another. So the per-date series is what gets cached, once per horizon, and decay
costs nothing beyond having asked about each horizon already.

Two readings are carried alongside every number because leaving them off is how a result gets
misread. `names` says how large the cross-section actually was: an idea scoring 53 names is not
measuring the same thing as one scoring 2,000, whatever its t-statistic says. `expected_false_
positives` says how many significant-looking results a library this size produces by luck alone.
"""

import numpy as np
import pandas as pd

from investalyze.analysis import panels
from investalyze.backtest import evaluation
from investalyze.research import cache, ideas

HORIZONS = (5, 21, 63, 126)
DEFAULT_HORIZON = 21
DEFAULT_QUANTILES = 5


def _closes(con, tickers: list[str], first, last, horizon: int) -> pd.DataFrame:
    """Adjusted closes over the sessions an idea's dates need, plus room for the forward window."""
    padded = pd.Timestamp(last) + pd.Timedelta(days=2 * horizon + 10)
    calendar = panels.trading_calendar(con, start=pd.Timestamp(first).date().isoformat(),
                                       end=padded.date().isoformat())
    return panels.close_panel(con, tickers, calendar)


def ic_series(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Per date: the rank correlation with the following `horizon` sessions, and how many names it used."""
    scores = evaluation.factor_scores(frame, name)
    tickers = sorted(scores.columns)
    closes = _closes(con, tickers, scores.index.min(), scores.index.max(), horizon)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    ic = evaluation.rank_ic(scores, forwards)

    shared = scores.index.intersection(forwards.index)
    paired = scores.loc[shared].notna() & forwards.loc[shared, scores.columns.intersection(forwards.columns)].notna()
    return pd.DataFrame({'Date': ic.index, 'ic': ic.to_numpy(),
                         'names': paired.sum(axis=1).reindex(ic.index).to_numpy()})


def summarise(ic_frame: pd.DataFrame, *, higher_is_better: bool = True) -> dict:
    """The headline reading of an IC series, oriented so a positive number always means it worked."""
    summary = evaluation.ic_summary(pd.Series(ic_frame['ic'].to_numpy(), index=ic_frame['Date']))
    counted = ic_frame.loc[ic_frame['ic'].notna(), 'names']
    direction = 1.0 if higher_is_better else -1.0
    summary['oriented_ic'] = summary['mean'] * direction
    summary['oriented_t'] = summary['t_stat'] * direction
    summary['names_median'] = float(counted.median()) if len(counted) else np.nan
    summary['names_min'] = float(counted.min()) if len(counted) else np.nan
    return summary


def decay(con, name: str, frame: pd.DataFrame, *, horizons: tuple = HORIZONS,
          store: cache.Cache | None = None, fingerprint: str = '') -> pd.DataFrame:
    """Mean IC at each horizon, which is the argument for how often to rebalance."""
    rows = []
    for horizon in horizons:
        series = answer(con, name, frame, horizon=horizon, store=store, fingerprint=fingerprint)
        summary = summarise(series)
        rows.append({'horizon': horizon, 'mean_ic': summary['mean'], 't_stat': summary['t_stat'],
                     'n_periods': summary['n_periods'], 'names_median': summary['names_median']})
    return pd.DataFrame(rows)


def quantiles(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
              n_quantiles: int = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Mean and median forward return of each score bucket, with how many names each held."""
    scores = evaluation.factor_scores(frame, name)
    closes = _closes(con, sorted(scores.columns), scores.index.min(), scores.index.max(), horizon)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    table = evaluation.quantile_frame(scores, forwards, n_quantiles=n_quantiles)
    return table.groupby('Quantile', as_index=False).agg(
        mean_return=('Return', 'mean'), median_return=('MedianReturn', 'mean'), names=('NNames', 'mean'))


def is_monotonic(bucket_table: pd.DataFrame, *, higher_is_better: bool = True) -> bool:
    """Whether returns climb from the lowest bucket to the highest, which is the shape a real edge has."""
    ordered = bucket_table.sort_values('Quantile')['mean_return'].to_numpy()
    if len(ordered) < 2 or np.isnan(ordered).any():
        return False
    steps = np.diff(ordered)
    return bool((steps > 0).all()) if higher_is_better else bool((steps < 0).all())


def expected_false_positives(tested: int, alpha: float = 0.05) -> float:
    """How many ideas in a library this size look significant by luck alone.

    Testing enough ideas manufactures winners. Reading a t-statistic without this number beside it
    is how the twentieth idea gets mistaken for the first.
    """
    return max(tested, 0) * alpha


def answer(con, name: str, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
           store: cache.Cache | None = None, fingerprint: str = '') -> pd.DataFrame:
    """The per-date IC series, from cache when it is current and recomputed when it is not."""
    question = cache.signature(question='ic', horizon=horizon)
    if store is not None and fingerprint:
        stored = store.get(cache.PREDICT, name, question, fingerprint)
        if stored is not None:
            return stored
    computed = ic_series(con, name, frame, horizon=horizon)
    if store is not None and fingerprint:
        store.put(cache.PREDICT, name, question, fingerprint, computed,
                  meta={'horizon': horizon, 'dates': int(computed['ic'].notna().sum())})
    return computed


def verdict(con, name: str, library: dict, frame: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON,
            store: cache.Cache | None = None) -> dict:
    """Everything level two knows about one idea, in the order a reader should weigh it."""
    fingerprint = ideas.fingerprint(name, library) if name in library else ''
    higher = library[name].higher_is_better if name in library else True
    series = answer(con, name, frame, horizon=horizon, store=store, fingerprint=fingerprint)
    buckets = quantiles(con, name, frame, horizon=horizon)
    return {
        'idea': name,
        'horizon': horizon,
        'summary': summarise(series, higher_is_better=higher),
        'decay': decay(con, name, frame, store=store, fingerprint=fingerprint),
        'quantiles': buckets,
        'monotonic': is_monotonic(buckets, higher_is_better=higher),
        'tested': len(library),
        'expected_false_positives': expected_false_positives(len(library)),
    }
