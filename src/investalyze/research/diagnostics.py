"""What a draft idea tells you before it is worth computing anything expensive.

Five readings, each able to kill an idea on its own:

- `coverage` says how many names actually have a value, per year. An idea that speaks about 53 names
  is not a factor, whatever its correlation with returns turns out to be.
- `distribution` catches a near-zero denominator producing values that will dominate any ranking.
- `extremes` names who is at each end right now, which is the fastest sanity check there is.
- `correlations` says whether the idea already exists in the library under another name.
- `rank_autocorrelation` says how fast the ranking churns, which is the turnover it will cost,
  known before a backtest rather than after one.
"""

import numpy as np
import pandas as pd

DATE, TICKER = 'Date', 'Ticker'


def coverage(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per year: how many names carry a value, out of how many were present."""
    years = pd.to_datetime(frame[DATE]).dt.year
    grouped = frame.groupby(years)[column]
    table = pd.DataFrame({'names': grouped.count(), 'universe': grouped.size()})
    table['share'] = (table['names'] / table['universe'] * 100.0).round(1)
    return table.reset_index(names='year')


def distribution(values: pd.Series) -> dict:
    """The shape of an idea's values, and how much of it is unusable."""
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {'count': 0, 'missing': int(values.isna().sum())}
    percentiles = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        'count': int(finite.size),
        'missing': int(values.size - finite.size),
        'mean': float(finite.mean()),
        'std': float(finite.std()),
        'p1': float(percentiles.loc[0.01]),
        'p25': float(percentiles.loc[0.25]),
        'median': float(percentiles.loc[0.5]),
        'p75': float(percentiles.loc[0.75]),
        'p99': float(percentiles.loc[0.99]),
    }


def extremes(frame: pd.DataFrame, column: str, *, n: int = 10, inputs: list[str] | None = None) -> dict:
    """The highest and lowest names on the most recent date, with the inputs behind them."""
    latest = frame[frame[DATE] == frame[DATE].max()]
    ranked = latest.dropna(subset=[column]).sort_values(column, ascending=False)
    shown = [TICKER, column] + [name for name in (inputs or []) if name in ranked.columns]
    return {'top': ranked.head(n)[shown].reset_index(drop=True),
            'bottom': ranked.tail(n)[shown].iloc[::-1].reset_index(drop=True)}


def _mean_cross_sectional_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    """Average of the per-date rank correlations between two columns."""
    per_date = []
    for _, group in frame.groupby(DATE):
        pair = group[[left, right]].dropna()
        if len(pair) > 2:
            per_date.append(pair[left].corr(pair[right], method='spearman'))
    if not per_date:
        return float('nan')
    return float(np.nanmean(per_date))


def correlations(frame: pd.DataFrame, column: str, others: list[str]) -> pd.Series:
    """Rank correlation of an idea to each other idea, strongest first."""
    scores = {}
    for other in others:
        if other != column and other in frame.columns:
            scores[other] = _mean_cross_sectional_corr(frame, column, other)
    return pd.Series(scores).sort_values(key=abs, ascending=False)


def rank_autocorrelation(frame: pd.DataFrame, column: str) -> float:
    """How much of the ranking survives to the next date, averaged over dates.

    Near 1 is a slow-moving idea that will trade rarely; near 0 turns the book over every rebalance
    and has to earn its costs back each time.
    """
    wide = frame.pivot_table(index=DATE, columns=TICKER, values=column)
    ranks = wide.rank(axis=1, pct=True)
    scores = []
    for earlier, later in zip(ranks.index[:-1], ranks.index[1:]):
        pair = pd.DataFrame({'a': ranks.loc[earlier], 'b': ranks.loc[later]}).dropna()
        if len(pair) > 2:
            scores.append(pair['a'].corr(pair['b']))
    if not scores:
        return float('nan')
    return float(np.nanmean(scores))
