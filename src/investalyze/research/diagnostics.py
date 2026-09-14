"""Factor diagnostics, run on a sample before IC analysis.

- `coverage`: names with a value per year, against the names in the universe. A factor covering 53
  names is not usable, whatever its IC.
- `distribution`: summary statistics and percentiles, which expose outliers from near-zero
  denominators.
- `top_bottom`: the highest and lowest names on the latest date, with their inputs.
- `correlations`: rank correlation with the other factors in the library, which exposes redundancy.
- `rank_autocorrelation`: persistence of the factor ranking between dates, a proxy for turnover.
"""

import numpy as np
import pandas as pd

DATE, TICKER = 'Date', 'Ticker'


def coverage(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per year: average names per date with a value, against average names per date in the universe."""
    if 'in_universe' in frame.columns:
        frame = frame[frame['in_universe']]
    per_date = frame.groupby(DATE)[column].agg(['count', 'size'])
    years = pd.to_datetime(per_date.index).year
    table = pd.DataFrame({'names': per_date['count'].groupby(years).mean(),
                          'universe': per_date['size'].groupby(years).mean()})
    table['share'] = (table['names'] / table['universe'] * 100.0).round(1)
    return table.reset_index(names='year')


def distribution(values: pd.Series) -> dict:
    """Summary statistics of factor values, counting infinite values as missing."""
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


def top_bottom(frame: pd.DataFrame, column: str, *, n: int = 10, inputs: list[str] | None = None) -> dict:
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
    """Mean cross-sectional rank correlation with each other factor, strongest first."""
    scores = {}
    for other in others:
        if other != column and other in frame.columns:
            scores[other] = _mean_cross_sectional_corr(frame, column, other)
    return pd.Series(scores).sort_values(key=abs, ascending=False)


def rank_autocorrelation(frame: pd.DataFrame, column: str) -> float:
    """Mean rank autocorrelation between consecutive dates.

    Near 1 means low turnover; near 0 means the portfolio turns over at every rebalance.
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
