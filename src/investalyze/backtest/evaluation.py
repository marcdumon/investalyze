"""Judging a signal before any portfolio exists: rank correlation with forward returns, and the
spread between the buckets a factor sorts names into.

Scores and forward returns are wide frames, sessions down the rows and tickers across, so ranking a
whole cross-section is one operation along the row. Everything is measured per date and then
summarized across dates, which is what separates a factor that works from one that worked once.

Rank correlation is computed by ranking both sides and taking the Pearson correlation of the ranks,
which is Spearman's coefficient and needs nothing beyond pandas.
"""

import numpy as np
import pandas as pd

MIN_NAMES = 2  # a correlation needs two points, so a thinner cross-section reads NaN
TRADING_DAYS = 252


def factor_scores(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """One factor's values from a long factor panel, widened to sessions by tickers."""
    if factor not in panel.columns:
        raise ValueError(f'{factor!r} is not a column of this panel')
    wide = panel.pivot_table(index='Date', columns='Ticker', values=factor, aggfunc='first')
    wide.index = pd.DatetimeIndex(wide.index, name='Date')
    return wide


def forward_returns(closes: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Simple return over the `horizon` sessions after each session, dated by the session it follows.

    Simple rather than log, because a portfolio return is linear in simple returns.
    """
    if horizon < 1:
        raise ValueError(f'horizon must be at least 1 session, got {horizon}')
    return closes.shift(-horizon) / closes - 1.0


def _aligned(scores: pd.DataFrame, forwards: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two frames narrowed to the sessions and tickers they share, with either side's NaNs dropped."""
    dates = scores.index.intersection(forwards.index)
    tickers = scores.columns.intersection(forwards.columns)
    left, right = scores.loc[dates, tickers], forwards.loc[dates, tickers]
    both = left.notna() & right.notna()
    return left.where(both), right.where(both)


def rank_ic(scores: pd.DataFrame, forwards: pd.DataFrame) -> pd.Series:
    """Per-session rank correlation between a score and the forward return that followed it.

    Sessions holding fewer than MIN_NAMES paired observations read NaN, which is what the sessions at
    the end of the sample look like once the forward window runs past the data.
    """
    left, right = _aligned(scores, forwards)
    ic = pd.Series(np.nan, index=left.index, name='ic', dtype=float)
    scored = left.notna().sum(axis=1) >= MIN_NAMES
    if scored.any():
        # a cross-section where every score ties has no rank variance; pandas yields NaN for it
        with np.errstate(invalid='ignore'):
            ic.loc[scored] = left.loc[scored].rank(axis=1).corrwith(right.loc[scored].rank(axis=1), axis=1)
    return ic


def names_per_date(scores: pd.DataFrame, forwards: pd.DataFrame) -> pd.Series:
    """How many names each session's rank correlation was computed over."""
    left, _ = _aligned(scores, forwards)
    return left.notna().sum(axis=1).rename('n_names')


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean, dispersion, t-statistic, hit rate and persistence of an information-coefficient series.

    The t-statistic is what separates a factor with an edge from one that had a good run, so it is
    part of the default summary.
    """
    clean = ic.dropna()
    n = len(clean)
    if n == 0:
        return {'n_periods': 0, 'mean': np.nan, 'std': np.nan, 't_stat': np.nan,
                'information_ratio': np.nan, 'hit_rate': np.nan, 'autocorrelation': np.nan}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else np.nan
    t_stat = mean / std * np.sqrt(n) if std and not np.isnan(std) and std > 0 else np.nan
    return {
        'n_periods': n,
        'mean': mean,
        'std': std,
        't_stat': float(t_stat) if t_stat == t_stat else np.nan,
        'information_ratio': float(mean / std) if std and std > 0 else np.nan,
        'hit_rate': float((clean > 0).mean()),
        'autocorrelation': float(clean.autocorr()) if n > 2 else np.nan,
    }


def ic_decay(scores: pd.DataFrame, closes: pd.DataFrame, *, horizons: tuple[int, ...] = (1, 5, 21, 63)) -> pd.DataFrame:
    """Mean rank correlation at each forward horizon, showing how quickly a signal loses its edge."""
    rows = []
    for horizon in horizons:
        ic = rank_ic(scores, forward_returns(closes, horizon=horizon))
        summary = ic_summary(ic)
        rows.append({'horizon': horizon, 'mean_ic': summary['mean'], 't_stat': summary['t_stat'],
                     'n_periods': summary['n_periods']})
    return pd.DataFrame(rows)


def quantile_buckets(scores: pd.DataFrame, *, n_quantiles: int = 5) -> pd.DataFrame:
    """Which bucket each name falls in per session, 0 for the lowest scores and n-1 for the highest."""
    if n_quantiles < 2:
        raise ValueError(f'n_quantiles must be at least 2, got {n_quantiles}')
    pct = scores.rank(axis=1, pct=True)
    buckets = np.ceil(pct * n_quantiles) - 1
    return buckets.clip(lower=0, upper=n_quantiles - 1)


def quantile_returns(scores: pd.DataFrame, forwards: pd.DataFrame, *, n_quantiles: int = 5) -> pd.DataFrame:
    """Mean forward return of each score bucket per session, buckets across the columns."""
    left, right = _aligned(scores, forwards)
    buckets = quantile_buckets(left, n_quantiles=n_quantiles)
    long = pd.DataFrame({'bucket': buckets.stack(future_stack=True),
                         'forward': right.stack(future_stack=True)}).dropna()
    if long.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name='Date'), columns=range(n_quantiles), dtype=float)
    grouped = long.groupby([long.index.get_level_values(0), long['bucket']])['forward'].mean()
    wide = grouped.unstack('bucket')
    wide.index = pd.DatetimeIndex(wide.index, name='Date')
    wide.columns = [int(column) for column in wide.columns]
    return wide.reindex(columns=range(n_quantiles))


def quantile_frame(scores: pd.DataFrame, forwards: pd.DataFrame, *, n_quantiles: int = 5) -> pd.DataFrame:
    """Long-form quantile results, one row per session and bucket, with the names behind each mean."""
    left, right = _aligned(scores, forwards)
    buckets = quantile_buckets(left, n_quantiles=n_quantiles)
    long = pd.DataFrame({'bucket': buckets.stack(future_stack=True),
                         'forward': right.stack(future_stack=True)}).dropna()
    if long.empty:
        return pd.DataFrame(columns=['Date', 'Quantile', 'Return', 'NNames'])
    grouped = long.groupby([long.index.get_level_values(0), long['bucket']])['forward'].agg(['mean', 'size'])
    frame = grouped.reset_index()
    frame.columns = ['Date', 'Quantile', 'Return', 'NNames']
    frame['Quantile'] = frame['Quantile'].astype(int)
    return frame


def quantile_spread(quantiles: pd.DataFrame) -> pd.Series:
    """Top bucket's return minus the bottom bucket's, per session."""
    if quantiles.empty:
        return pd.Series(dtype=float, name='spread')
    top, bottom = max(quantiles.columns), min(quantiles.columns)
    return (quantiles[top] - quantiles[bottom]).rename('spread')


def factor_turnover(scores: pd.DataFrame, *, n_quantiles: int = 5) -> pd.Series:
    """Share of the top bucket replaced between consecutive sessions, the trading a signal implies."""
    buckets = quantile_buckets(scores, n_quantiles=n_quantiles)
    top = n_quantiles - 1
    membership = buckets == top
    previous = membership.shift(1)
    held = (membership & previous).sum(axis=1)
    current = membership.sum(axis=1)
    turnover = 1 - held / current.where(current > 0)
    turnover.iloc[0] = np.nan
    return turnover.rename('turnover')


def rank_autocorrelation(scores: pd.DataFrame, *, lag: int = 1) -> pd.Series:
    """Per-session correlation between a score's ranks and its ranks `lag` sessions earlier."""
    ranks = scores.rank(axis=1)
    with np.errstate(invalid='ignore'):
        return ranks.corrwith(ranks.shift(lag), axis=1).rename('rank_autocorrelation')


def annualized_spread(spread: pd.Series, *, horizon: int) -> float:
    """A per-period spread scaled to a yearly figure, given the horizon each period covers."""
    clean = spread.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean() * TRADING_DAYS / horizon)


# --- array-shaped helpers, shared with the experiments/ ranking models ---

def spearman_by_group(preds: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> float:
    """Mean rank correlation between predictions and targets, computed within each group."""
    correlations = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < MIN_NAMES:
            continue
        pred_ranks = pd.Series(preds[mask]).rank()
        target_ranks = pd.Series(targets[mask]).rank()
        correlation = pred_ranks.corr(target_ranks)
        if not np.isnan(correlation):
            correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else float('nan')


def relevance_grades(y: np.ndarray, groups: np.ndarray, n_grades: int = 5) -> np.ndarray:
    """Per-group percentile grade of each value, rising with the value.

    Grades come from `percentile * n_grades` truncated and clipped, so the top two percentiles share
    the highest grade. This is the grading the `experiments/` ranking models were trained against and
    is kept as it was; `quantile_buckets` is the evenly split version portfolio formation uses.
    """
    grades = np.zeros(len(y), dtype=np.int32)
    for group in np.unique(groups):
        mask = groups == group
        percentile = pd.Series(y[mask]).rank(pct=True).to_numpy()
        grades[mask] = np.clip((percentile * n_grades).astype(int), 0, n_grades - 1)
    return grades


def precision_at_k(preds: np.ndarray, targets: np.ndarray, groups: np.ndarray, k_frac: float) -> float:
    """Share of each group's predicted top-k that is also in its true top-k, averaged over groups."""
    precisions = []
    for group in np.unique(groups):
        mask = groups == group
        n = int(mask.sum())
        if n < MIN_NAMES:
            continue
        k = max(1, round(n * k_frac))
        predicted_top = set(np.argsort(-preds[mask])[:k])
        true_top = set(np.argsort(-targets[mask])[:k])
        precisions.append(len(predicted_top & true_top) / k)
    return float(np.mean(precisions)) if precisions else float('nan')
