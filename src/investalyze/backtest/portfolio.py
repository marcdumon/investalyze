"""Turning a cross-section of scores into a cross-section of target weights.

Every function takes and returns a one-dimensional array covering the run universe in ticker order,
so a signal can chain them and hand the engine a weight vector. NaN in a score means no opinion and
becomes a zero weight; NaN never propagates into a weight.

The functions state intent. What a book can actually hold is enforced again by the engine, which
knows the positions and the day's liquidity.
"""

import numpy as np

TRADING_DAYS = 252
CAP_PASSES = 3  # clipping then renormalizing lifts the survivors, so the cap needs re-applying


def _finite(scores: np.ndarray) -> np.ndarray:
    """Boolean mask of the entries holding a usable number."""
    return np.isfinite(scores)


def winsorize(scores: np.ndarray, *, limit: float = 0.02) -> np.ndarray:
    """Pull the extreme `limit` share of each tail in to the value at that percentile."""
    if not 0 <= limit < 0.5:
        raise ValueError(f'limit must be in [0, 0.5), got {limit}')
    out = np.asarray(scores, dtype=np.float64).copy()
    usable = _finite(out)
    if limit == 0 or usable.sum() < 2:
        return out
    low, high = np.quantile(out[usable], [limit, 1 - limit])
    out[usable] = np.clip(out[usable], low, high)
    return out


def zscore(scores: np.ndarray) -> np.ndarray:
    """Standardize a cross-section to mean 0 and standard deviation 1, leaving NaN where it was."""
    out = np.asarray(scores, dtype=np.float64).copy()
    usable = _finite(out)
    if usable.sum() < 2:
        out[usable] = 0.0
        return out
    values = out[usable]
    spread = values.std(ddof=1)
    out[usable] = (values - values.mean()) / spread if spread > 0 else 0.0
    return out


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Rank a cross-section into [0, 1], where 1 is the highest score."""
    out = np.full(len(scores), np.nan, dtype=np.float64)
    usable = _finite(np.asarray(scores, dtype=np.float64))
    count = int(usable.sum())
    if count == 0:
        return out
    if count == 1:
        out[usable] = 1.0
        return out
    order = np.asarray(scores, dtype=np.float64)[usable].argsort().argsort()
    out[usable] = order / (count - 1)
    return out


def neutralize_groups(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean from its members, so every group centres on zero."""
    out = np.asarray(scores, dtype=np.float64).copy()
    usable = _finite(out)
    for group in np.unique(np.asarray(groups)[usable]):
        members = usable & (np.asarray(groups) == group)
        out[members] -= out[members].mean()
    return out


def neutralize_linear(scores: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Residual of the scores after a least-squares fit on the exposure columns, plus an intercept."""
    out = np.asarray(scores, dtype=np.float64).copy()
    matrix = np.asarray(exposures, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    usable = _finite(out) & np.isfinite(matrix).all(axis=1)
    if usable.sum() <= matrix.shape[1] + 1:
        return out
    design = np.column_stack([np.ones(usable.sum()), matrix[usable]])
    coefficients, *_ = np.linalg.lstsq(design, out[usable], rcond=None)
    out[usable] -= design @ coefficients
    return out


def quantile_weights(scores: np.ndarray, *, n_quantiles: int = 5, long_quantile: int | None = None,
                     short_quantile: int | None = None) -> np.ndarray:
    """Equal weight across the names in the chosen buckets, long side summing to 1.

    Buckets run 0 for the lowest scores to `n_quantiles - 1` for the highest; `long_quantile`
    defaults to the top one. Naming a `short_quantile` adds a short side summing to -1.
    """
    if n_quantiles < 2:
        raise ValueError(f'n_quantiles must be at least 2, got {n_quantiles}')
    long_quantile = n_quantiles - 1 if long_quantile is None else long_quantile
    ranks = percentile_rank(scores)
    weights = np.zeros(len(ranks), dtype=np.float64)
    usable = _finite(ranks)
    if not usable.any():
        return weights

    buckets = np.full(len(ranks), -1, dtype=np.int64)
    buckets[usable] = np.clip((ranks[usable] * n_quantiles).astype(int), 0, n_quantiles - 1)
    for bucket, target in ((long_quantile, 1.0), (short_quantile, -1.0)):
        if bucket is None:
            continue
        members = buckets == bucket
        if members.any():
            weights[members] = target / members.sum()
    return weights


def score_weights(scores: np.ndarray, *, top_n: int, power: float = 1.0) -> np.ndarray:
    """Weight the `top_n` highest scores in proportion to their rank raised to `power`.

    `power=0` gives equal weight across the selection, and larger powers concentrate on the best
    names. Weights sum to 1.
    """
    if top_n < 1:
        raise ValueError(f'top_n must be at least 1, got {top_n}')
    ranks = percentile_rank(scores)
    weights = np.zeros(len(ranks), dtype=np.float64)
    usable = np.flatnonzero(_finite(ranks))
    if usable.size == 0:
        return weights

    chosen = usable[np.argsort(-ranks[usable])[:top_n]]
    strength = np.power(ranks[chosen] + 1e-9, power) if power != 0 else np.ones(len(chosen))
    total = strength.sum()
    weights[chosen] = strength / total if total > 0 else 1.0 / len(chosen)
    return weights


def cap_weights(weights: np.ndarray, *, max_weight: float = 1.0, max_gross: float = 1.0,
                max_net: float = 1.0) -> np.ndarray:
    """Hold a weight vector to a per-name cap and to gross and net exposure limits.

    Clipping to the per-name cap and renormalizing lifts the names that were not clipped, so the pair
    is applied `CAP_PASSES` times. Gross and net are then scaled down if either exceeds its limit.
    """
    out = np.asarray(weights, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 0.0
    live = np.count_nonzero(out)
    if live and max_weight * live < 1.0:
        raise ValueError(f'max_weight={max_weight} over {live} names cannot reach full investment')

    for _ in range(CAP_PASSES):
        breached = np.abs(out) > max_weight
        if not breached.any():
            break
        target_gross = np.abs(out).sum()
        out = np.clip(out, -max_weight, max_weight)
        gross = np.abs(out).sum()
        if gross > 0 and target_gross > 0:
            headroom = np.abs(out) < max_weight
            shortfall = target_gross - gross
            room = (max_weight - np.abs(out[headroom])).sum()
            if shortfall > 0 and room > 0:
                out[headroom] += np.sign(out[headroom]) * (max_weight - np.abs(out[headroom])) * min(
                    shortfall / room, 1.0)

    gross = np.abs(out).sum()
    if gross > max_gross and gross > 0:
        out *= max_gross / gross
    net = out.sum()
    if abs(net) > max_net and abs(net) > 0:
        out *= max_net / abs(net)
    return out


def vol_target(weights: np.ndarray, vol: np.ndarray, *, target_annual_vol: float) -> np.ndarray:
    """Scale a weight vector so its estimated volatility matches the target.

    Volatility is estimated from the per-name volatilities alone, treating the names as uncorrelated.
    """
    if target_annual_vol <= 0:
        raise ValueError(f'target_annual_vol must be positive, got {target_annual_vol}')
    out = np.asarray(weights, dtype=np.float64).copy()
    sigma = np.asarray(vol, dtype=np.float64)
    usable = np.isfinite(sigma) & np.isfinite(out)
    estimated = np.sqrt(np.square(out[usable] * sigma[usable]).sum())
    if estimated <= 0:
        return out
    return out * (target_annual_vol / estimated)


def sector_neutral(weights: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    """Rescale within each sector so every sector holding a position carries the same gross weight."""
    out = np.asarray(weights, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 0.0
    labels = np.asarray(sectors)
    active = [group for group in np.unique(labels) if np.abs(out[labels == group]).sum() > 0]
    if not active:
        return out

    total_gross = np.abs(out).sum()
    share = total_gross / len(active)
    for group in active:
        members = labels == group
        out[members] *= share / np.abs(out[members]).sum()
    return out
