# Backtest metrics: formulas

Every formula here is transcribed from the function that computes it, named alongside it, so this
stays checkable against the code rather than trusted on its own. `docs/manual.md` explains workflow
and what a number means for you; this is the arithmetic underneath it.

Notation: `t` indexes a session (a date the market traded), `i` indexes a name, `h` a horizon in
sessions. A cross-section is "the names at one `t`." `rank(x)` ranks lowest to highest; ties share
the average rank.

## Cross-sectional transforms

`src/investalyze/backtest/portfolio.py`, applied to one cross-section at a time.

**Winsorize** (`winsorize`): clip each tail's outer `limit` share to the value at that percentile.

    q_lo, q_hi = quantile(scores, [limit, 1 - limit])
    scores' = clip(scores, q_lo, q_hi)

**Z-score** (`zscore`): centre and scale to unit variance.

    z_i = (score_i - mean(score)) / std(score, ddof=1)

**Percentile rank** (`percentile_rank`): rank onto `[0, 1]`, 1 is the best name.

    p_i = rank(score_i) / (n - 1)

**Group / linear neutralization** (`neutralize_groups`, `neutralize_linear`): subtract each group's
mean, or the fitted value of an OLS regression of the score on the given exposures plus an
intercept. What is left is the score unexplained by sector (or by whatever columns were given).

## Factor evaluation

`src/investalyze/backtest/evaluation.py`.

**Forward return** (`forward_returns`), simple, not log, because a portfolio return is linear in
simple returns:

    R_i,t(h) = C_i,t+h / C_i,t - 1

**IC horizon** is `h` above: how many sessions ahead the return is measured.

**Rank IC** (`rank_ic`), per session, Spearman's coefficient computed as Pearson correlation of the
ranks, which is exactly equivalent and needs nothing beyond `corrwith`:

    IC_t = corr( rank_i(score_i,t), rank_i(R_i,t(h)) )   across names i present at t

A session with fewer than 2 scored names reads `NaN`. **Mean IC** is `mean(IC_t)` over the dates
evaluated, which is what the Factors tab shows per factor and horizon.

**IC summary** (`ic_summary`), given the per-date series `IC_t` with `n` non-`NaN` values:

    mean       = mean(IC)
    std        = std(IC, ddof=1)
    t_stat     = mean / std * sqrt(n)
    info_ratio = mean / std                    # per-period, not annualized
    hit_rate   = share of dates with IC_t > 0
    autocorr   = corr(IC_t, IC_t-1)             # lag-1 autocorrelation

`t_stat` is the number that separates a factor with a real edge from one that had a good run: a
mean IC near zero with a large `n` and a small `std` can still have a t-stat worth trusting.

**IC decay** (`ic_decay`): `rank_ic` recomputed at several horizons, showing where the factor's edge
peaks and how fast it fades. Measured on this database, top-100 liquid names, 2020-2024:

| factor | 5d | 21d | 63d | 126d |
|---|---|---|---|---|
| ret_12_1 | +0.019 | **+0.040** | +0.032 | +0.015 |
| earnings_yield | -0.046 | -0.028 | **-0.068** | -0.066 |

**Quantile buckets** (`quantile_buckets`): rank each name into `n_quantiles` equal-count groups by
percentile rank, 0 the lowest scores:

    bucket_i,t = clip( ceil( pct_rank_i,t * n_quantiles ) - 1,  0,  n_quantiles - 1 )

**Quantile spread** (`quantile_spread`): top bucket's mean forward return minus the bottom bucket's,
per date. `quantile_frame` stores the median forward return alongside the mean for the same reason
a factor's IC is rank-based: a mean is one extreme name away from meaning nothing, a median is not.

**Factor turnover** (`factor_turnover`): share of the top bucket's membership that was not in the
top bucket the previous session, i.e. the trading a signal would actually imply:

    turnover_t = 1 - |top_t ∩ top_t-1| / |top_t|

**Rank autocorrelation** (`rank_autocorrelation`): `corr(rank_t, rank_t-lag)` across names, a second
way of asking how fast a factor's ordering changes.

## Portfolio construction

`src/investalyze/backtest/portfolio.py`. A weight vector is one cross-section, one entry per name.

**Quantile weights** (`quantile_weights`): equal weight within the chosen bucket(s), long side
summing to 1, an optional short side to -1.

**Score weights** (`score_weights`): among the `top_n` highest-ranked names, weight proportional to
`percentile_rank ^ power`, normalized to sum to 1. `power = 0` is equal weight; larger powers
concentrate on the best-ranked names.

**Weight caps** (`cap_weights`): clip to `max_weight` per name, redistribute the clipped amount
across the names with headroom, repeat `CAP_PASSES` times (clipping then renormalizing lifts names
that were not clipped, so the cap needs re-applying), then scale the whole vector down if gross or
net exposure still exceeds its limit.

**Volatility targeting** (`vol_target`): scale a weight vector so its estimated volatility matches a
target, treating names as uncorrelated:

    estimated_vol = sqrt( sum( (w_i * sigma_i)^2 ) )
    w' = w * (target_vol / estimated_vol)

**Sector neutral** (`sector_neutral`): rescale within each sector holding a position so every active
sector carries the same total gross weight.

## Execution and costs

`src/investalyze/backtest/orders.py`. Signals decide at a close; orders fill at the next open.

**Reference price** (`reference_price`), before slippage:

- market: the session's open.
- limit: fills only if the session traded through the limit; takes the better of the open and the
  limit price (so a favourable gap is captured, not given back).
- stop: fills only if the session traded through the stop; takes the worse of the open and the stop
  (so a gap through the stop fills at the open, the way a real stop would).

**Fill price** (`fill_price`): the reference price moved against the order by fixed slippage plus a
size-dependent market-impact term:

    impact     = impact_coef * sqrt(notional / ADV)     # 0 if ADV or notional is 0
    fill_price = reference * (1 + side * (slippage_bps / 1e4 + impact))

`side` is +1 for a buy, -1 for a sell, so both costs always work against the order. `ADV` is the
session's average daily dollar volume (`market.py`, a rolling median).

**Commission** (`commission`): a rate on notional, floored at a per-fill minimum.

    commission = max(commission_bps / 1e4 * |notional|,  min_commission)

## Performance statistics

`src/investalyze/backtest/performance.py`. `returns` is `equity.pct_change()`; ratios annualize from
`TRADING_DAYS = 252` unless stated otherwise.

**Drawdown** (`drawdown_series`, `max_drawdown`): fractional distance below the running peak.

    drawdown_t = equity_t / max(equity_0..t) - 1
    max_drawdown = min(drawdown_t)

**CAGR** (`cagr`): compound growth rate implied by the first and last point of the curve.

    years = (last_date - first_date).days / 365.25
    CAGR  = (equity_end / equity_start) ^ (1 / years) - 1

**Annualized volatility** (`annualized_volatility`): `std(returns, ddof=1) * sqrt(252)`.

**Sharpe** (`sharpe`), `excess = returns - risk_free / 252`:

    Sharpe = mean(excess) / std(excess, ddof=1) * sqrt(252)

**Sortino** (`sortino`): the same excess return, but the denominator uses only the losing sessions'
downside deviation instead of the full standard deviation.

    downside = excess[excess < 0]
    Sortino  = mean(excess) / sqrt(mean(downside^2)) * sqrt(252)

**Calmar** (`calmar`): `CAGR / |max_drawdown|`.

**Hit rate** (`hit_rate`): share of sessions with a positive return.

**Turnover** (`turnover`, `annual_turnover`): traded notional as a share of that session's equity,
averaged and annualized.

    turnover_t     = sum(fill notional on date t) / equity_t
    annual_turnover = mean(turnover_t) * 252

**Exposures** (`exposures`): mean and peak of `gross / equity` and `net / equity` across the run.

**Beta and alpha** (`beta_and_alpha`), against a benchmark return series on shared sessions:

    beta  = cov(strategy, benchmark) / var(benchmark)
    alpha = ( mean(strategy) - beta * mean(benchmark) ) * 252

**Information ratio** (`information_ratio`): the annualized mean of the return difference over its
own volatility, `active = strategy - benchmark`:

    info_ratio = mean(active) / std(active, ddof=1) * sqrt(252)

**Capture ratios** (`capture_ratios`): mean strategy return on the benchmark's up (or down) sessions,
divided by the benchmark's own mean return on those same sessions.
