# Backlog

Things worth coming back to, roughly in order of what they cost to leave undone. Not a task tracker:
each entry says what is wrong, why it matters, and what "done" would look like, so it can be picked
up cold.

Add to this rather than relying on remembering.

---

## 1. Find a data provider with delisted company prices

**The single most valuable thing outstanding.** Until it is solved, no backtest result here can be
taken at face value.

`prices` holds the full history of tickers listed **today**. A company that was taken over or failed
is gone from Yahoo entirely: its chart endpoint returns `404 No data found, symbol may be delisted`,
verified directly against CDXC, PCYG, ALVR and IKNA on 2026-09-03.

Measured the same day:

| | count |
|---|---|
| companies with fundamentals | 8,987 |
| tickers with price history | 6,470 |
| **fundamentals but no prices at all** | **2,181** |

966 of those 2,181 filed in 2024 or later, so they are recent disappearances, not ancient history.
That is roughly a quarter of the universe missing, all of it in the same direction: the losers.

Ruled out:

- **Yahoo**: deletes delisted symbols. Structural, not a query problem.
- **SimFin prices**: rejected. SimFin has a serious restatement problem and its prices are not
  trusted here. Its fundamentals are fine and are already in use.

What "done" looks like: a provider that retains delisted history, ingested alongside Yahoo into
`prices`. Whether a caveat belongs back on a run's report once real delisted history exists is a
separate call, made then, not promised now.

When evaluating any new provider, ask this question first, before price or coverage or API quality.

**Partly mitigated already:** the database keeps history after Yahoo deletes a symbol, so names
delisting from now on stay. Runs include names that listed or stopped mid-range, and the engine
liquidates a holding whose name stops trading. The gap is historical, not ongoing.

## 2. The SimFin API key is rejected

`401 Invalid API Key - check the key again and also if you confirmed your e-mail on registration`,
returned on 2026-09-03 for `income-annual-full`, a dataset already downloaded here in July. So the
key expired or rotated rather than the request being wrong.

Fundamentals cannot be refreshed until it is replaced. The stored zips are from July, so nothing has
broken yet, only stopped moving.

## 3. Old evaluations have their periods fragmented, one per factor's warm-up boundary

**Fixed 2026-09-09** for evaluations run from now on: `factor_ic`, `factor_quantiles` and the new
`factor_ic_decay` are keyed by version, universe, factor, horizon and period, and the Factors tab
carries a period picker (defaulting to the newest, poolable by clearing it).

What is still open: every evaluation stored before that date has its period backfilled from its own
rows, and different factors within the same historical evaluate run often cover slightly different
date ranges (a growth ratio needing warm-up data starts later than the rest). That reads today as
several near-duplicate periods per old scope instead of one. Verified on `all`: the blank-universe
picker lists nine periods from 2001 to 2026 for what was really a handful of evaluate runs.

What "done" looks like: re-running `evaluate` over each affected scope's original range, which
consolidates it back to one period per run, the way a fresh evaluation already does. The heaviest is
`all`, a long run over the full universe, not something to do casually.

## 4. `delete-evaluation` still clears a whole version-and-universe scope

`replace_evaluation` now keys a re-run by period and horizon, so a scope can hold several readings
side by side. `delete-evaluation` did not follow: it still removes every period and horizon a scope
holds in one action, which is a coarser tool than the picker that reads them apart. Not urgent, and
not decided: whether deleting should narrow to the period showing on the Factors tab, or stay
scope-wide, is a design question nobody has picked up yet.
