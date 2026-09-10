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

Every evaluation stored before that date had its period backfilled per `(version, universe, factor,
horizon)` from that factor's own rows, and different factors within the same historical run often
covered slightly different date ranges (a growth ratio needing warm-up data starts later than the
rest), reading as several near-duplicate periods per old scope instead of one. The backfill also ran
independently per table, so `factor_ic` and `factor_quantiles` fragments for the same historical run
did not even share an `EndDate` (`factor_quantiles` trailed `factor_ic` by about five weeks in every
scope cleaned up so far).

**Cleaned up 2026-09-10** for `as_reported`/`all` (re-evaluated, fragments deleted),
`as_reported`/`mid_large_mega_non_fin_realest` (re-evaluated, fragments deleted), and
`as_reported`/`unrecorded` (deleted outright: a pre-migration label for rows written before the
`Universe` column existed, not a real universe to re-evaluate).

**Still open:** `solar_small_mid_large`, both `as_reported` (2 periods in `factor_ic`) and `restated`
(4 periods). These are not warm-up fragmentation: every period there covers the same factor count but
genuinely different `EndDate`s (one stops 2024-12-31, another 2022-06-30), so they read as separate
deliberate evaluate calls, not pieces of one run. `as_reported`'s `factor_ic_decay` is messier still:
it holds two extra fragments (2020-01-03 to 2023-07-21, and 2021-11-12 to 2024-12-31) that don't even
align with either `factor_ic` period. Consolidating this scope needs a decision about whether the
distinct periods were meant to stay separate, not just a re-run.

## 4. `delete-evaluation` still clears a whole version-and-universe scope

**Fixed 2026-09-10.** `delete-evaluation` now narrows to the period picked on the Factors tab (all
its horizons), and stays scope-wide only when the picker is blank, pooled the same way the grid
reads. The CLI carries this through `--start`/`--end`, already shared flags. Fixing this also
surfaced that `delete_evaluation` never deleted `factor_ic_decay` rows: `EVALUATION_TABLES` predated
that table and was never updated, so decay rows outlived every earlier delete. Both are fixed
together in `writer.delete_evaluation`.
