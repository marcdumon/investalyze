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

## 3. Evaluations are not keyed by the period they cover

`factor_ic` and `factor_quantiles` are keyed by version, universe and factor. Evaluate 2000 to 2010,
then 2015 alone, and the second silently replaces the first: verified. The Factors tab shows a
smaller period count with nothing saying why.

Universes sit side by side correctly; periods do not. Doing it means adding the range to the key and
a period picker on the Factors tab, which is why it has not been done casually.

## 4. Factor IC decay is computed but never shown

`evaluation.ic_decay` exists and nothing on the page calls it. It is the measurement that says what
rebalance frequency a factor deserves, and answering that question currently means writing a script.

Measured by hand on the top 100 liquid names, 2020 to 2024, mean IC by horizon:

| factor | 5d | 21d | 63d | 126d |
|---|---|---|---|---|
| ret_12_1 | +0.019 | **+0.040** | +0.032 | +0.015 |
| earnings_yield | -0.046 | -0.028 | **-0.068** | -0.066 |

`ret_12_1` peaking at 21 days is the argument against rebalancing weekly, and it belongs on screen
next to the Rebalance field rather than in a chat log.

## 5. The Screener and the factor panel compute factors separately

The Screener computes factors live from `analysis/factors.py`; a backtest reads them from
`factor_panel`. The values differ slightly, because one is a snapshot and the other is
point-in-time. Nobody has reconciled them, and a reader comparing a Screener column against a
backtest factor will find numbers that do not match.

Deliberately not converged: making the Screener read `factor_panel` would give a live page a hard
dependency on a table that must be built first, which is a regression in robustness for a page that
currently always works.
