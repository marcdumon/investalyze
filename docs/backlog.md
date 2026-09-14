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

## 3. Banks, insurers and several large listings have no financial statements

Found 2026-09-13 building the Factors page. Of the 1,500 most liquid tickers, 353 have no balance
sheet, including JPM, BAC, GS, WFC, MS, C, AXP, SCHW and COF. SimFin publishes banks and insurers under
separate statement templates, and only the general template is ingested. The rest are share-class
tickers (GOOGL, BRK-B), foreign filers (ASML, BABA, SHOP, PDD, NU) and recent listings (GEV, CRWV,
CRCL, OKLO).

Every fundamental factor therefore excludes the financial sector, which biases value and quality
factors in particular. Price-based factors are unaffected.

What "done" looks like: the SimFin bank and insurance templates ingested and mapped onto the general
line items, and share-class tickers resolved to the filer that reports them. Depends on item 2 (the
SimFin key) for a fresh download.
