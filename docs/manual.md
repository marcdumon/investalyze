# investalyze: Manual

User guide for the `investalyze` repo, built in parts and documented here as each lands.

**Part 1, the CLIs**: **ingest** (getting market data into the DB), **cleaning** (persistent
manual fixes for bad vendor data) and **quality** (anomaly detection over the ingested data).

**Part 2, the apps**: a five-page Dash UI over the same data. See
[Apps: the Dash UI](#apps-the-dash-ui), which is also the easier way to drive everything in
Part 1.

**Part 3, the backtest**: point-in-time factors, factor evaluation and an event-driven engine.
See [Backtest](#backtest-signals-evaluation-and-simulated-portfolios).

---

## Ingest: load market data

Paths come from `ingest.toml`. Typical flow:

```bash
# 1. create the data folders (once)
python -m investalyze.ingest setup

# 2. manually download the source files and drop them in data/<provider>/raw/
#    (Stooq is captcha-protected, URLs and steps are in ingest.toml)

# 3. load the full history
python -m investalyze.ingest

# 4. later, apply the daily update
python -m investalyze.ingest --update
```

Options:

| Flag | Effect |
|------|--------|
| `-p NAME` | Run only this provider; repeatable (`-p stooq -p yahoo`). Default: all. |
| `--update` | Incremental load instead of full history. For Yahoo, re-fetches each ticker's last `refetch_days` (`[yahoo]`, default 7) so provisional/revised closes get corrected; for Stooq, loads the daily update file. |
| `--config PATH` | Use a different config file (default: `./ingest.toml`). |
| `--data-root PATH` | Override the data directory for this run. |

---

## Yahoo metadata: company profile & officers

A separate provider, included in the default `python -m investalyze.ingest` run like any
other; run it on its own with:

```bash
python -m investalyze.ingest -p yahoo-meta
```

Fetches `yf.Ticker(t).info` per ticker into two tables: `_yahoo_companies` (address,
website, industry/sector, business summary, employee count, governance risk scores) and
`company_officers` (one row per officer). Reuses the Yahoo price provider's ticker list
and blacklist/dead exclusions, but tracks its own metadata-fetch failures independently
in `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv`, alongside the price provider's
state, with no subdir of its own (see Housekeeping below).

A ticker is only re-fetched once its stored `FetchedOn` is older than
`refresh_days_meta` (`ingest.toml [yahoo-meta]`), running this repeatedly is cheap once
the DB is populated.

---

## Housekeeping

Maintenance tasks run separately from the regular provider ingest. Three kinds:

- **Blacklist rechecks**: each provider that can blacklist a ticker (a fetch failure,
  retried and tracked) gets a task that rechecks its blacklist: revives tickers that now
  succeed, ages out ones that still fail, and moves a ticker to its dead list past the
  provider's `blacklist_max_attempts`.
- **Combined-table rebuild**: `companies` merges the per-source `_yahoo_companies` +
  `_simfin_companies` raw metadata tables into one row per ticker.
- **Ticker identification rebuild**: `market_instruments` gives every `market_data` ticker
  (indices, bonds, currencies) a name and country, the equivalent of `companies` for
  non-stock instruments. Indices are manually curated in `stooq_tickers.toml` (arbitrary vendor
  symbols, no decodable structure); bonds and currencies follow systematic Stooq ticker patterns
  and are decoded automatically. Supports lookup in both directions, e.g. `WHERE Name ILIKE
  '%Nasdaq-100%'` finds `^NDX`.

```bash
# run every registered housekeeping task
python -m investalyze.ingest housekeeping

# run only one task
python -m investalyze.ingest housekeeping -t yahoo-blacklist
```

| Flag | Effect |
|------|--------|
| `-t NAME` | Run only this task; repeatable (`-t yahoo-blacklist -t companies`). Default: all. |

Current tasks:

| Task | Does |
|------|------|
| `yahoo-blacklist` | Rechecks `data/yahoo/state/price_blacklist.csv`/`price_dead.csv`, the Yahoo price provider's failed tickers. |
| `yahoo-meta-blacklist` | Rechecks `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv`, the Yahoo metadata provider's failed tickers (independent of the price provider's lists, but in the same dir). |
| `companies` | Rebuilds the combined `companies` table from `_yahoo_companies` + `_simfin_companies` (full outer join on Ticker; Yahoo wins overlapping fields). |
| `market-instruments` | Rebuilds `market_instruments`, name/country per market_data ticker (indices, bonds, currencies), see `stooq_tickers.toml`. |

---

## Cleaning: persistent manual data fixes

Vendor data is sometimes wrong at the source (e.g. `^NDX` ships pre-launch history that is a
scaled proxy of `^NDQ`). Ingest never deletes (every write is a merge-upsert), so a full
reload resurrects manually deleted rows. Fixes therefore live in a persistent registry and
are re-applied after any reload, as an explicit manual step (never inside ingest).

`cleaning.toml` lists the fix instances; each fix *type* is a module in
`src/investalyze/cleaning/` exposing `detect(con, fix)` and `apply(con, fix)`:

```toml
[[delete_date_range]]
table = 'market_data'
tickers = ['^NDX']
end = 1985-10-01          # inclusive TOML date; start/end optional (omitted = open-ended)
reason = 'pre-launch history is a scaled proxy of ^NDQ, see notebooks/9999_data_quirks.ipynb'
```

```bash
# report what each fix would touch (read-only): 0 rows = clean, N rows = pending
python -m investalyze.cleaning check

# delete the matching rows (idempotent; clean fixes are skipped)
python -m investalyze.cleaning apply
```

| Flag | Effect |
|------|--------|
| `--config PATH` | Fixes TOML (default: `./cleaning.toml`). |
| `--ingest-config PATH` | Ingest TOML giving the DB location (default: `./ingest.toml`). |

Workflow when a new problem is found:

1. Log the evidence in `notebooks/9999_data_quirks.ipynb`.
2. Known fix type → add a TOML entry. New kind of problem → add one fix-type module and
   register it in `registry.FIX_TYPES`.
3. `check`, then `apply`. After any full reload: `check` shows the resurrected rows, `apply`
   removes them again.

Note: `check` confirms the target rows *exist*, not that the quirk still *holds*: if a
vendor ever replaces bogus rows with real data, re-evaluate the entry by hand (its `reason`
points at the evidence).

Fix types:

| Type | Does |
|------|------|
| `delete_date_range` | Deletes all rows for the listed tickers within an inclusive date range. |

---

## Quality: anomaly detection

Systematic checks over the ingested data: prices, corporate actions and fundamentals.
Detect and report only: findings land in the `anomalies` table, source tables are never
touched, and fixes go through the Cleaning workflow above.

```bash
# run every check (close notebook kernels first: DuckDB allows one writer)
python -m investalyze.quality

# run a subset
python -m investalyze.quality extreme_return stale_run
```

| Flag | Effect |
|------|--------|
| `--ingest-config PATH` | Ingest TOML giving the DB location (default: `./ingest.toml`). |

Each run replaces the selected checks' rows in `anomalies` (delete-then-insert), so re-runs
are idempotent and a check that goes clean clears its old findings.

`anomalies` columns: `CheckName`, `Severity` (`error` = hard invariant broken, `warn` =
tunable threshold exceeded), `SrcTable`, `Ticker`, `Date` (prices family; NULL for
fundamentals), `Key` (`Market|Period|Fiscal Year|Fiscal Period|IsRestated` for
fundamentals), `Details` (the offending values, human-readable), `DetectedAt`.

Checks (thresholds are module constants in `src/investalyze/quality/`, and kwargs on each
check function for tuned re-runs from a notebook):

| Check | Sev | Flags |
|-------|-----|-------|
| `nonpositive_price` | error | O/H/L/C/AC <= 0 in `prices`; O/H/L/C <= 0 in `market_data` except bonds (yields, negative is fine). |
| `ohlc_inconsistent` | error | H < L, or O or C outside [L, H]; both tables, all asset classes. |
| `negative_volume` | error | Volume < 0 (guard, currently clean). |
| `bond_yield_bound` | warn | Yield-quoted bond abs(C) > 50: likely a price-quoted series misfiled as a yield. Stooq's `P`-suffixed bond-price series are excluded, since they sit near par by definition. |
| `extreme_return` | warn | Close more than doubles or halves overnight with no same-day split; tagged `(spike-and-revert)` when the next close jumps back. |
| `stale_run` | warn | 20+ consecutive identical closes; one finding per run, dated at its end. |
| `date_gap` | warn | More than 30 days between consecutive rows of a ticker. |
| `nonpositive_dividend` | error | Dividend <= 0 (guard, currently clean). |
| `oversized_dividend` | warn | Dividend above 25% of the same-day close. |
| `invalid_split_ratio` | error | Split ratio <= 0 or = 1 (guard, currently clean). |
| `balance_identity` | warn | Total Liabilities + Total Equity vs Total Assets. |
| `balance_subtotals` | warn | Current + noncurrent vs total, assets side and liabilities side. |
| `income_chain` | warn | Revenue → Gross Profit → Operating Income → Pretax Income Adj., link by link. |
| `cashflow_identity` | warn | Operating + investing + financing (+ FX, disc. ops) vs Net Change in Cash. |
| `hard_invariants` | error | Shares (Basic/Diluted) <= 0 on any statement; negative Total Assets. |
| `negative_revenue` | warn | Revenue < 0; legitimate for some financials, review per ticker. |
| `quarters_vs_fy` | warn | Sum of a year's 4 quarters vs the FY row (Revenue, Net Income, operating cash). |

The identity checks follow SimFin's signed convention (expenses stored negative, so every
identity is additive) and tolerate `greatest(1% of the total, 100k)` to skip rounding noise.

Review findings in `notebooks/3_data_quality.ipynb`: a per-check summary, then per check the
worst tickers and sample rows. Confirmed data bug → quirks-log entry → `cleaning.toml` →
`apply` → re-run the checks.

---

## Backtest: signals, evaluation and simulated portfolios

The **Backtest** page is the easier way to drive all of this; these are the commands behind it.

```bash
# score every factor against forward returns, over one universe or over everything
python -m investalyze.backtest evaluate --horizon 21 --universe mid_large_mega_non_fin_realest

# run a signal through the engine
python -m investalyze.backtest run --signal factor_composite --start 2015-01-01 --end 2024-12-31

# build the point-in-time factor panel both of those read. Add --build-panel to either command
# above to have it built first when the stored one does not cover what was asked for
python -m investalyze.backtest factors --start 2015-01-01 --end 2024-12-31 --panel-freq M

# or run a signal fold by fold, out of sample
python -m investalyze.backtest walkforward --signal factor_composite --start 2014-01-01 \
    --end 2024-12-31 --train 756 --test 252 --embargo 21

# what signals exist
python -m investalyze.backtest list

# print what a finished run wrote
python -m investalyze.backtest report --run-id 8894ca7dc4f0

# throw stored runs away
python -m investalyze.backtest delete --run-id 8894ca7dc4f0 b9da938b2434
```

Settings live in `backtest.toml`: starting cash, benchmark, rebalance frequency, the universe
rule, the cost model (slippage, commission, market impact, borrow) and the position limits.
Every key is optional.

**Read the survivorship note before any number.** `prices` holds the full history of the names
listed *today* and nothing for names that delisted earlier, so every backtest over this data is
biased upward. The note is computed per run, stored in `backtest_runs.SurvivorshipNote`, and
printed above the metrics rather than beneath them. It will weaken on its own if delisted price
history is ever ingested.

### Defining your own factors

**Screener page -> "Your factors"** at the top. The panel lists what is defined, and the form below
adds one, checking the formula as you type and showing the exact TOML it will write. Saving makes the
factor live in the Screener immediately; giving it values in backtests takes one press of "1. Build
factor panel" on the Backtest page.

It is stored in `factors.toml` at the repo root, which you can also edit by hand. Either way a
factor behaves like a built-in: a Screener column, a rank and a composite-score option, a column in
`factor_panel`, a row in the `evaluate` output, and a name a signal can trade on.

Two kinds share the file.

**Expression** factors combine columns the panel already holds:

```toml
[factors.cheap_quality]
expression = '(rank_fcf_yield + rank_roe + rank_op_margin) / 3'
higher_is_better = true
description = 'A name has to be cheap on free cash flow and profitable, not just one of the two.'
```

Alongside every built-in factor `x` you may use `rank_x`, its percentile from 0 to 100 within that
date's cross-section, already oriented so 100 is the good end. Ranking before combining is what puts
factors measured in different units on one scale. `last_close`, `mcap` and `mcap_bn` are available
too. Arithmetic and comparisons only: a function call is refused when the file is read.

**Ratio** factors read the statements directly, reaching quantities the built-in library does not
hold:

```toml
[factors.rd_intensity]
numerator = '-Research & Development'
denominator = 'Revenue'
higher_is_better = true
```

The statement is inferred from the column name, and a name appearing in two statements is refused as
ambiguous rather than resolved by guesswork. Income and cashflow terms are summed over the trailing
four published quarters; balance terms take the latest published quarter. A denominator of zero or
less yields NaN, matching how every built-in ratio guards its scale term.

SimFin stores expenses as negative numbers, so prefix a term with `-` to read its magnitude:
`-Research & Development` gives research spend as a positive share of revenue.

A new factor adds a column to `factor_panel` rather than needing the table rebuilt.

Errors are raised when the file is read, naming the factor and the offending term, so a typo fails
at startup rather than producing a column of NaN much later. Setting `INVESTALYZE_FACTORS_FILE` to
another path reads definitions from there; setting it empty leaves the library at its built-in 24,
which is what the test suite does so nobody's own factors can change what it asserts.

### Selecting the stocks a run trades

The **Universe** picker on the Backtest page lists every universe saved from the Screener. Leaving
it blank uses the whole database.

The two commands treat it differently, on purpose. **Building a panel uses the universe whole**: a
panel row exists as soon as a ticker has a close, so a name that listed halfway through the range
belongs in it. **A run keeps only the names priced across its entire range**, capped at Top N and
ordered by Ranked by, because a run needs one tradable set for its whole span.

That makes a longer range stricter, not looser, which is counterintuitive enough that the card now
says how much of the universe survives before you press:

| range | of 7 solar names |
|---|---|
| 2022 to 2026 | Trades 6, the other 1 was not priced across the whole range |
| 2016 to 2026 | Trades 4 |
| 2010 to 2026 | Trades 1 |
| 2000 to 2026 | None of them traded across this whole range; start later |

Without that, a sixteen-year backtest of a seven-name universe silently becomes a one-stock
backtest. Refusing outright, the CLI names the earliest start that would keep something: *the
earliest start that keeps one is 2006-11-17, when FSLR first traded*.

The same settings live in `backtest.toml` as defaults for a run started from the command line:

```toml
[universe]
name = 'mid_large_mega_non_fin_realest'
n_top = 300
rank_by = 'mcap'                          # or 'liquidity', 'ticker'
require_sector = true
```


### Delisted companies, and the bias they leave behind

`prices` holds the full history of the tickers listed **today**. A company that was taken over or
failed is gone from Yahoo entirely: asking its chart endpoint returns `404 No data found, symbol may
be delisted`. Measured on this database, 8,987 companies have fundamentals and 6,470 have prices, so
**2,181 companies filed statements and have no price history at all**, 966 of them filing in 2024 or
later. Every backtest here is biased upward by their absence and no engine change can recover them.

**No source for them has been found yet.** SimFin still serves their fundamentals but its prices are
not trusted here. Finding a provider that retains delisted history is the single most valuable thing
outstanding, because until then no strategy result can be taken at face value. It is item 1 in
[the backlog](backlog.md).

Two things are in place so the bias stops growing:

**The database keeps what Yahoo deletes.** A ticker delisting tomorrow leaves its stored history
untouched; only new bars stop arriving. So delisted names accumulate from now on, and the gap is
historical rather than ongoing.

**A run no longer excludes them.** The universe rule used to demand a name be priced on every
session of the range, which dropped every company that listed partway through and every one that
stopped: exactly the names whose absence is the bias. Runs now take every name that traded at any
point in the range. A name cannot be bought before it exists, and one that stops trading is
liquidated at its last traded price and recorded as a fill:

```
      Date Ticker Side  Shares  Price   Notional  Cost      Kind
2026-07-31   GREE sell 10000.0   2.14   21400.00   0.0  delisting
```

No costs are charged, because a delisting is a corporate event rather than a trade, and the last
close is the honest figure when the reason is unknown: a takeover paid close to it, and a failure had
already fallen to near nothing. Before this the position stayed frozen at that price to the end of
the run, untradeable and unusable, with the capital trapped.


### Two frequencies, two names

They used to be one setting. They answer different questions and no longer have to agree:

| | means | set on |
|---|---|---|
| **Panel cadence** | how often factor values are recomputed | both cards, used only when one would build |
| **Rebalance** | how often the portfolio trades | the Backtest card |

A session between panel dates reads the most recent one before it, so a monthly panel drives a
weekly run correctly and the card says the scores repeat between panel dates.

Neither is read from `backtest.toml` any more. That key had exactly one live effect, silently
choosing the cadence of a panel built from the Research card, which had no such field. A setting
that changes a result without appearing on the page that asked for it is worse than no setting.

### What the panel can and cannot hold

A factor like `0.5 * rank_roe + 0.5 * rank_roa` is a percentile, so its value depends on which names
were in the cross-section. That makes it a property of a ticker, a date **and a population**, and
`factor_panel` is keyed by the first two only. A build over seven solar names wrote those tickers
ranks against seven peers, into the same column that held ranks against six thousand.

So rank-based factors are not stored. They are computed by whoever reads the panel, over the names
that reader is working with, which is also the more correct answer: a signal choosing among 500
names should rank within those 500. Everything else is an absolute quantity and is stored as before.

The columns an earlier build left behind are dropped the next time one runs, since the numbers in
them are not comparable with each other.

### What a build actually costs

Measured on a 6,465-name database: about 4.5 seconds fixed, plus 0.13 seconds per as-of date. The
fixed part is six SQL joins over the whole fundamentals history, paid once whatever comes out of
them, which is why building only the factors a signal uses would save nothing worth having.

So a build computes only the dates not already stored for that vintage and those names. Refreshing
2000 to 2026 after a month passes builds one date in about 5 seconds instead of 321 in 46. Two
things override that: `--rebuild`, and a factor definition edited since the last build, whose stored
values came from a formula that no longer exists.

```bash
python -m investalyze.backtest factors --start 2015-01-01 --end 2024-12-31 --panel-freq M
python -m investalyze.backtest factors --start 2015-01-01 --end 2024-12-31 --panel-freq M --rebuild
```


### Two cards, one button each

The page has two cards because there are two jobs, and the factor panel is not one of them:

- **Research factors** scores the library against forward returns. From, To, Universe, Vintage, IC horizon.
- **Backtest a strategy** walks one signal through the engine. Signal, From, To, Rebalance, Universe, Top N, Ranked by, Vintage.

Each card owns its own copy of the fields it shares with the other. Editing the research range
cannot stale a backtest, and nothing on the page depends on the order you press things in.

`factor_panel` is a cache, not a task. Neither card asks you to build it. Each says under its button
what pressing it will do:

```
Scores 29 factors over 6,465 names from the stored panel.
Runs factor_composite over 7 names in solar_small_mid_large from the stored panel.
sma_cross reads prices only. Nothing to build.
```

When the panel cannot answer, the line names what would be built and why:

```
Builds the restated panel: 321 monthly dates for 7 names in solar_small_mid_large.
The panel holds nothing for this vintage and these names yet.
```

Pressing then does exactly that, in one job and one press. There is no confirmation, because the
line has already said what will happen. What a press cannot promise is how long a build takes, so
a **Cancel** appears beside the running job instead.

Two things make a build necessary, and both are detected: dates that were never built, and a factor
whose formula changed since the build. The second is invisible in the panel rows, so every build
records a fingerprint of the definitions it used and the card compares against it. The line says
that pressing is what builds them, because "do not exist yet" otherwise leaves open who is meant to
act.

A third case is reported but is not a reason to build: a factor with no values **for these names**,
where the inputs do not exist. None of seven solar companies has ever paid a dividend, so
`div_growth` is empty for them and no amount of building will produce one. Listing that among the
reasons to press a button said something untrue, so it is now a caveat and does not hold the card
up.

A rebalance finer than the panel's own cadence is not a gap. A session between panel dates reads the
most recent one before it, so a weekly run on a monthly panel works; the card says the scores repeat
between panel dates rather than asking you to rebuild.

A run the panel genuinely cannot reach is refused rather than traded flat: if fewer than half its
rebalance dates carry a score, or a factor it needs has no values, it stops and says so.

Nothing is ever built implicitly. The CLI only builds behind `--build-panel`, and the page passes
that flag only when its line said it would:

```bash
python -m investalyze.backtest run --signal factor_composite --start 2015-01-01 \
    --end 2024-12-31 --build-panel
```

### Reading a factor evaluation

Button 2 scores every factor against forward returns and stores the result under the universe it
ran on, so evaluating on the S&P 500 and then on everything leaves two readings rather than one
overwriting the other. The picker on the Factors tab chooses which to read; leaving it blank pools
them all, which is only meaningful when there is one.

Two figures are stored per score bucket: the mean forward return and the median. They part company
badly on an unfiltered universe, where a handful of sub-dollar names carry a bucket's mean far from
what its typical member did. Measured on the whole database, `ret_12_1`'s bottom bucket averaged
+22.6% while its median was +1.5%, with 11% of the rows under $5. Evaluate on a named universe when
you want the mean to mean anything. The Mean IC and t columns are rank-based and unaffected.


### Throwing an evaluation away

Re-running Evaluate replaces every factor for that vintage and universe, so a scope is rebuilt by
running it again. Removing one is the Factors tab: pick the universe, press **Delete evaluation**.
The confirmation names what goes:

```
Delete the as reported evaluation stored under unrecorded?

  29 factors, 2000-01-31 to 2026-09-02
  9,128 IC rows and 43,895 quantile rows
```

This is what clears a scope that should not be there any more, such as `unrecorded`, the label given
to rows written before evaluations recorded the universe they covered.

```bash
python -m investalyze.backtest delete-evaluation --vintage as_reported --universe unrecorded
```


### Throwing runs away

Tick the runs in the Stored runs table and press **Delete selected**; the header checkbox ticks all
of them. The confirmation lists each run and what goes with it: its summary row, its equity curve,
its fills, and the report written under `data/backtests/<run id>/`. Nothing else is touched, so the
runs left unticked and the factor evaluation stay where they are.

The whole selection goes in one command, so an unknown run id among good ones still deletes the
good ones and then says which id was not found.

A run is reproducible: the same signal, dates and universe give the same run id and rebuild it.
Deleting is for clearing out trial runs, not for protecting a result.


### What the pieces do

| Step | Module | Produces |
|------|--------|----------|
| Factor panel | `analysis/factors_pit.py` | `factor_panel`, one row per ticker and as-of date |
| Evaluation | `backtest/evaluation.py` | `factor_ic`, `factor_quantiles` |
| Weights | `backtest/portfolio.py` | Target weights from a cross-section of scores |
| Engine | `backtest/engine.py` | `backtest_runs`, `backtest_equity`, `backtest_trades` |
| Walk-forward | `backtest/walkforward.py` | Per-fold out-of-sample statistics |

### Point-in-time rules

Fundamentals are read as first reported (`IsRestated = FALSE`) and a quarterly window becomes
visible only once its newest filing was published. Publish lag runs a median 43 days and a 95th
percentile of 397, so quarters do arrive out of order; each window carries the publish date of
its latest member, which is what holds a window back until all of it was knowable.

`--vintage restated` reads the same rows the screener reads. That is look-ahead for any
historical date and exists so the two can be compared on the latest one.

### Two rules the engine holds to

- **Decide at a close, fill at the next open.** Orders generated from information at session `t`
  fill at session `t + 1`'s open, which is the primary look-ahead guard.
- **Fills come from the split-adjusted bars; dividends arrive as cash.** `O/H/L/C/V` are
  mutually consistent, so fills, stops and marks all read from them, and `AC` serves the signal
  layer alone. A position's share count is therefore constant across a split.

Long-only is the default. With total survivorship bias a short book never contains a company
that went to zero, which makes a long-short result the most misleading output available here.

---

## Database: what is in each table

The **Database** page at `/database` shows this live, with columns, how empty each one is, and
the first rows. Both it and this list read `apps/database/catalog.py`, so they cannot disagree,
and a test fails when a table exists in the database without a line here.

### Market data

| Table | Holds | Written by |
|---|---|---|
| `prices` | Daily equity bars. O/H/L/C/V are split-adjusted; AC is adjusted for dividends too, which is why a total-return calculation uses AC and a share count does not. | `python -m investalyze.ingest yahoo` |
| `market_data` | Daily bars for everything that is not an equity: indices, currencies, commodities. The benchmark a backtest measures against is read from here. | `python -m investalyze.ingest stooq` |
| `market_instruments` | What each non-equity ticker in market_data is, by name, country and asset class. | `python -m investalyze.ingest stooq` |
| `dividends` | Cash dividends per share, dated on the ex-date. A backtest credits these to cash rather than assuming they are reinvested. | `python -m investalyze.ingest yahoo` |
| `splits` | Share splits, as the ratio applied on that date. Used when checking that a price series moves for a reason. | `python -m investalyze.ingest yahoo` |

### Fundamentals

| Table | Holds | Written by |
|---|---|---|
| `income` | Income statements, quarterly and annual, in both vintages: as first published and as later restated. `IsRestated` separates them, and `Publish Date` is what makes point-in-time work. | `python -m investalyze.ingest simfin` |
| `balance` | Balance sheets, same shape and same two vintages as the income statements. | `python -m investalyze.ingest simfin` |
| `cashflow` | Cash flow statements, same shape and same two vintages as the income statements. | `python -m investalyze.ingest simfin` |

### Reference

| Table | Holds | Written by |
|---|---|---|
| `companies` | One row per ticker: sector, industry, name, exchange, employee count. The sector a screener filters on and a universe is built from comes from here. | `python -m investalyze.ingest housekeeping` |
| `company_officers` | Named officers and their reported pay, one row per person per company. | `python -m investalyze.ingest yahoo-profile` |
| `_yahoo_companies` (staging) | What Yahoo said about each company, kept as landed. Merged into companies rather than read directly, so a provider changing its mind does not silently change a result. | `python -m investalyze.ingest yahoo-profile` |
| `_simfin_companies` (staging) | The same for SimFin. Merged into companies alongside the Yahoo rows. | `python -m investalyze.ingest simfin` |

### Quality

| Table | Holds | Written by |
|---|---|---|
| `anomalies` | Every finding the quality checks have raised, with the check that raised it and enough of the row to go and look. A finding stays here until it is logged or fixed. | `python -m investalyze.quality` |

### Backtest

| Table | Holds | Written by |
|---|---|---|
| `factor_panel` | The precomputed factor table: one row per ticker per as-of date, holding each factor as it was knowable on that date. This is what a cross-sectional signal ranks. Rank-based factors are absent on purpose, since their value depends on which names were in the cross-section. | `python -m investalyze.backtest factors` |
| `factor_panel_builds` | One row per build of the factor panel: what was asked for, how long it took, and a fingerprint of the factor formulas in force, which is how a redefined factor is noticed. | `python -m investalyze.backtest factors` |
| `factor_ic` | Information coefficient per date: the rank correlation between a factor and the return that followed it. Averaged, this is the Mean IC column on the Factors tab. | `python -m investalyze.backtest evaluate` |
| `factor_quantiles` | Forward return of each score bucket per date, as a mean and as a median. The two part company on a universe that was never filtered for price. | `python -m investalyze.backtest evaluate` |
| `backtest_runs` | One row per stored run: the settings behind it, its headline statistics, and the survivorship note that belongs with any number read off it. | `python -m investalyze.backtest run` |
| `backtest_equity` | A stored run day by day: equity, gross and net exposure, cash, and the benchmark beside it. | `python -m investalyze.backtest run` |
| `backtest_trades` | Every fill a stored run made, with the price it paid and what the trade cost. | `python -m investalyze.backtest run` |

The query box runs on a read-only connection, so DuckDB itself refuses anything that writes:
`DELETE FROM prices` comes back as *Cannot execute statement of type "DELETE" ... attached in
read-only mode*. Results stop at 500 rows and a query is interrupted after 20 seconds.

## Apps: the Dash UI

```bash
python -m investalyze.apps          # then open http://127.0.0.1:8050
```

One server, five pages, a shared shell with a light/dark toggle in the sidebar. Every page
reads through a short-lived read-only connection.

DuckDB allows a single writer, so a running job locks the DB. Pages that hit it show a
"database busy" notice instead of a traceback, and notebook kernels holding the DB will block
jobs started from the UI. Close them first.

| Page | Path | For |
|------|------|-----|
| Control Panel | `/` | Run ingest, housekeeping, quality and cleaning; watch the job log. |
| Screener | `/screener` | Define your own factors, then filter the universe down to a saved list of tickers. |
| Data Quality | `/quality` | Triage anomalies into the quality log or a staged cleaning fix. |
| Stock | `/ticker` | One ticker in depth: performance, fundamentals, statements, peers, risk. |
| Backtest | `/backtest` | Research the factor library or backtest one signal, then read what they stored. |
| Database | `/database` | What every table holds, its columns and first rows, and a read-only query box. |

---

### Control Panel

The CLI commands documented above, as buttons, with live output. Three status cards across the
top (Freshness, Row counts, Anomalies) refresh on a 1 second poll.

Command cards, each labelled with what it writes:

| Card | Runs | Writes |
|------|------|--------|
| Ingest | Provider chips, none checked = all | Price + fundamentals tables |
| Housekeeping | Task chips (blacklist rechecks, derived-table rebuilds) | Derived tables |
| New listings | NASDAQ/NYSE listings scan | `ticker.csv` (new tickers added automatically) |
| Blacklist ticker | Deletes a ticker's data and blocks it from future scans | Every table, plus the blacklist |
| Quality checks | Check chips, grouped error / warn | `anomalies` only |
| Cleaning | Preview, then apply `cleaning.toml` | The rows each fix targets |

One job runs at a time. The console streams its log, offers Cancel, and keeps a History
accordion of finished runs. Below the cards sit two review grids: listings candidates (new,
reactivated-safe, reactivated-collision, the last needing a manual rename) and blacklisted
tickers, searchable, with an unblacklist action.

---

### Screener

Narrows the pool to a ticker list you save and reuse. Two grids: Candidates on the left with a
`matched/total` count, Selected on the right. Checking a row moves it across; unchecking sends
it back.

Filtering happens in two layers.

1. **Metadata filter bar**: search, sector, industry, minimum dollar volume, minimum years of
   history, maximum anomaly count, listing state, market-cap buckets.
2. **Factor value filters** (the accordion): a min and a max per factor, in the factor's own
   units. Hover a factor name for what it measures and the expression it is computed from.
   Spinner steps and displayed precision are sized per factor, so `interest_coverage` steps by
   1 while `margin_trend` steps by 0.001. "Clear bounds" blanks all 48 inputs.

Ranks are percentiles over the metadata-filtered candidates, so a percentile always means
"among my current candidates". They drive two things: the Rank display mode, and the composite
score, an equal-weight mean of whichever factors you pick.

Other controls: Raw / Rank display toggle, Save and Load for named universes, Blacklist for the
current selection, and Refresh to rebuild the cached pool. Clicking a row opens a detail panel
below; clicking a Ticker cell opens that ticker's Stock page in a new tab.

**Saved universes** are plain `data/universes/<name>.csv` files, and they are the handoff to the
rest of the repo:

```bash
python -m investalyze.ingest blacklist-ticker --universe <name> --reason '...'
```

The Stock page can use one as its peer group, and `experiments/dataset.py:load_universe` reads
them into notebooks.

---

### Data Quality

Triage for the `anomalies` table. Open findings on the left (filter by check, table, severity,
ticker), evidence and two forms on the right. Both forms act on the checked rows, falling back
to the clicked row when none are checked.

**Log finding(s)** writes to `quality_log.toml` with a tag: `real-problem`, `false-alarm`,
`investigate`, `known` or `fix-staged`. Logged findings drop off the open list.

**Stage cleaning fix** queues an entry in `cleaning.toml` and logs the rows as `fix-staged`.
Actions: delete rows, delete date range, delete entire ticker history, correct a value, clear a
value, rebuild adjusted close. Staging only queues the fix. Run it from Control Panel → Cleaning.

---

### Stock

One ticker end to end, reachable directly at `/ticker?symbol=AAPL`. Pick a ticker by symbol or
company name; the arrows step through the current list.

Sections, linked from the sticky bar: performance (price history, trailing returns, optional
multi-select benchmarks), fundamentals history (TTM flows from restated quarters, balance items
point-in-time), statements (income, balance, cashflow; annual or quarterly; click a line item to
chart it), factor profile and peer comparison, and risk.

The peer basis is selectable: industry, sector, or any saved universe. It sets the group behind
the percentiles and the violins, so peer percentiles mean whatever you chose. A ticker can also
be blacklisted from here.
