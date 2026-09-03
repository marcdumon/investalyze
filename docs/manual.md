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

Everything below is a button on the **Backtest** page, which is the easier way to drive it. The
commands are what those buttons run.

```bash
# 1. build the point-in-time factor panel (needed by every cross-sectional signal)
python -m investalyze.backtest factors --start 2015-01-01 --end 2024-12-31 --freq M

# 2. score every factor against forward returns
python -m investalyze.backtest evaluate --horizon 21

# 3. run a signal through the engine
python -m investalyze.backtest run --signal factor_composite --start 2015-01-01 --end 2024-12-31

# 4. or run it fold by fold, out of sample
python -m investalyze.backtest walkforward --signal factor_composite --start 2014-01-01 \
    --end 2024-12-31 --train 756 --test 252 --embargo 21

# what signals exist
python -m investalyze.backtest list
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

The **Universe** picker on the Backtest page lists every universe saved from the Screener. It is the
candidate pool; **Top N** and **Ranked by** then cut it to the names priced across the whole range.
Leaving it blank ranks the whole database.

The same settings live in `backtest.toml` as defaults for a run started from the command line:

```toml
[universe]
name = 'mid_large_mega_non_fin_realest'
n_top = 300
rank_by = 'mcap'                          # or 'liquidity', 'ticker'
require_sector = true
```


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
| Backtest | `/backtest` | Configure and launch a run, then read its equity curve, fills and factor evaluation. |

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
