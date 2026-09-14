# investalyze: Manual

User guide for the `investalyze` repo, built in parts and documented here as each lands.

**Part 1, the CLIs**: **ingest** (getting market data into the DB), **cleaning** (persistent
manual fixes for bad vendor data) and **quality** (anomaly detection over the ingested data).

**Part 2, the apps**: a six-page Dash UI over the same data. See
[Apps: the Dash UI](#apps-the-dash-ui), which is also the easier way to drive everything in
Part 1.

**Part 3, factors and backtests**: a factor library, IC analysis and an event-driven engine.
See [Factors and backtests](#factors-and-backtests). For the actual formula
behind a number (IC, Sharpe, fill price, ...), see
[`docs/backtest_metrics.md`](backtest_metrics.md) instead of this file.

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

## Factors and backtests

Two pages, one flow. On **Factors** you define a factor, check its diagnostics and run IC analysis,
which answers whether it predicts. On **Backtest** you trade it, which answers whether it pays after
costs. **Backtest this factor** on the Factors page opens the Backtest page with the factor selected.

```bash
python -m investalyze.research list                                  # the factor library
python -m investalyze.research fields research                       # search the data fields
python -m investalyze.research diagnostics momentum_12_1             # coverage, distribution, top and bottom names
python -m investalyze.research tearsheet momentum_12_1               # IC analysis
python -m investalyze.research backtest momentum_12_1,earnings_yield \
    --rebalance Q --quantiles 5 --size 1000 --holdout 2022-01-01       # store a factor portfolio run
python -m investalyze.research timing new_252d_high --trailing-stop 0.15 \
    --max-positions 20 --size 500                                      # store a timing rule run

python -m investalyze.backtest report --run-id 8894ca7dc4f0
python -m investalyze.backtest delete --run-id 8894ca7dc4f0 b9da938b2434
```

Costs, limits, starting cash and benchmark live in `backtest.toml`. For the formula behind a number,
see [`docs/backtest_metrics.md`](backtest_metrics.md).

### The factor library

One TOML file per factor under `factors/`, in git:

```toml
name = 'momentum_12_1'
expression = 'ret_12_1'
description = '12-1 month price momentum: winners keep winning over the next months.'
higher_is_better = true
status = 'candidate'
notes = ''
```

Status is `candidate`, `accepted` or `rejected`. Rejecting keeps the definition, so a dead end stays
recorded.

Results are cached under `data/cache`, keyed by the factor's fingerprint: a hash of its expression and
of every factor it references. Changing an expression invalidates its results and those of every
factor built on it. Changing the description, status or notes keeps them. The cache is safe to delete.

### Expressions

An expression reads data fields, other factors and functions.

**Statement line items**, named from their headings (`Research & Development` is
`research_development`). Income and cash flow items are TTM, balance sheet items MRQ, both read
point-in-time. A heading on two statements has only prefixed names, such as `income_net_income` and
`cashflow_net_income`.

**Market data**: `open`, `high`, `low`, `close`, `adj_close`, `volume`, `adv20`, `market_cap`, `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`,
`ret_12_1`, `vol_63`, `vol_252`, `beta_252`, `high_52w`, `div_yield`. **Reference**: `sector`,
`industry`, `name`.

**Functions**:

| kind | functions |
|---|---|
| cross-sectional, within a date's universe | `rank`, `zscore`, `winsorize(x, p)`, `neutralize(x, by=sector)` |
| time series, per ticker | `lag(x, n)`, `delta(x, n)`, `pct_change(x, n)`, `mean(x, n)`, `std(x, n)`, `max(x, n)`, `min(x, n)` |
| elementwise | `where(cond, a, b)`, `mask(x, cond)`, `safe_div`, `abs`, `log`, `clip`, `sign` |

Comparisons, `and`, `or` and `a if cond else b` work too. Division by zero gives NaN. A comparison gives 1
or 0, so a factor can be a condition, such as `adj_close >= max(adj_close, 252)`.

Time-series windows count trading days. A factor that uses one is computed on every session, starting
far enough before the first requested date to fill its longest window.

Use `mask(x, cond)` to restrict a factor to some names: excluded names become NaN and leave the
ranking. `where(cond, x, 0)` gives them zero instead, which still takes part in every ranking.

### The Factors page

**Diagnostics** run on the 400 most liquid names on each month-end from 2016 to 2024, without saving:
coverage (names per date with a value), distribution, top and bottom names with their inputs,
highest correlation with the rest of the library, and rank autocorrelation as a turnover proxy.

**IC analysis** runs a saved factor on the 1,000 most liquid names on each month-end: IC, t-stat and
hit rate at a 21-day horizon, in-sample against out-of-sample split at 2022-01-01, IC decay at 5, 21, 63
and 126 days, quintile returns, IC by year, and the number of false positives expected at p<0.05 for
a library of this size.

The factor library table shows each factor's t-stat, out-of-sample t-stat and Sharpe ratio from its
latest IC analysis and single-factor backtest.

### The Backtest page

**Factor portfolio**: pick one or more factors from the library. Several factors are combined as the
average of their percentile ranks, over names with a value for every factor. On each rebalance date
the universe is the most liquid names by trailing dollar volume known on that date. The portfolio is
long the top quantile, equal weight capped at `max_weight` per name, decided at the close and filled
at the next open. A single-factor portfolio holds the names IC analysis puts in the top quantile.

**Timing rule**: each name is entered and exited on its own conditions, all factors from the library.
A name is entered when its **entry factor** is above zero. It is exited on the first session one of the
chosen exits applies:

- the entry factor is no longer above zero (on by default),
- an **exit factor** is above zero,
- the close falls the **trailing stop** percentage below the highest close since entry.

At least one exit is required. At most **Max positions** names are held, each at `1 / max positions`
of equity. When more names qualify than there are free positions, the highest **priority factor**
values go first, or the most liquid names when none is picked. Decisions are made at every close and
filled at the next open.

The rule trades a saved Screener universe, or when none is picked, the most liquid names by trailing
dollar volume on each month-end. To change a rule, edit or add its factors on the Factors page.

Both run as a job and are stored in the same run tables. Selecting a stored run shows its equity
against the benchmark with drawdown, the in-sample and out-of-sample split, its settings (including
each factor's expression) and its fills.


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

### Throwing runs away

Tick the runs in the Stored runs table and press **Delete selected**; the header checkbox ticks all
of them. The confirmation lists each run and what goes with it: its summary row, its equity curve,
its fills, and the report written under `data/backtests/<run id>/`. Nothing else is touched.

The whole selection goes in one command, so an unknown run id among good ones still deletes the
good ones and then says which id was not found.

A run is reproducible: the same settings give the same run id and rebuild it. A factor portfolio's
run id includes each factor's fingerprint, so redefining a factor makes a new run.
Deleting is for clearing out trial runs, not for protecting a result.


### Point-in-time rules

Fundamentals are read as first reported (`IsRestated = FALSE`) and a quarterly window becomes
visible only once its newest filing was published. Publish lag runs a median 43 days and a 95th
percentile of 397, so quarters do arrive out of order; each window carries the publish date of
its latest member, which is what holds a window back until all of it was knowable.

Market data fields read the last session on or before each as-of date. The universe on each date is
chosen from trailing dollar volume known on that date, never from today's liquidity.

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
| `income` | Income statements, quarterly and annual, in both versions: as first published and as later restated. `IsRestated` separates them, and `Publish Date` is what makes point-in-time work. | `python -m investalyze.ingest simfin` |
| `balance` | Balance sheets, same shape and same two versions as the income statements. | `python -m investalyze.ingest simfin` |
| `cashflow` | Cash flow statements, same shape and same two versions as the income statements. | `python -m investalyze.ingest simfin` |

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
| `backtest_runs` | One row per stored run: the settings behind it and its headline statistics. | `python -m investalyze.research backtest`, `python -m investalyze.research timing` |
| `backtest_equity` | A stored run day by day: equity, gross and net exposure, cash, and the benchmark beside it. | `python -m investalyze.research backtest`, `python -m investalyze.research timing` |
| `backtest_trades` | Every fill a stored run made, with the price it paid and what the trade cost. | `python -m investalyze.research backtest`, `python -m investalyze.research timing` |

The query box runs on a read-only connection, so DuckDB itself refuses anything that writes:
`DELETE FROM prices` comes back as *Cannot execute statement of type "DELETE" ... attached in
read-only mode*. Results stop at 500 rows and a query is interrupted after 20 seconds.

## Apps: the Dash UI

```bash
python -m investalyze.apps          # then open http://127.0.0.1:8050
```

One server, six pages, a shared shell with a light/dark toggle in the sidebar. Every page
reads through a short-lived read-only connection.

DuckDB allows a single writer, so a running job locks the DB. Pages that hit it show a
"database busy" notice instead of a traceback, and notebook kernels holding the DB will block
jobs started from the UI. Close them first.

| Page | Path | For |
|------|------|-----|
| Control Panel | `/` | Run ingest, housekeeping, quality and cleaning; watch the job log. |
| Screener | `/screener` | Filter the universe down to a saved list of tickers. |
| Data Quality | `/quality` | Triage anomalies into the quality log or a staged cleaning fix. |
| Stock | `/ticker` | One ticker in depth: performance, fundamentals, statements, peers, risk. |
| Factors | `/factors` | Define factors, run diagnostics and IC analysis. |
| Backtest | `/backtest` | Backtest factor portfolios and timing rules, and read the stored runs. |
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
   1 while `margin_trend` steps by 0.001. "Clear bounds" blanks every input. Factors from the factor
   library appear as their own group, computed at the latest trading session; one named like a
   built-in factor is left out, and the panel says which.

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
