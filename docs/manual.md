# investalyze — Manual

User guide for the `investalyze` repo. Work in progress — built in parts; each part is documented
here as it lands. **Part 1: ingest** (getting market data into the DB), **cleaning**
(persistent manual fixes for bad vendor data) and **quality** (anomaly detection over the
ingested data).

---

## Ingest — load market data

Paths come from `ingest.toml`. Typical flow:

```bash
# 1. create the data folders (once)
python -m investalyze.ingest setup

# 2. manually download the source files and drop them in data/<provider>/raw/
#    (Stooq is captcha-protected — URLs and steps are in ingest.toml)

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

## Yahoo metadata — company profile & officers

A separate provider, included in the default `python -m investalyze.ingest` run like any
other; run it on its own with:

```bash
python -m investalyze.ingest -p yahoo-meta
```

Fetches `yf.Ticker(t).info` per ticker into two tables — `_yahoo_companies` (address,
website, industry/sector, business summary, employee count, governance risk scores) and
`company_officers` (one row per officer). Reuses the Yahoo price provider's ticker list
and blacklist/dead exclusions, but tracks its own metadata-fetch failures independently
in `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv` — alongside the price provider's
state, with no subdir of its own (see Housekeeping below).

A ticker is only re-fetched once its stored `FetchedOn` is older than
`refresh_days_meta` (`ingest.toml [yahoo-meta]`) — running this repeatedly is cheap once
the DB is populated.

---

## Housekeeping

Maintenance tasks run separately from the regular provider ingest. Three kinds:

- **Blacklist rechecks** — each provider that can blacklist a ticker (a fetch failure,
  retried and tracked) gets a task that rechecks its blacklist: revives tickers that now
  succeed, ages out ones that still fail, and moves a ticker to its dead list past the
  provider's `blacklist_max_attempts`.
- **Combined-table rebuild** — `companies` merges the per-source `_yahoo_companies` +
  `_simfin_companies` raw metadata tables into one row per ticker.
- **Ticker identification rebuild** — `market_instruments` gives every `market_data` ticker
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
| `yahoo-blacklist` | Rechecks `data/yahoo/state/price_blacklist.csv`/`price_dead.csv` — the Yahoo price provider's failed tickers. |
| `yahoo-meta-blacklist` | Rechecks `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv` — the Yahoo metadata provider's failed tickers (independent of the price provider's lists, but in the same dir). |
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
| `bond_yield_bound` | warn | Bond abs(C) > 50: likely a price-quoted series misfiled as a yield. |
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
| `fundamentals_sanity` | error | Shares (Basic/Diluted) <= 0 on any statement; negative Total Assets. |
| `negative_revenue` | warn | Revenue < 0; legitimate for some financials, review per ticker. |
| `quarters_vs_fy` | warn | Sum of a year's 4 quarters vs the FY row (Revenue, Net Income, operating cash). |

The identity checks follow SimFin's signed convention (expenses stored negative, so every
identity is additive) and tolerate `greatest(1% of the total, 100k)` to skip rounding noise.

Review findings in `notebooks/3_data_quality.ipynb`: a per-check summary, then per check the
worst tickers and sample rows. Confirmed data bug → quirks-log entry → `cleaning.toml` →
`apply` → re-run the checks.
