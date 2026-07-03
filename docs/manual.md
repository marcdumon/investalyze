# investalyze — Manual

User guide for the `investalyze` repo. Work in progress — built in parts; each part is documented
here as it lands. **Part 1: ingest** (getting market data into the DB) and **cleaning**
(persistent manual fixes for bad vendor data).

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

Maintenance tasks run separately from the regular provider ingest. Two kinds:

- **Blacklist rechecks** — each provider that can blacklist a ticker (a fetch failure,
  retried and tracked) gets a task that rechecks its blacklist: revives tickers that now
  succeed, ages out ones that still fail, and moves a ticker to its dead list past the
  provider's `blacklist_max_attempts`.
- **Combined-table rebuild** — `companies` merges the per-source `_yahoo_companies` +
  `_simfin_companies` raw metadata tables into one row per ticker.

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

