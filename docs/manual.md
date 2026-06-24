# investalyze — Manual

User guide for the `investalyze` repo. Work in progress — built in parts; each part is documented
here as it lands. **Part 1: ingest** (getting market data into the DB).

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
| `--update` | Load the daily update file instead of the full history. |
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

