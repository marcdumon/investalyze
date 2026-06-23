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

Fetches `yf.Ticker(t).info` per ticker into two tables — `company_profile` (address,
website, industry/sector, business summary, employee count, governance risk scores) and
`company_officers` (one row per officer). Reuses the Yahoo price provider's ticker list
and blacklist/dead exclusions, but tracks its own metadata-fetch failures independently
in `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv` — alongside the price provider's
state, with no subdir of its own (see Housekeeping below).

A ticker is only re-fetched once its stored `FetchedOn` is older than
`refresh_days_meta` (`ingest.toml [yahoo-meta]`) — running this repeatedly is cheap once
the DB is populated.

---

## Housekeeping — recheck blacklisted tickers

Each provider that can blacklist a ticker (a fetch failure, retried and tracked) gets a
housekeeping task that rechecks its blacklist: revives tickers that now succeed, ages out
ones that still fail, and moves a ticker to `dead.csv` past the provider's
`blacklist_max_attempts`.

```bash
# run every registered housekeeping task
python -m investalyze.ingest housekeeping

# run only one task
python -m investalyze.ingest housekeeping -t yahoo-blacklist
```

| Flag | Effect |
|------|--------|
| `-t NAME` | Run only this task; repeatable (`-t yahoo-blacklist -t yahoo-meta-blacklist`). Default: all. |

Current tasks:

| Task | Checks |
|------|--------|
| `yahoo-blacklist` | `data/yahoo/state/blacklist.csv`/`dead.csv` — the Yahoo price provider's failed tickers. |
| `yahoo-meta-blacklist` | `data/yahoo/state/meta_blacklist.csv`/`meta_dead.csv` — the Yahoo metadata provider's failed tickers (independent of the price provider's lists, but in the same dir). |

