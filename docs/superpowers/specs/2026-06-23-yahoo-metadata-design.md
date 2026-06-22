# Yahoo company metadata — design

## Context

Yahoo's `.info` (via `yf.Ticker(t)`) carries company profile data — address, website,
industry/sector, business summary, employee count, governance risk scores — plus a
`companyOfficers` list, none of which is captured today. `companies` (written by the
SimFin provider) has no Yahoo rows and lacks several of these fields entirely. This adds
a Yahoo-only metadata fetch, independent of the existing price/dividend/split ingest,
storing profile + officer data in two new tables.

## Scope

A new, independently-runnable provider (`yahoo-meta`) and its own housekeeping task
(`yahoo-meta-blacklist`). It reuses the existing Yahoo price provider's ticker universe
and blacklist/dead state — it does not introduce a second ticker list.

## Tables

**`company_profile`** — one row per ticker. PK `(Ticker, Src)`.

Columns: `Ticker, Src, address1, city, state, zip, country, website, industry, sector,
longBusinessSummary, fullTimeEmployees, auditRisk, boardRisk, compensationRisk,
shareHolderRightsRisk, overallRisk, irWebsite, FetchedOn`.

`Src` is always `'yahoo'` today; the column exists so another source could write here
later without a migration, mirroring `companies`. `FetchedOn` (ISO date) drives the
refresh cadence below.

**`company_officers`** — one row per officer. PK `(Ticker, Src, name)`.

Columns: `Ticker, Src, name, title, age, yearBorn, fiscalYear, totalPay, exercisedValue,
unexercisedValue`. Built from a fixed known key set (yfinance's typical
`companyOfficers` dict keys) so every batch produces the same DataFrame schema even when
a given officer's dict is missing some keys (missing -> null for that row).

Upserted via `storage.write` like every other table — no delete-then-insert. A roster
change (e.g. an officer leaving) leaves the old row in place; accepted, not handled.

## Ticker universe & failure tracking

Iterates `data/yahoo/raw/ticker.csv` (the existing Yahoo price provider's master list),
excluding tickers in *that provider's* `blacklist.csv`/`dead.csv` (no Yahoo price data ->
essentially never any Yahoo profile data either). No separate `yahoo-meta` ticker list.

`yahoo-meta` tracks its own metadata-fetch failures independently, under
`data/yahoo-meta/state/`, in the same shape as the price provider's state files:

- **`blacklist.csv`** — `ticker, market, attempts, first_blacklisted, last_checked`.
- **`dead.csv`** — `ticker, attempts, first_blacklisted, died_on`.

Identical column shape to the price provider's own state files (`market` is looked up
from `ticker.csv` the same way `provider.py:run` does it) — so `meta.py` reuses
`_load_blacklist`/`_load_dead` and the `_BLACKLIST_COLS`/`_DEAD_COLS` constants from
`provider.py` verbatim, against its own `data/yahoo-meta/state/` directory.

A ticker whose `.info` raises or comes back empty/minimal is treated as a failure:
blacklisted with `attempts=1` on first failure, incremented on each
`yahoo-meta-blacklist` recheck, moved to `dead.csv` past `blacklist_max_attempts` —
exactly the price provider's `recheck_blacklist` pattern, reapplied to a second,
independent state directory. This blacklist is *not* pruned from `ticker.csv` —
`yahoo-meta` doesn't own that file.

## Refresh cadence

`refresh_days_meta` (config, like SimFin's). On each run: fetch a ticker if it has no
`company_profile` row yet, or its `FetchedOn` is older than `refresh_days_meta` days.
Unlike SimFin's bulk-file-level cadence, this is evaluated per ticker (per-row
`FetchedOn`), since `yfinance` has no bulk metadata endpoint — one HTTP call per ticker.

## Fetch pacing

One ticker at a time (`yf.Ticker(t).info`, not the batched `yf.download`), sleeping
`sleep` seconds between calls — there's no multi-ticker `.info` call to batch, unlike
price history.

## Config

New `ingest.toml [yahoo-meta]` section:

```toml
[yahoo-meta]
batch_size              = 75   # tickers per save (controls commit granularity, not fetch batching)
sleep                   = 1.0  # seconds between per-ticker .info calls
refresh_days_meta       = 90   # re-fetch a ticker's profile once its FetchedOn is older than this
blacklist_max_attempts  = 5    # retries (via housekeeping) before a ticker is permanently dropped
```

## Module layout

**`src/investalyze/ingest/providers/yahoo/meta.py`** (new) — sibling to `provider.py`
(which already invites a split: "Split into more files in this folder if it grows").

```python
def fetch_meta(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict, *, update: bool = False) -> int:
    """Fetch + store Yahoo company profile + officers for due tickers. Returns company_profile row count."""

def recheck_meta_blacklist(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict) -> dict:
    """Retry every yahoo-meta-blacklisted ticker; revive/age out exactly like the price provider's recheck_blacklist."""
```

`fetch_meta` determines the due set (no row, or stale `FetchedOn`) by reading
`company_profile`'s existing `(Ticker, FetchedOn)` pairs from `con`, the same way
`provider.py`'s `run` reads existing tickers via `_load_existing_tickers`. Builds and
writes both tables per batch (batch = a chunk of due tickers, sized by `batch_size`),
same incremental-commit rationale as the price provider.

`update=False` is the only mode (`update` flag is accepted for signature parity with
every other provider's `run`, but ignored — there's no incremental/full distinction for
profile data, only "due" vs "not due").

**`providers/yahoo/provider.py`** gains nothing — `meta.py` imports its existing
`_load_blacklist`/`_load_dead`/`_chunk`/`_fetch`-style helpers and `_BLACKLIST_COLS`/
`_DEAD_COLS` constants rather than duplicating them (the CSV shape is identical, just
pointed at `data/yahoo-meta/state/` instead of `data/yahoo/state/`).

**`orchestrator.py`**: add `'yahoo-meta': meta.fetch_meta` to `PROVIDERS`. No other
changes — `create_data_dirs` and the `-p/--provider` CLI flag both already work off the
`PROVIDERS` dict generically. `data/yahoo-meta/raw/` and `processed/` will exist but go
unused (this provider has no raw files of its own) — same idle-subdir pattern as any
provider that doesn't use every standard subdir.

**`housekeeping.py`**: add `'yahoo-meta-blacklist': ('yahoo-meta', meta.recheck_meta_blacklist)`
to `HOUSEKEEPING_TASKS`. The `'yahoo-meta'` provider-name key means
`config.provider('yahoo-meta')` supplies `recheck_meta_blacklist`'s settings — same
`[yahoo-meta]` section `fetch_meta` uses, so `blacklist_max_attempts` lives in one place.
No `__main__.py` changes — `-t/--task` choices are already derived from
`sorted(housekeeping.HOUSEKEEPING_TASKS)`.

## Out of scope

- Deleting/cleaning up stale `company_officers` rows when a roster changes.
- Any non-Yahoo source writing to `company_profile`/`company_officers` (the `Src` column
  exists for future use only).
- Backfilling `companies` with Yahoo rows, or reconciling `company_profile` against
  `companies` — separate concern, not addressed here.
- A `yahoo-meta` ticker-coverage/discovery script (deferred — explicitly the user's
  next, separate task once this exists).

## Verification

- Unit-level: `fetch_meta` against a mocked `yf.Ticker` — due-ticker selection (missing
  row vs stale `FetchedOn` vs fresh-skip), profile + officer row shape, blacklist-on-
  failure. `recheck_meta_blacklist` — revive/increment/die-out against a fixture
  blacklist, same structure as `test_yahoo_housekeeping.py`.
- End-to-end: `python -m investalyze.ingest -p yahoo-meta` against real `data/`, then
  `python -m investalyze.ingest housekeeping -t yahoo-meta-blacklist`; confirm
  `company_profile`/`company_officers` populate and `data/yahoo-meta/state/` behaves
  like the price provider's state dir across two consecutive runs.
