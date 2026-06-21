# Data Reference — `investalyze.ingest`

Living document. Single source of truth for: what data we have, where it comes from, how it's
cleaned (ETL), and how it lands in the DB. Update this as decisions are made.

Status legend: ✅ decided · 🔶 pending decision · ❓ to verify

---

## 1. Provider responsibilities

Each asset class has **exactly one** source. No multi-source overlap on the same instrument.

| Provider | Owns | Datasets |
|----------|------|----------|
| **Yahoo** (`yfinance`) | Stock prices | OHLCV, dividends, splits |
| **SimFin** (bulk REST) | Fundamentals | income, balance, cashflow — **as-filed + restated** (US) |
| **Stooq** (manual download) | Bonds, currencies, indices | OHLC(V) |

Design rule ✅: a provider is a self-contained module. Adding a 4th provider must not touch the
others. More providers expected later.

> Change from old `dataload`: Stooq no longer supplies **stock** prices (its stock data isn't
> dividend-adjusted → poor quality). Yahoo owns all equities; Stooq is scoped to non-equity.

---

## 2. Provider detail

### 2.1 Yahoo — stock prices ✅ (built)
- **Acquire:** `yfinance`, batched `yf.download(auto_adjust=False, actions=True, group_by='ticker')`.
  `batch_size`/`sleep` come from `[yahoo]` in `ingest.toml`. Dividends/splits come from the same
  download (`Dividends`, `Stock Splits` columns) — no separate `yf.Ticker.actions` call.
- **Fields:** `Open, High, Low, Close, Volume` → stored as `O, H, L, C, V`; `Dividends`/`Stock Splits`
  → `dividends`/`splits` event tables.
- **Adjustment** ✅: store **raw** OHLCV + our own derived adjusted close (`AC`), back-adjusted from
  raw close + events (not Yahoo's, not `auto_adjust`). Validated against Yahoo's `Adj Close` at
  `ac_tolerance` (0.1%) during ingest. A new dividend/split rewrites the ticker's whole `AC` history,
  recomputed cheaply from stored raw + events (no re-download).
- **Update:** incremental — one batched call per batch from the earliest stored date across it
  (full history if any ticker is new); overlap re-fetches a few rows, idempotent via merge upsert.
- **Symbol mapping:** canonical ticker → Yahoo symbol can differ (e.g. preferred `ARRY_A` →
  `ARRY-PA`). For equities the canonical usually equals the Yahoo symbol.
- **State** ✅: delisted / no-data tickers recorded in `state/blacklist.csv` and skipped on later runs;
  AC sanity-check offenders written to `state/ac_discrepancies.csv` (non-fatal).

### 2.2 SimFin — fundamentals ✅ (built)
- **Acquire:** bulk ZIPs from SimFin REST API (auth `api-key` header → presigned S3 redirect; auth not
  forwarded to S3), refreshed by file age (`refresh_days_fundamentals`, `refresh_days_meta`). No
  incremental feed — `--update` is governed by file age, same as a full run. API key from env
  **`SIMFIN_API_KEY`** (e.g. `.env`), never config.
- **Scope:** `us` market only (de excluded — re-addable, the `Market` column + keys already carry it).
  Statements `income`, `balance`, `cashflow`. `derived` (SimFin's pre-computed ratios) **excluded** —
  recomputable downstream.
- **Vintages** ✅: as-reported (original) + restated (latest revised), folded into one boolean
  `IsRestated` column (`False`/`True`), NOT separate `_restated` tables. Verified the two vintages are
  schema-identical per statement (income 68 / balance 95 / cashflow 62 cols), so the union folds with
  no NULL columns.
- **Granularity:** Annual (`A`) + Quarterly (`Q`), in one `Period` column.
- **Tables:** `income`/`balance`/`cashflow` — wide source columns **verbatim** + added `Market`,
  `Period`, `IsRestated`, `Src` (`simfin`), `SrcId` (renamed `SimFinId`). `companies` — per-market
  companies `LEFT JOIN industries` (Industry/Sector folded in). Merge keys: fundamentals
  `[Ticker, Market, Period, IsRestated, 'Fiscal Year', 'Fiscal Period']`, companies `[Ticker, Market]`.
- **Temporal columns** ✅: rows carry `Fiscal Year`, `Fiscal Period`, `Report Date`, `Publish Date`,
  `Restated Date` → **point-in-time is feasible** (original-as-published + latest restated, split by
  `IsRestated`).
- **Format:** semicolon-delimited CSVs inside the zips; wide, source-defined columns (kept verbatim).

### 2.3 Stooq — bonds, currencies, indices ✅
- **Acquire:** NOT programmatic. User manually downloads from stooq.com and drops files in raw dir.
  **Two inputs, two read paths** ✅ (both → `market_data`, both saved via the merge/upsert so they converge):
  | Input | Use | Format | AssetClass from |
  |-------|-----|--------|-----------------|
  | **`d_world_txt.zip`** (bulk) | full history | folder tree `world/<category>/[sub]/<ticker>.txt` | **folder name** (`_asset_class_from_category`) |
  | **`data_d.txt`** (flat) | daily update | one file, all instruments, no category | **ticker pattern** (`_asset_class_from_ticker`) |
  - **Skip `d_us_txt.zip`** ✅ — that's US equities (Yahoo owns them).
- **Raw row format:** `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`,
  comma-delimited; date `YYYYMMDD`; VOL always 0.
- **Ticker-pattern classification** (flat file, no folders) ✅:
  `.B`→bonds · `^`→indices · bare 6-letter alpha→currencies · drop `.US`/`.V`/other.
  Fuzzy at edges ❓ — the ~1,248 "other" (money market/commodities) and the 6-letter=fx assumption.
- **Bonds = two series** ✅: yield (`10YUSY.B`) + price (`10YUSP.B`), both `AssetClass='bonds'`,
  distinguished by ticker (Y vs P). Exchange suffix `.B` stripped in transform → `10YUSY` / `10YUSP`.

---

## 3. Universe (ticker registry)

- `universe.csv` columns: `Ticker` (canonical), `Market`, `stooq_ticker`, `yahoo_ticker`.
- Old seed: built from Stooq `markets.csv`, deduped by **market tier** (equities outrank
  crypto/fx/bonds so an equity never gets shadowed by a colliding crypto symbol).
- `BANNED_MARKETS = {cryptocurrencies}` — never seeded, fetched, or stored.
- 🔶 With the new clean asset-class split, the universe model may change — the canonical ticker and
  per-asset-class identity need redesign (separate equity vs bond/fx/index namespaces?).

---

## 4. ETL / cleaning (known facts)

- **Crypto:** banned at the source; never enters the DB.
- **Stooq stocks:** dropped (unadjusted → unreliable). Keep only bonds/fx/indices.
- **Delisted:** track per-ticker fetch success/failure; don't keep re-querying dead Yahoo tickers.
- **Column normalization:** sources use different names (Stooq `<OPEN>...`, Yahoo `Open...`) →
  canonical column set per dataset.
- **Side-effect discipline** ✅ (architecture goal): acquire → land raw (immutable) → transform
  (pure/testable) → validate → load. No side effects interleaved with logic.

---

## 5. Directory structure ✅ (`./data`)

**Provider-first**: each provider owns one self-contained subtree and creates only its own folder.
The DB is shared and lives outside every provider tree. A provider never touches another's data.

```
data/
├── investalyze.duckdb          # the one DB — shared, outside provider trees
├── universe.csv                # cross-provider registry (or move into DB) 🔶
├── yahoo/
│   ├── raw/                     # immutable landed source artifacts (never mutated)
│   ├── processed/               # canonical parquet, regenerable from raw (safe to wipe)
│   └── state/                   # markers, per-ticker success/failure, resume cursors
├── simfin/
│   ├── raw/   processed/   state/
└── stooq/
    ├── raw/   processed/   state/
```

- **Dirs are scaffolded once, up front** ✅ by `orchestrator.create_data_dirs(config)` (terminal:
  `python -m investalyze.ingest setup`). It builds `data/<provider>/{raw,processed,state}/` for every
  **registered** provider, idempotently — run it, then drop the manual source files into `raw/`.
  (Storage stays config-free plumbing; the orchestrator is the one place that reads config.)
- `raw/` immutable · `processed/` regenerable · `state/` holds all resume/freshness/log metadata
  (kept out of raw & processed so neither carries side-effect state — fixes the old scattered-JSON mess).

> The investalyze `data` dir is **decoupled** from the old project's data — fresh start, no inherited files.

---

## 6. DB architecture 🔶 (evolving)

- **Engine:** DuckDB. **Target file:** `data/investalyze.duckdb` ✅ (fresh, decoupled from `irp.duckdb`).
- **Price data is split by role, not unified** ✅ — main vs secondary, one source per table:

  | Table | Source | Role | Columns |
  |-------|--------|------|---------|
  | `prices` ✅ | Yahoo | **main** — prediction target, joins fundamentals | Ticker, Date, O, H, L, C, V, AC |
  | **`market_data`** ✅ | Stooq | **secondary / features** — bonds, indices, currencies | `Ticker, Date, O, H, L, C, AssetClass` |
  | `income`/`balance`/`cashflow` ✅ | SimFin | **fundamentals** — wide statements | source cols + Market, Period, IsRestated, Src, SrcId |
  | `companies` ✅ | SimFin | **metadata** — name/sector/industry/… | joined to industries |

  `dividends(Ticker, Date, Dividend)` and `splits(Ticker, Date, Ratio)` carry the events. `AC` is
  our adjusted close, computed from raw close + events (not Yahoo's, not stored adjusted) and
  validated against Yahoo's `Adj Close` at 0.1% during ingest.

  - **Short column names** ✅: `O/H/L/C` (not Open/High/Low/Close).
  - `market_data` has **no volume** (always 0 for these instruments).
  - **`AssetClass`** ∈ {bonds, currencies, indices} — derived from the Stooq source category dir.
  - **Bonds = two tickers**: yield (`10YUSY`) + price (`10YUSP`), both `AssetClass='bonds'`.
- **Fundamentals** ✅: one table per statement, both vintages in a boolean `IsRestated` column (the old
  `_restated` shadow tables are retired). Point-in-time via `Publish Date`/`Restated Date` + `IsRestated`.
- **Still to design:**
  - Per-dataset **column contract** (no silently drifting wide SimFin tables).

---

## 7. Open decisions

| # | Decision | Status |
|---|----------|--------|
| 1 | Store raw vs adjusted Yahoo prices | ✅ raw OHLCV + our derived `AC` (see §2.1) |
| 2 | Stooq scope: keep `world/{bonds,currencies,indices,stooq stocks indices}`; drop us stocks/etfs, money market, crypto | 🔶 |
| 3 | Point-in-time fundamentals table design | 🔶 |
| 4 | Universe / canonical-ticker model under clean asset split | 🔶 |
| 5 | `market_data` keys: how yield vs price bonds are distinguished beyond ticker (a Quote flag?) | ❓ |
| 6 | Per-ticker fetch success/failure tracking | ✅ Yahoo `state/blacklist.csv` (Stooq/SimFin TBD) |
