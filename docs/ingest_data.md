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
| **SimFin** (bulk REST) | Fundamentals | income, balance, cashflow, derived — **as-filed + restated** |
| **Stooq** (manual download) | Bonds, currencies, indices | OHLC(V) |

Design rule ✅: a provider is a self-contained module. Adding a 4th provider must not touch the
others. More providers expected later.

> Change from old `dataload`: Stooq no longer supplies **stock** prices (its stock data isn't
> dividend-adjusted → poor quality). Yahoo owns all equities; Stooq is scoped to non-equity.

---

## 2. Provider detail

### 2.1 Yahoo — stock prices ✅
- **Acquire:** `yfinance`. Prices via batched `yf.download(...)` (old batch size 75, rate-limited
  sleeps); dividends/splits via `yf.Ticker(sym).actions`.
- **Fields:** `Open, High, Low, Close, Volume`; actions → `dividend`, `split` events.
- **Adjustment** 🔶: old code used `auto_adjust=True` (stored adjusted prices, raw lost).
  Decision pending — store **raw** OHLCV + apply adjustments downstream, or store adjusted?
- **Symbol mapping:** canonical ticker → Yahoo symbol can differ (e.g. preferred `ARRY_A` →
  `ARRY-PA`). For equities the canonical usually equals the Yahoo symbol.
- **Known issues** ❓: delisted tickers (e.g. SimFin lists `ARRY_delisted`) — Yahoo gives no prices
  for delisted names. Need a per-ticker success/failure record to stop re-querying dead tickers.

### 2.2 SimFin — fundamentals ✅
- **Acquire:** bulk ZIPs from SimFin REST API (auth → presigned S3 redirect), refreshed by file age
  (`refresh_days_fundamentals`, `refresh_days_meta`). No incremental feed.
- **Statements:** `income`, `balance`, `cashflow`, plus `derived` (ratios).
- **Vintages** ✅: **as-reported** (original figures per fiscal period) AND **restated** (latest
  revised figures). Both wanted.
- **Granularity:** Annual (`A`) + Quarterly (`Q`).
- **Markets:** `us`, `de`.
- **Metadata:** `companies` (name, sector/industry, business summary, CIK, ISIN, main currency,
  employees, fiscal-year-end month) joined to `industries`.
- **Temporal columns** ✅: rows carry `Fiscal Year`, `Fiscal Period`, `Report Date`, `Publish Date`,
  `Restated Date` → **point-in-time is feasible** (≥2 vintages: original-as-published + latest).
- **Format:** semicolon-delimited CSVs inside the zips; wide, source-defined columns.

### 2.3 Stooq — bonds, currencies, indices ✅
- **Acquire:** NOT programmatic. User manually downloads from stooq.com and drops files in raw dir.
  **Two inputs, two read paths** ✅ (both → `market_data`, both saved via the merge/upsert so they converge):
  | Input | Use | Format | AssetClass from |
  |-------|-----|--------|-----------------|
  | **`d_world_txt.zip`** (bulk) | full history | folder tree `world/<category>/[sub]/<ticker>.txt` | **folder name** (`asset_class_for`) |
  | **`data_d.txt`** (flat) | daily update | one file, all instruments, no category | **ticker pattern** (`asset_class_from_ticker`) |
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

- **First action of every provider** ✅: ensure its own `data/<provider>/{raw,processed,state}/`
  exists (idempotent setup). It creates nothing outside that subtree.
- `raw/` immutable · `processed/` regenerable · `state/` holds all resume/freshness/log metadata
  (kept out of raw & processed so neither carries side-effect state — fixes the old scattered-JSON mess).

> The investalyze `data` dir is **decoupled** from the old project's data — fresh start, no inherited files.

---

## 6. DB architecture 🔶 (evolving)

- **Engine:** DuckDB. **Target file:** `data/investalyze.duckdb` ✅ (fresh, decoupled from `irp.duckdb`).
- **Price data is split by role, not unified** ✅ — main vs secondary, one source per table:

  | Table | Source | Role | Columns |
  |-------|--------|------|---------|
  | stock prices 🔶 | Yahoo | **main** — prediction target, joins fundamentals | Ticker, Date, O/H/L/C, **V**, … (TBD) |
  | **`market_data`** ✅ | Stooq | **secondary / features** — bonds, indices, currencies | `Ticker, Date, O, H, L, C, AssetClass` |

  - **Short column names** ✅: `O/H/L/C` (not Open/High/Low/Close).
  - `market_data` has **no volume** (always 0 for these instruments).
  - **`AssetClass`** ∈ {bonds, currencies, indices} — derived from the Stooq source category dir.
  - **Bonds = two tickers**: yield (`10YUSY`) + price (`10YUSP`), both `AssetClass='bonds'`.
- **Still to design:**
  - First-class **point-in-time fundamentals** (retire `_restated` shadow tables; use Publish/Restated dates).
  - Per-dataset **column contract** (no silently drifting wide tables).

---

## 7. Open decisions

| # | Decision | Status |
|---|----------|--------|
| 1 | Store raw vs adjusted Yahoo prices | 🔶 |
| 2 | Stooq scope: keep `world/{bonds,currencies,indices,stooq stocks indices}`; drop us stocks/etfs, money market, crypto | 🔶 |
| 3 | Point-in-time fundamentals table design | 🔶 |
| 4 | Universe / canonical-ticker model under clean asset split | 🔶 |
| 5 | `market_data` keys: how yield vs price bonds are distinguished beyond ticker (a Quote flag?) | ❓ |
| 6 | Per-ticker fetch success/failure tracking | 🔶 |
