# investalyze

A stock research workbench: pull market data from several vendors into one DuckDB, check it,
correct it, and work through it in a Dash UI and in notebooks.

Everything is local. One DuckDB file (`data/investalyze.duckdb`, currently around 27M price rows
and 437k rows per financial statement) is the single source of truth, and every tool reads or
writes that file directly. There is no service to deploy and no API to call.

## Install

The app runs from its own folder, apart from this repo. `install.sh` installs one version (a git tag) from GitHub in a
Python environment in `~/.local/share/md_apps/investalyze/` (or `$INVESTALYZE_HOME`). It needs uv and curl.

```bash
./install.sh v0.5.0                                      # also to update: an update replaces only env/
~/.local/share/md_apps/investalyze/env/bin/investalyze   # the UI, at http://127.0.0.1:8050
```

The app folder holds the settings: `investalyze.toml` (`data_dir`, the port, the debug mode), `ingest.toml`,
`backtest.toml`, `cleaning.toml`, `stooq_tickers.toml` (the first install makes them from this repo) and `.env` (the
SimFin key and the SEC contact address). `data_dir` sets the data folder: the database, the provider downloads, the backtests, the factor library
(`factors/`) and the quality log (`quality_log.toml`). Put it where your backups are made; empty, it is `data/` in the
app folder. The control panel runs its commands with the app's own Python, in the app folder.

## Development

Requires Python 3.13. The environment lives in `.venv`. This repo is the app folder of the development copy: copy
`investalyze.example.toml` to `investalyze.toml` (not in git) and give it its own port, for example 18050 with
`debug = true`, so it can run next to the installed app.

```bash
uv sync                                # install into .venv
python -m investalyze.ingest setup     # scaffold the data dirs, once
```

Some sources need a manual download (Stooq is captcha-protected); `ingest.toml` carries the URLs
and the steps. SimFin needs an API key and SEC a contact address, both in `.env`: `SIMFIN_API_KEY=...` and
`SEC_CONTACT_EMAIL=...` (SEC refuses requests without a real contact address).

`torch` is pinned to 2.5.1+cu121 on a dedicated index: newer CUDA builds dropped the Pascal
compute capability this machine's GPUs need. Leave that pin alone.

```bash
python -m investalyze.ingest           # load the full history
python -m investalyze.ingest --update  # daily incremental
python -m investalyze.apps             # the UI, at the port of investalyze.toml
```

Start with the UI. Every CLI below is also a button in its Control Panel.

## What is where

| Path | Holds |
|------|-------|
| `src/investalyze/ingest/` | Provider adapters (Yahoo, SimFin, Stooq), orchestration, housekeeping, listings scans. |
| `src/investalyze/quality/` | Anomaly checks over prices, corporate actions and fundamentals. Findings only. |
| `src/investalyze/cleaning/` | Persistent manual corrections, replayed after any reload. |
| `src/investalyze/analysis/` | Snapshot factors for the Screener, point-in-time readers, return panels, price segments, encodings, dimensionality reduction. |
| `src/investalyze/research/` | The factor library, expression language, data fields, diagnostics, IC analysis, factor portfolio and timing rule backtests. |
| `src/investalyze/backtest/` | The event-driven engine, costs and limits, performance statistics, reports and run storage. |
| `src/investalyze/apps/` | The Dash UI: control panel, screener, data quality, stock, factors, backtest, database. |
| `notebooks/` | Exploration, plus `9999_data_quirks.ipynb`, the running findings-and-decisions log. |
| `experiments/` | Model work over saved universes (classification, ranking, forecasting). |
| `tests/` | `pytest`, 922 tests. |
| `data/factors/` | The factor library: one TOML file per factor (in the data folder, not in git). |
| `docs/manual.md` | The user guide: how to actually run all of the above. |
| `docs/ingest_data.md` | Data reference: every table, column and provider decision. |
| `docs/backtest_metrics.md` | Every backtest formula (IC, Sharpe, fill price, ...), traced to its function. |
| `docs/backlog.md` | Running list of what is worth coming back to, most valuable first. |

Config lives in the app folder (the repo root in development): `ingest.toml` (providers, blacklist policy),
`cleaning.toml` (the manual fix list), `backtest.toml`, `stooq_tickers.toml`. The triaged anomaly findings are in
`quality_log.toml` in the data folder.

## How the pieces fit

```
vendors -> ingest -> DuckDB -> quality  -> anomalies table -> Data Quality page
                       |                                          |
                       |                                     cleaning.toml -> cleaning apply
                       |
                       +-> analysis/factors -> Screener -> data/universes/<name>.csv
                       |                                        |
                       |                        +---------------+---------------+
                       |                        |               |               |
                       |                   Stock page      experiments      ingest --universe
                       |
                       +-> factors/ -> research -> Factors page (diagnostics, IC analysis)
                                                -> Backtest page (factor portfolios, timing rules) -> backtest_runs
```

Two rules that explain most of the design:

- **Ingest never deletes.** Every write is a merge-upsert, so a full reload resurrects any row
  you removed by hand. Corrections therefore live in `cleaning.toml` and get replayed.
- **Quality never corrects.** Checks write findings to the `anomalies` table and touch nothing
  else. Deciding what to do about a finding is a separate, manual step.

DuckDB allows a single writer. A running job locks the DB, and an open notebook kernel will
block jobs started from the UI.

## Tests

```bash
python -m pytest -q
```

The handful of tests that read the live DB skip themselves when it is absent or write-locked.
