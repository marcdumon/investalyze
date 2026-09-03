# investalyze

A stock research workbench: pull market data from several vendors into one DuckDB, check it,
correct it, and work through it in a Dash UI and in notebooks.

Everything is local. One DuckDB file (`data/investalyze.duckdb`, currently around 27M price rows
and 437k rows per financial statement) is the single source of truth, and every tool reads or
writes that file directly. There is no service to deploy and no API to call.

## Getting started

Requires Python 3.13. The environment lives in `.venv`.

```bash
uv sync                                # install into .venv
python -m investalyze.ingest setup     # scaffold the data dirs, once
```

Some sources need a manual download (Stooq is captcha-protected); `ingest.toml` carries the URLs
and the steps. SimFin needs an API key in the environment.

`torch` is pinned to 2.5.1+cu121 on a dedicated index: newer CUDA builds dropped the Pascal
compute capability this machine's GPUs need. Leave that pin alone.

```bash
python -m investalyze.ingest           # load the full history
python -m investalyze.ingest --update  # daily incremental
python -m investalyze.apps             # the UI, at http://127.0.0.1:8050
```

Start with the UI. Every CLI below is also a button in its Control Panel.

## What is where

| Path | Holds |
|------|-------|
| `src/investalyze/ingest/` | Provider adapters (Yahoo, SimFin, Stooq), orchestration, housekeeping, listings scans. |
| `src/investalyze/quality/` | Anomaly checks over prices, corporate actions and fundamentals. Findings only. |
| `src/investalyze/cleaning/` | Persistent manual corrections, replayed after any reload. |
| `src/investalyze/analysis/` | Factor computation (snapshot and point-in-time), return panels, price segments, encodings, dimensionality reduction. |
| `src/investalyze/backtest/` | Signals, portfolio construction, the event-driven engine, factor evaluation, walk-forward. |
| `src/investalyze/apps/` | The Dash UI: control panel, screener (with the factor editor), data quality, stock, backtest. |
| `notebooks/` | Exploration, plus `9999_data_quirks.ipynb`, the running findings-and-decisions log. |
| `experiments/` | Model work over saved universes (classification, ranking, forecasting). |
| `tests/` | `pytest`, 899 tests. |
| `factors.toml` | Your own factor definitions, merged into the factor library at import. |
| `docs/manual.md` | The user guide: how to actually run all of the above. |
| `docs/ingest_data.md` | Data reference: every table, column and provider decision. |

Config lives at the repo root: `ingest.toml` (paths, providers, blacklist policy), `cleaning.toml`
(the manual fix list), `quality_log.toml` (triaged anomaly findings).

## How the pieces fit

```
vendors -> ingest -> DuckDB -> quality  -> anomalies table -> Data Quality page
                       |                                          |
                       |                                     cleaning.toml -> cleaning apply
                       |
                       +-> analysis/factors -> Screener -> data/universes/<name>.csv
                                                                |
                                                +---------------+---------------+
                                                |               |               |
                                           Stock page      experiments      ingest --universe
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
