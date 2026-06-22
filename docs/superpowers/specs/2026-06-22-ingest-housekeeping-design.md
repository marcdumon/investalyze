# Ingest housekeeping — design

## Context

The Yahoo provider already blacklists tickers that return no data (`data/yahoo/state/blacklist.csv`,
written by `providers/yahoo/provider.py:run`), so they're skipped on later ingest runs. That blacklist
is a flat list of tickers with no attempt tracking — a ticker blacklisted once stays blacklisted
forever, with no way to notice if it starts returning data again (e.g. a relisting), and no way to
stop bothering with tickers that are clearly gone for good.

This introduces a `housekeeping` command: a small, extensible runner for maintenance tasks against
the ingest state, separate from the regular provider ingest runs. First task: recheck the Yahoo
blacklist, age out permanently-dead tickers, and keep the master ticker list pruned.

## Scope

Yahoo only. It's the only provider with a blacklist today; other providers can grow one later if
needed, following the same pattern.

## State files (`data/yahoo/state/`)

**`blacklist.csv`** — tickers currently believed to have no data, still eligible for retry.
Columns: `ticker, attempts, first_blacklisted, last_checked` (dates as ISO `YYYY-MM-DD`).

- Written first by the regular yahoo `run()` when a ticker returns no data: `attempts=1`,
  `first_blacklisted=today`, `last_checked=today`.
- Updated by housekeeping's recheck on each subsequent failed retry: `attempts += 1`,
  `last_checked=today`.

**`dead.csv`** (new) — tickers that exhausted `blacklist_max_attempts` retries. Never retried again.
Columns: `ticker, attempts, first_blacklisted, died_on`.

- A ticker moves here (and is removed from `blacklist.csv`) the moment its retry count reaches the
  configured max.

**`data/yahoo/raw/ticker.csv`** (existing master universe list) — pruned by housekeeping: any ticker
present in `blacklist.csv` or `dead.csv` is removed. Defensive — this file may be refreshed
externally (e.g. regenerated from a markets list) and could reintroduce known-bad tickers.

## Config

`ingest.toml [yahoo]` gains `blacklist_max_attempts = 5` (retries before permanent removal).

## Module layout

**`src/investalyze/ingest/housekeeping.py`** (new) — mirrors `orchestrator.py`'s shape:

```python
HousekeepingTask = Callable[..., dict]
HOUSEKEEPING_TASKS: dict[str, HousekeepingTask] = {
    'yahoo-blacklist': yahoo.recheck_blacklist,
}

def run_housekeeping(config: Config, tasks: Sequence[str] | None = None) -> dict[str, dict]:
    """Run the selected housekeeping tasks against the shared DB. Returns {task: result}."""
```

Opens one DB connection (via `storage.connect`), dispatches to each selected task, closes when done
— same lifecycle as `orchestrator.run`. Adding a future housekeeping task = one new function + one
registry entry, no changes to the runner itself.

**`providers/yahoo/provider.py`** gains `recheck_blacklist(con, data_root, settings) -> dict`:

1. Load `blacklist.csv` (empty dict if absent).
2. Batch-fetch every listed ticker via yfinance, reusing the existing `_fetch`/`_chunk` helpers and
   the same `batch_size`/`sleep` settings as the main run.
3. **Data found** → remove from `blacklist.csv`; add the ticker back into `ticker.csv` (the master
   list) so the next regular ingest run picks it up and ingests it normally. Housekeeping itself does
   not write price/dividend/split rows — keeps it single-purpose (state hygiene only). Trade-off:
   one redundant full-history fetch (the data just checked is discarded, re-fetched on the next
   regular run) in exchange for not duplicating the ingest-write path here.
4. **Still no data** → `attempts += 1`, `last_checked = today`. If `attempts >= blacklist_max_attempts`,
   move the row to `dead.csv` (with `died_on = today`) instead of rewriting it back to `blacklist.csv`.
5. Rewrite `ticker.csv` dropping anything now in `blacklist.csv` or `dead.csv`.
6. Return a result dict, e.g. `{'rechecked': n, 'revived': n, 'died': n}`, for the CLI to log.

## CLI

`__main__.py`: add `'housekeeping'` to the existing `command` positional's `choices` (alongside
`'setup'`). Add a repeatable `-t/--task` flag (default: all registered tasks), mirroring the existing
`-p/--provider` flag, so:

```
python -m investalyze.ingest housekeeping                  # all tasks
python -m investalyze.ingest housekeeping -t yahoo-blacklist
```

On `command == 'housekeeping'`, call `housekeeping.run_housekeeping(cfg, args.tasks)` and log each
task's result dict, then return (same shape as the existing `setup` branch).

## Out of scope

- Retesting `dead.csv` tickers — permanent means permanent; if this needs revisiting later it's a
  separate, deliberate task.
- Any provider other than Yahoo.
- Changing the regular ingest run's blacklist-write behavior beyond the schema upgrade described
  above (it still blacklists on first failure exactly as today, just with richer columns).

## Verification

- Unit-level: run `recheck_blacklist` against a fixture `blacklist.csv` with a mix of revivable /
  still-dead / attempts-at-threshold tickers (mock the yfinance fetch), assert the three output files
  end up in the right state.
- End-to-end: `python -m investalyze.ingest housekeeping` against the real `data/` dir, confirm
  `blacklist.csv`/`dead.csv`/`ticker.csv` are consistent (no ticker in more than one of
  `ticker.csv ∩ blacklist.csv`, `ticker.csv ∩ dead.csv`, `blacklist.csv ∩ dead.csv`) and that
  `attempts`/`last_checked`/`died_on` move as expected across two consecutive runs.
