# Ingest Housekeeping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `housekeeping` command that rechecks Yahoo-blacklisted tickers, revives ones that
return data again, permanently retires chronic failures, and keeps `ticker.csv` pruned.

**Architecture:** Upgrade the existing flat `blacklist.csv` (yahoo provider) to a richer schema with
attempt-tracking columns; add a `recheck_blacklist()` function to the yahoo provider that retries,
promotes, and ages out entries; add a small `housekeeping.py` runner (mirrors `orchestrator.py`'s
registry pattern) that dispatches named tasks; wire a new `housekeeping` CLI command.

**Tech Stack:** Python 3.13, pandas, duckdb, pytest (existing stack — no new dependencies).

## Global Constraints

- Scope: Yahoo provider only — no other provider gets a blacklist in this plan.
- `blacklist.csv` columns: `ticker, market, attempts, first_blacklisted, last_checked`.
- `dead.csv` (new) columns: `ticker, attempts, first_blacklisted, died_on`.
- `blacklist_max_attempts` config key under `ingest.toml [yahoo]`, default value `5`.
- Retry cadence: every housekeeping run retries every non-dead blacklisted ticker (no staleness
  check).
- Revived tickers are added back to `data/yahoo/raw/ticker.csv`; housekeeping does NOT ingest data
  itself — the next regular ingest run picks revived tickers up.
- `ticker.csv` is pruned of every ticker still in `blacklist.csv` or now in `dead.csv` on every
  housekeeping run.
- CLI: new `housekeeping` command (alongside existing `setup`), with a repeatable `-t/--task` flag
  (default: all registered tasks), mirroring the existing `-p/--provider` flag.
- No fallback defaults inside provider settings dicts (existing convention) — a missing config key
  raises `KeyError`, exactly like `run()` today.

---

### Task 1: Upgrade the blacklist schema in the main yahoo `run()`

**Files:**
- Modify: `src/investalyze/ingest/providers/yahoo/provider.py:1-22` (imports + constants)
- Modify: `src/investalyze/ingest/providers/yahoo/provider.py:105-126` (add `_load_blacklist`/`_load_dead` helpers)
- Modify: `src/investalyze/ingest/providers/yahoo/provider.py:140-213` (`run()`)
- Test: `tests/test_yahoo_run.py`

**Interfaces:**
- Produces: `provider._BLACKLIST_COLS: list[str]`, `provider._DEAD_COLS: list[str]`,
  `provider._load_blacklist(path: Path) -> pd.DataFrame`, `provider._load_dead(path: Path) -> pd.DataFrame`
  — Task 2 reuses all four.
- Consumes: nothing new from outside this file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_yahoo_run.py` (the existing `test_run_skips_empty_ticker_and_blacklists_it` test
gets replaced — richer assertions on the new columns — and one new test for dead-ticker skipping is
added):

```python
def test_run_skips_empty_ticker_and_blacklists_it(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'DEAD')
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'DEAD': pd.DataFrame()})
    con = duckdb.connect()
    n = provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert n == 0
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'blacklist.csv')
    assert blacklist.loc[0, 'ticker'] == 'DEAD'
    assert blacklist.loc[0, 'market'] == 'nyse'
    assert blacklist.loc[0, 'attempts'] == 1
    assert blacklist.loc[0, 'first_blacklisted'] == blacklist.loc[0, 'last_checked']


def test_run_skips_tickers_already_dead(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'GONE', 'AAA')
    state_dir = tmp_path / 'yahoo' / 'state'
    state_dir.mkdir(parents=True)
    pd.DataFrame([{'ticker': 'GONE', 'attempts': 5, 'first_blacklisted': '2024-01-01', 'died_on': '2024-02-01'}]
                 ).to_csv(state_dir / 'dead.csv', index=False)
    fetched: list[str] = []
    def fake_fetch(syms, **k):
        fetched.extend(syms)
        return {'AAA': _single('AAA', 10.0, 10.0)}
    monkeypatch.setattr(provider, '_fetch', fake_fetch)
    con = duckdb.connect()
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert 'GONE' not in fetched
    assert 'AAA' in fetched
```

`_ticker_csv` already writes a `market` column (`nyse`) for every symbol — no change needed there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_yahoo_run.py -v`
Expected: `test_run_skips_empty_ticker_and_blacklists_it` FAILs (no `market`/`attempts` columns yet —
`KeyError` or `pd.read_csv` only has a `ticker` column). `test_run_skips_tickers_already_dead` FAILs
(`GONE` gets fetched because `dead.csv` isn't read at all).

- [ ] **Step 3: Add the schema constants + load helpers**

In `provider.py`, the existing import block groups bare `import`s before `from`-imports within
each group. Replace:

```python
import logging
import time
from pathlib import Path
```

with:

```python
import logging
import time
from datetime import date
from pathlib import Path
```

Add the schema constants right after the existing `_PRICES, _DIVS, _SPLITS = ...` / `_KEY = ...`
constants:

```python
_BLACKLIST_COLS = ['ticker', 'market', 'attempts', 'first_blacklisted', 'last_checked']
_DEAD_COLS = ['ticker', 'attempts', 'first_blacklisted', 'died_on']
```

Add two helpers right after `_load_last_dates` (before `_load_events`):

```python
def _load_blacklist(path: Path) -> pd.DataFrame:
    """Blacklist records (empty frame with the right columns if the file is absent)."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=_BLACKLIST_COLS)


def _load_dead(path: Path) -> pd.DataFrame:
    """Permanently-dead records (empty frame with the right columns if the file is absent)."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=_DEAD_COLS)
```

- [ ] **Step 4: Rewrite the blacklist load/write in `run()`**

Replace:

```python
    symbols = pd.read_csv(raw_dir / settings['ticker_file'])['ticker'].tolist()
    blacklist_file = state_dir / 'blacklist.csv'
    blacklist = set(pd.read_csv(blacklist_file)['ticker']) if blacklist_file.exists() else set()
    done = _load_existing_tickers(con, _PRICES) if not update else set()
    todo = [s for s in symbols if s not in blacklist and s not in done]
```

with:

```python
    ticker_df = pd.read_csv(raw_dir / settings['ticker_file'])
    symbols = ticker_df['ticker'].tolist()
    market_by_ticker = dict(zip(ticker_df['ticker'], ticker_df['market']))
    blacklist_file = state_dir / 'blacklist.csv'
    dead_file = state_dir / 'dead.csv'
    orig_blacklist_df = _load_blacklist(blacklist_file)
    blacklist_tickers = set(orig_blacklist_df['ticker'])
    dead_tickers = set(_load_dead(dead_file)['ticker'])
    done = _load_existing_tickers(con, _PRICES) if not update else set()
    todo = [s for s in symbols if s not in blacklist_tickers and s not in dead_tickers and s not in done]
```

Replace the per-batch write:

```python
        if newly_blacklisted:
            pd.DataFrame({'ticker': sorted(blacklist | set(newly_blacklisted))}).to_csv(blacklist_file, index=False)
```

with:

```python
        if newly_blacklisted:
            today = date.today().isoformat()
            new_rows = pd.DataFrame({
                'ticker': newly_blacklisted,
                'market': [market_by_ticker.get(s, '') for s in newly_blacklisted],
                'attempts': 1,
                'first_blacklisted': today,
                'last_checked': today,
            }).drop_duplicates('ticker', keep='first')
            pd.concat([orig_blacklist_df, new_rows], ignore_index=True).sort_values('ticker').to_csv(blacklist_file, index=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_yahoo_run.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the full suite to check nothing else broke**

Run: `.venv/bin/pytest -q`
Expected: all PASS (no other test reads `blacklist.csv` with the old single-column assumption).

- [ ] **Step 7: Commit**

```bash
git add src/investalyze/ingest/providers/yahoo/provider.py tests/test_yahoo_run.py
git commit -m "feat(ingest): track blacklist attempts and skip permanently-dead tickers"
```

---

### Task 2: Add `recheck_blacklist()` to the yahoo provider

**Files:**
- Modify: `src/investalyze/ingest/providers/yahoo/provider.py` (new function, after `run()`/`_prepare_ticker`)
- Modify: `ingest.toml` (add `blacklist_max_attempts`)
- Modify: `docs/ingest_data.md` (state bullet under 2.1 Yahoo)
- Test: `tests/test_yahoo_housekeeping.py` (new)

**Interfaces:**
- Consumes: `provider._BLACKLIST_COLS`, `provider._DEAD_COLS`, `provider._load_blacklist`,
  `provider._load_dead`, `provider._fetch`, `provider._chunk` (all from Task 1 / existing code).
- Produces: `provider.recheck_blacklist(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict) -> dict[str, int]`
  — Task 3's registry calls this directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_yahoo_housekeeping.py`:

```python
"""Tests for the Yahoo blacklist recheck (housekeeping)."""
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest.providers.yahoo import provider

_SETTINGS = {'ticker_file': 'ticker.csv', 'batch_size': 10, 'sleep': 0, 'blacklist_max_attempts': 3}


def _state(tmp_path: Path, blacklist_rows: list[dict]) -> None:
    state_dir = tmp_path / 'yahoo' / 'state'
    state_dir.mkdir(parents=True)
    pd.DataFrame(blacklist_rows).to_csv(state_dir / 'blacklist.csv', index=False)


def _ticker_csv(tmp_path: Path, *rows: tuple[str, str]) -> None:
    raw = tmp_path / 'yahoo' / 'raw'
    raw.mkdir(parents=True)
    pd.DataFrame(list(rows), columns=['ticker', 'market']).to_csv(raw / 'ticker.csv', index=False)


def _single(symbol: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp('2024-03-21')], name='Date')
    return pd.DataFrame({'Open': [1.0], 'High': [1.0], 'Low': [1.0], 'Close': [1.0],
                         'Adj Close': [1.0], 'Volume': [100], 'Dividends': [0.0], 'Stock Splits': [0.0]}, index=idx)


def test_recheck_returns_zeros_when_no_blacklist(tmp_path):
    con = duckdb.connect()
    result = provider.recheck_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 0, 'revived': 0, 'died': 0}


def test_recheck_revives_ticker_with_data(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'BACK', 'market': 'nyse', 'attempts': 1,
                       'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}])
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'BACK': _single('BACK')})
    con = duckdb.connect()
    result = provider.recheck_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 1, 'revived': 1, 'died': 0}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'blacklist.csv')
    assert blacklist.empty
    tickers = pd.read_csv(tmp_path / 'yahoo' / 'raw' / 'ticker.csv')
    assert set(tickers['ticker']) == {'AAA', 'BACK'}
    assert tickers.loc[tickers['ticker'] == 'BACK', 'market'].iloc[0] == 'nyse'


def test_recheck_increments_attempts_when_still_empty(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'QUIET', 'market': 'nyse', 'attempts': 1,
                       'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}])
    _ticker_csv(tmp_path, ('QUIET', 'nyse'))
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'QUIET': pd.DataFrame()})
    con = duckdb.connect()
    result = provider.recheck_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 1, 'revived': 0, 'died': 0}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'blacklist.csv')
    assert blacklist.loc[0, 'attempts'] == 2
    assert blacklist.loc[0, 'last_checked'] != '2024-01-01'
    tickers = pd.read_csv(tmp_path / 'yahoo' / 'raw' / 'ticker.csv')
    assert 'QUIET' not in set(tickers['ticker'])


def test_recheck_moves_to_dead_after_max_attempts(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'GONE', 'market': 'nyse', 'attempts': 2,
                       'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-05'}])
    _ticker_csv(tmp_path, ('GONE', 'nyse'))
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'GONE': pd.DataFrame()})
    con = duckdb.connect()
    result = provider.recheck_blacklist(con, tmp_path, _SETTINGS)  # max_attempts=3, this is the 3rd try
    assert result == {'rechecked': 1, 'revived': 0, 'died': 1}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'blacklist.csv')
    assert blacklist.empty
    dead = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'dead.csv')
    assert dead.loc[0, 'ticker'] == 'GONE'
    assert dead.loc[0, 'attempts'] == 3
    tickers = pd.read_csv(tmp_path / 'yahoo' / 'raw' / 'ticker.csv')
    assert 'GONE' not in set(tickers['ticker'])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_yahoo_housekeeping.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'recheck_blacklist'`.

- [ ] **Step 3: Implement `recheck_blacklist`**

Add to `provider.py`, after `_prepare_ticker`:

```python
def recheck_blacklist(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict) -> dict:
    """Retry every blacklisted ticker; revive successes, age out chronic failures, prune ticker.csv.

    `con` is unused here — kept so this matches the (con, data_root, settings) shape every
    housekeeping task is dispatched with. `settings` is the `[yahoo]` config: uses `ticker_file`,
    `batch_size`, `sleep` (same as `run`) plus `blacklist_max_attempts` (no fallback — missing
    raises `KeyError`).
    """
    raw_dir = data_root / 'yahoo' / 'raw'
    state_dir = data_root / 'yahoo' / 'state'
    blacklist_file = state_dir / 'blacklist.csv'
    dead_file = state_dir / 'dead.csv'
    ticker_file = raw_dir / settings['ticker_file']

    blacklist_df = _load_blacklist(blacklist_file)
    if blacklist_df.empty:
        return {'rechecked': 0, 'revived': 0, 'died': 0}

    max_attempts = settings['blacklist_max_attempts']
    today = date.today().isoformat()
    tickers = blacklist_df['ticker'].tolist()
    batches = _chunk(tickers, settings['batch_size'])

    revived_rows: list[dict] = []
    still_blacklisted: list[dict] = []
    died_rows: list[dict] = []
    for i, batch in enumerate(batches):
        frames = _fetch(batch, start=None)
        records = blacklist_df[blacklist_df['ticker'].isin(batch)].to_dict('records')
        for record in records:
            frame = frames.get(record['ticker'], pd.DataFrame())
            if not frame.empty:
                revived_rows.append(record)
                continue
            record['attempts'] += 1
            record['last_checked'] = today
            if record['attempts'] >= max_attempts:
                died_rows.append({'ticker': record['ticker'], 'attempts': record['attempts'],
                                  'first_blacklisted': record['first_blacklisted'], 'died_on': today})
            else:
                still_blacklisted.append(record)
        if settings['sleep'] and i < len(batches) - 1:
            time.sleep(settings['sleep'])

    pd.DataFrame(still_blacklisted, columns=_BLACKLIST_COLS).to_csv(blacklist_file, index=False)
    if died_rows:
        dead_df = pd.concat([_load_dead(dead_file), pd.DataFrame(died_rows, columns=_DEAD_COLS)], ignore_index=True)
        dead_df.to_csv(dead_file, index=False)

    if ticker_file.exists():
        ticker_df = pd.read_csv(ticker_file)
        if revived_rows:
            revived_df = pd.DataFrame([{'ticker': r['ticker'], 'market': r['market']} for r in revived_rows])
            ticker_df = pd.concat([ticker_df, revived_df], ignore_index=True).drop_duplicates('ticker')
        exclude = {r['ticker'] for r in still_blacklisted} | {r['ticker'] for r in died_rows}
        ticker_df = ticker_df[~ticker_df['ticker'].isin(exclude)]
        ticker_df.to_csv(ticker_file, index=False)

    return {'rechecked': len(tickers), 'revived': len(revived_rows), 'died': len(died_rows)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_yahoo_housekeeping.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the config key**

In `ingest.toml`, under the `[yahoo]` section, add:

```toml
blacklist_max_attempts = 5      # retries (via housekeeping) before a ticker is permanently dropped
```

- [ ] **Step 6: Update the data reference doc**

In `docs/ingest_data.md`, section **2.1 Yahoo — stock prices**, replace the existing `State` bullet:

```
- **State** ✅: delisted / no-data tickers recorded in `state/blacklist.csv` and skipped on later runs;
  AC sanity-check offenders written to `state/ac_discrepancies.csv` (non-fatal).
```

with:

```
- **State** ✅: delisted / no-data tickers recorded in `state/blacklist.csv` (with an `attempts`
  counter) and skipped on later runs; AC sanity-check offenders written to
  `state/ac_discrepancies.csv` (non-fatal). The `housekeeping` command (see §5) retries blacklisted
  tickers, reviving ones that return data again and moving chronic failures (past
  `blacklist_max_attempts`) to `state/dead.csv`, never retried again.
```

(Section §5 is added by Task 4 below — this cross-reference is correct once that lands in the same
PR/branch.)

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/investalyze/ingest/providers/yahoo/provider.py tests/test_yahoo_housekeeping.py ingest.toml docs/ingest_data.md
git commit -m "feat(ingest): add yahoo blacklist recheck (revive/age-out/prune)"
```

---

### Task 3: Add the `housekeeping.py` runner

**Files:**
- Create: `src/investalyze/ingest/housekeeping.py`
- Test: `tests/test_housekeeping.py`

**Interfaces:**
- Consumes: `storage.connect` (`src/investalyze/ingest/storage.py:19`), `Config`/`Config.provider`
  (`src/investalyze/ingest/config.py:16-26`), `yahoo.recheck_blacklist` (Task 2).
- Produces: `housekeeping.HOUSEKEEPING_TASKS: dict[str, tuple[str, Callable[..., dict]]]`,
  `housekeeping.run_housekeeping(config: Config, tasks: Sequence[str] | None = None) -> dict[str, dict]`
  — Task 4's CLI calls `run_housekeeping` and reads `HOUSEKEEPING_TASKS` for the `-t` flag's choices.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_housekeeping.py`:

```python
"""Tests for the housekeeping runner: connect, run selected tasks, summarize."""
from pathlib import Path

import pandas as pd

from investalyze.ingest import housekeeping
from investalyze.ingest.config import Config


def _config(data_root: Path) -> Config:
    return Config(data_root=data_root, db='investalyze.duckdb', log_level='INFO',
                  providers={'yahoo': {'ticker_file': 'ticker.csv', 'batch_size': 10, 'sleep': 0,
                                       'blacklist_max_attempts': 3}})


def _ticker_csv(tmp_path: Path, *rows: tuple[str, str]) -> None:
    raw = tmp_path / 'yahoo' / 'raw'
    raw.mkdir(parents=True)
    pd.DataFrame(list(rows), columns=['ticker', 'market']).to_csv(raw / 'ticker.csv', index=False)


def test_run_housekeeping_executes_yahoo_blacklist(tmp_path: Path):
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    summary = housekeeping.run_housekeeping(_config(tmp_path), tasks=['yahoo-blacklist'])
    assert summary == {'yahoo-blacklist': {'rechecked': 0, 'revived': 0, 'died': 0}}


def test_run_housekeeping_defaults_to_all_tasks(tmp_path: Path):
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    summary = housekeeping.run_housekeeping(_config(tmp_path))
    assert list(summary) == list(housekeeping.HOUSEKEEPING_TASKS)


def test_yahoo_blacklist_is_registered():
    assert 'yahoo-blacklist' in housekeeping.HOUSEKEEPING_TASKS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_housekeeping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'investalyze.ingest.housekeeping'`.

- [ ] **Step 3: Implement `housekeeping.py`**

```python
"""The housekeeping runner: open the shared DB, then run each selected maintenance task.

Mirrors orchestrator.py's shape — a registry of name -> (provider name, task callable), one
place that opens the DB connection. Adding a task = one new function + one registry entry.
"""
import logging
from collections.abc import Callable, Sequence

from investalyze.ingest import storage
from investalyze.ingest.config import Config
from investalyze.ingest.providers.yahoo import provider as yahoo

log = logging.getLogger('investalyze.ingest')

# name -> (settings section to use, the task's (con, data_root, settings) -> dict result)
HousekeepingTask = Callable[..., dict]
HOUSEKEEPING_TASKS: dict[str, tuple[str, HousekeepingTask]] = {
    'yahoo-blacklist': ('yahoo', yahoo.recheck_blacklist),
}


def run_housekeeping(config: Config, tasks: Sequence[str] | None = None) -> dict[str, dict]:
    """Run the selected housekeeping tasks against the shared DB. Returns {task: result}.

    `tasks=None` runs every registered task. Opens one connection (at `config.data_root`/
    `config.db`), passes each task its provider's settings, and closes the connection when done.
    """
    selected = list(tasks) if tasks is not None else list(HOUSEKEEPING_TASKS)
    con = storage.connect(config.data_root, config.db)
    try:
        results: dict[str, dict] = {}
        for name in selected:
            provider_name, task = HOUSEKEEPING_TASKS[name]
            log.info(f'housekeeping: {name}')
            results[name] = task(con, config.data_root, config.provider(provider_name))
        return results
    finally:
        con.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_housekeeping.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/investalyze/ingest/housekeeping.py tests/test_housekeeping.py
git commit -m "feat(ingest): add housekeeping task runner"
```

---

### Task 4: Wire the `housekeeping` CLI command

**Files:**
- Modify: `src/investalyze/ingest/__main__.py`
- Modify: `docs/ingest_data.md` (new §5 usage note)

**Interfaces:**
- Consumes: `housekeeping.run_housekeeping`, `housekeeping.HOUSEKEEPING_TASKS` (Task 3).

- [ ] **Step 1: Update the import**

Replace:

```python
from investalyze.ingest import config, orchestrator
```

with:

```python
from investalyze.ingest import config, housekeeping, orchestrator
```

- [ ] **Step 2: Add the `housekeeping` command choice + `-t/--task` flag**

Replace:

```python
    parser.add_argument('command', nargs='?', choices=('setup',),
                        help="'setup' scaffolds the data dirs and exits; omit to run the ingest")
```

with:

```python
    parser.add_argument('command', nargs='?', choices=('setup', 'housekeeping'),
                        help="'setup' scaffolds the data dirs, 'housekeeping' runs maintenance "
                             "tasks (both exit without ingesting); omit to run the ingest")
```

Replace:

```python
    parser.add_argument('-p', '--provider', action='append', dest='providers',
                        choices=sorted(orchestrator.PROVIDERS),
                        help='provider to run; repeatable (default: all)')
    parser.add_argument('--update', action='store_true',
                        help='apply the daily update instead of a full load')
```

with:

```python
    parser.add_argument('-p', '--provider', action='append', dest='providers',
                        choices=sorted(orchestrator.PROVIDERS),
                        help='provider to run; repeatable (default: all)')
    parser.add_argument('-t', '--task', action='append', dest='tasks',
                        choices=sorted(housekeeping.HOUSEKEEPING_TASKS),
                        help='housekeeping task to run; repeatable (default: all)')
    parser.add_argument('--update', action='store_true',
                        help='apply the daily update instead of a full load')
```

- [ ] **Step 3: Dispatch the new command**

Replace:

```python
    if args.command == 'setup':
        orchestrator.create_data_dirs(cfg)
        log.info(f'data dirs ready under {cfg.data_root}')
        return

    summary = orchestrator.run(cfg, args.providers, update=args.update)
    for name, rows in summary.items():
        log.info(f'{name}: {rows} rows')
```

with:

```python
    if args.command == 'setup':
        orchestrator.create_data_dirs(cfg)
        log.info(f'data dirs ready under {cfg.data_root}')
        return

    if args.command == 'housekeeping':
        summary = housekeeping.run_housekeeping(cfg, args.tasks)
        for name, result in summary.items():
            log.info(f'{name}: {result}')
        return

    summary = orchestrator.run(cfg, args.providers, update=args.update)
    for name, rows in summary.items():
        log.info(f'{name}: {rows} rows')
```

- [ ] **Step 4: Manual smoke check**

This repo has no existing tests for `__main__.py`'s argparse wiring (consistent with current
coverage — CLI glue is verified manually here, same as `setup` always has been).

Run: `.venv/bin/python -m investalyze.ingest housekeeping --help`
Expected: prints usage including `{setup,housekeeping}` and a `-t TASK, --task TASK` option listing
`yahoo-blacklist` as a choice; exits 0.

Run: `.venv/bin/python -m investalyze.ingest housekeeping -t yahoo-blacklist`
Expected: logs `housekeeping: yahoo-blacklist` then `yahoo-blacklist: {'rechecked': N, 'revived': N, 'died': N}`
against the real `data/` dir, exits 0. (Safe to run for real — it only retries tickers already in
`data/yahoo/state/blacklist.csv`, which is currently empty/absent, so `N` will be 0 until the main
yahoo ingest run has blacklisted something under the new schema.)

- [ ] **Step 5: Add the usage note to the data reference doc**

In `docs/ingest_data.md`, add a new section at the end (after section 4 "ETL / cleaning"):

```markdown
## 5. Housekeeping

`python -m investalyze.ingest housekeeping` runs maintenance tasks against the ingest state,
separate from the regular provider ingest. Tasks are selectable with `-t/--task` (repeatable;
default: all).

| Task | Does |
|------|------|
| `yahoo-blacklist` | Retries every ticker in `data/yahoo/state/blacklist.csv`. Tickers that return data again are removed from the blacklist and added back to `ticker.csv` (picked up by the next regular ingest run, not ingested by housekeeping itself). Tickers still empty get `attempts` incremented; past `blacklist_max_attempts` (`ingest.toml [yahoo]`, default 5) they move to `data/yahoo/state/dead.csv` and are never retried again. `ticker.csv` is pruned of anything still blacklisted or now dead on every run. |
```

- [ ] **Step 6: Run the full suite one more time**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/investalyze/ingest/__main__.py docs/ingest_data.md
git commit -m "feat(ingest): add housekeeping CLI command"
```
