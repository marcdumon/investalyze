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


def test_yahoo_meta_blacklist_is_registered():
    assert 'yahoo-meta-blacklist' in housekeeping.HOUSEKEEPING_TASKS
