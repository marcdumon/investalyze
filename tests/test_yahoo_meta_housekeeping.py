"""Tests for the Yahoo metadata blacklist recheck (housekeeping)."""

from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest.providers.yahoo import meta_data as meta

_SETTINGS = {'sleep': 0, 'workers': 1, 'blacklist_max_attempts': 3}


def _state(tmp_path: Path, blacklist_rows: list[dict]) -> None:
    state_dir = tmp_path / 'yahoo' / 'state'
    state_dir.mkdir(parents=True)
    pd.DataFrame(blacklist_rows).to_csv(state_dir / 'meta_blacklist.csv', index=False)


def test_recheck_returns_zeros_when_no_blacklist(tmp_path):
    con = duckdb.connect()
    result = meta.recheck_meta_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 0, 'revived': 0, 'died': 0}


def test_recheck_revives_ticker_with_info(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'BACK', 'market': 'nyse', 'attempts': 1, 'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}])
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: {'industry': 'Tech'})
    con = duckdb.connect()
    result = meta.recheck_meta_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 1, 'revived': 1, 'died': 0}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'meta_blacklist.csv')
    assert blacklist.empty


def test_recheck_increments_attempts_when_still_empty(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'QUIET', 'market': 'nyse', 'attempts': 1, 'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}])
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: {})
    con = duckdb.connect()
    result = meta.recheck_meta_blacklist(con, tmp_path, _SETTINGS)
    assert result == {'rechecked': 1, 'revived': 0, 'died': 0}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'meta_blacklist.csv')
    assert blacklist.loc[0, 'attempts'] == 2
    assert blacklist.loc[0, 'last_checked'] != '2024-01-01'


def test_recheck_moves_to_dead_after_max_attempts(tmp_path, monkeypatch):
    _state(tmp_path, [{'ticker': 'GONE', 'market': 'nyse', 'attempts': 2, 'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-05'}])
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: {})
    con = duckdb.connect()
    result = meta.recheck_meta_blacklist(con, tmp_path, _SETTINGS)  # max_attempts=3, this is the 3rd try
    assert result == {'rechecked': 1, 'revived': 0, 'died': 1}
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'meta_blacklist.csv')
    assert blacklist.empty
    dead = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'meta_dead.csv')
    assert dead.loc[0, 'ticker'] == 'GONE'
    assert dead.loc[0, 'attempts'] == 3
