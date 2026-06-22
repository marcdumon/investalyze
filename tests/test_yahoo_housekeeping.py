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
