"""Tests for the Yahoo metadata provider (yfinance .info mocked)."""
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest.providers.yahoo import meta_data as meta

_SETTINGS = {'sleep': 0, 'workers': 1, 'refresh_days_meta': 90, 'blacklist_max_attempts': 3}


def _ticker_csv(tmp_path: Path, *rows: tuple[str, str]) -> None:
    raw = tmp_path / 'yahoo' / 'raw'
    raw.mkdir(parents=True)
    pd.DataFrame(list(rows), columns=['ticker', 'market']).to_csv(raw / 'ticker.csv', index=False)


def _info(**overrides) -> dict:
    base = {'address1': '1 Infinite Loop', 'city': 'Cupertino', 'state': 'CA', 'zip': '95014',
            'country': 'United States', 'website': 'https://example.com', 'industry': 'Tech',
            'sector': 'Technology', 'longBusinessSummary': 'Makes things.', 'fullTimeEmployees': 100,
            'auditRisk': 1, 'boardRisk': 1, 'compensationRisk': 1, 'shareHolderRightsRisk': 1,
            'overallRisk': 1, 'irWebsite': 'https://ir.example.com',
            'companyOfficers': [{'name': 'Jane Doe', 'title': 'CEO', 'age': 50, 'yearBorn': 1974,
                                 'fiscalYear': 2024, 'totalPay': 1000000, 'exercisedValue': 0,
                                 'unexercisedValue': 0}]}
    base.update(overrides)
    return base


def test_run_writes_profile_and_officers(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: _info())
    con = duckdb.connect()
    n = meta.run(con, tmp_path, _SETTINGS)
    assert n == 1
    profile = con.execute("SELECT Ticker, Src, City, FullTimeEmployees FROM _yahoo_companies").fetchall()
    assert profile == [('AAA', 'yahoo', 'Cupertino', 100)]
    officers = con.execute("SELECT Ticker, Name, Title FROM company_officers").fetchall()
    assert officers == [('AAA', 'Jane Doe', 'CEO')]


def test_run_skips_ticker_fetched_recently(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    calls: list[str] = []
    def fake_info(sym):
        calls.append(sym)
        return _info()
    monkeypatch.setattr(meta, '_fetch_info', fake_info)
    con = duckdb.connect()
    meta.run(con, tmp_path, _SETTINGS)
    assert calls == ['AAA']
    meta.run(con, tmp_path, _SETTINGS)
    assert calls == ['AAA']   # second run: FetchedOn is today, not stale -> skipped


def test_run_refetches_stale_ticker(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('AAA', 'nyse'))
    calls: list[str] = []
    def fake_info(sym):
        calls.append(sym)
        return _info()
    monkeypatch.setattr(meta, '_fetch_info', fake_info)
    con = duckdb.connect()
    meta.run(con, tmp_path, _SETTINGS)
    con.execute("UPDATE _yahoo_companies SET FetchedOn = DATE '2020-01-01' WHERE Ticker = 'AAA'")
    meta.run(con, tmp_path, _SETTINGS)
    assert calls == ['AAA', 'AAA']
    row = con.execute("SELECT FetchedOn FROM _yahoo_companies WHERE Ticker = 'AAA'").fetchone()
    assert row[0] == date.today()


def test_run_skips_price_blacklisted_and_dead_tickers(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('GOOD', 'nyse'), ('BLACK', 'nyse'), ('DEAD', 'nyse'))
    price_state = tmp_path / 'yahoo' / 'state'
    price_state.mkdir(parents=True)
    pd.DataFrame([{'ticker': 'BLACK', 'market': 'nyse', 'attempts': 1,
                  'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}]
                 ).to_csv(price_state / 'price_blacklist.csv', index=False)
    pd.DataFrame([{'ticker': 'DEAD', 'attempts': 5, 'first_blacklisted': '2024-01-01', 'died_on': '2024-02-01'}]
                 ).to_csv(price_state / 'price_dead.csv', index=False)
    calls: list[str] = []
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: (calls.append(sym) or _info()))
    con = duckdb.connect()
    meta.run(con, tmp_path, _SETTINGS)
    assert calls == ['GOOD']


def test_run_blacklists_ticker_with_no_info(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('EMPTY', 'nyse'))
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: {})
    con = duckdb.connect()
    n = meta.run(con, tmp_path, _SETTINGS)
    assert n == 0
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'meta_blacklist.csv')
    assert blacklist.loc[0, 'ticker'] == 'EMPTY'
    assert blacklist.loc[0, 'market'] == 'nyse'
    assert blacklist.loc[0, 'attempts'] == 1


def test_run_skips_already_meta_blacklisted_ticker(tmp_path, monkeypatch):
    _ticker_csv(tmp_path, ('QUIET', 'nyse'))
    meta_state = tmp_path / 'yahoo' / 'state'
    meta_state.mkdir(parents=True)
    pd.DataFrame([{'ticker': 'QUIET', 'market': 'nyse', 'attempts': 1,
                  'first_blacklisted': '2024-01-01', 'last_checked': '2024-01-01'}]
                 ).to_csv(meta_state / 'meta_blacklist.csv', index=False)
    calls: list[str] = []
    monkeypatch.setattr(meta, '_fetch_info', lambda sym: (calls.append(sym) or _info()))
    con = duckdb.connect()
    meta.run(con, tmp_path, _SETTINGS)
    assert calls == []
