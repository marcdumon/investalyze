"""End-to-end: simfin.run acquires (monkeypatched) then loads tables via storage.write."""
import zipfile
from pathlib import Path

import duckdb
import pytest

from investalyze.ingest.providers.simfin import fundamental_data as provider


def _zip(path: Path, member: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(member, text)


def _income(ticker: str, revenue: int) -> str:
    return f'Ticker;SimFinId;Fiscal Year;Fiscal Period;Revenue\n{ticker};101;2023;FY;{revenue}\n'


def _seed_raw(raw: Path) -> None:
    _zip(raw / 'us-income-annual-full-asreported.zip', 'a.csv', _income('AAPL', 100))
    _zip(raw / 'us-income-annual-full.zip', 'b.csv', _income('AAPL', 110))
    _zip(raw / 'us-companies.zip', 'c.csv',
         'Ticker;SimFinId;Company Name;IndustryId;Market;Number Employees;Business Summary;'
         'Main Currency;End of financial year (month)\n'
         'AAPL;101;Apple Inc;100001;us;1000;Makes phones;USD;9\n')
    _zip(raw / 'industries.zip', 'i.csv', 'IndustryId;Industry;Sector\n100001;HW;Tech\n')


def test_run_loads_fundamentals_and_companies(tmp_path: Path, monkeypatch):
    _seed_raw(tmp_path / 'simfin' / 'raw')
    monkeypatch.setattr(provider, '_acquire', lambda *a, **k: None)   # no network; use seeded zips
    monkeypatch.setenv('SIMFIN_API_KEY', 'K')
    con = duckdb.connect()
    n = provider.run(con, tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 90})
    assert n > 0
    assert con.execute('SELECT COUNT(*) FROM income').fetchone()[0] == 2     # asreported + restated
    assert con.execute("SELECT Sector FROM _simfin_companies WHERE Ticker = 'AAPL'").fetchone()[0] == 'Tech'
    # re-run is idempotent (merge-upsert on the key, not append)
    provider.run(con, tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 90})
    assert con.execute('SELECT COUNT(*) FROM income').fetchone()[0] == 2


def test_run_raises_without_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv('SIMFIN_API_KEY', raising=False)
    con = duckdb.connect()
    with pytest.raises(RuntimeError, match='SIMFIN_API_KEY'):
        provider.run(con, tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 90})


def test_companies_columns_are_canonical(tmp_path: Path, monkeypatch):
    _seed_raw(tmp_path / 'simfin' / 'raw')
    monkeypatch.setattr(provider, '_acquire', lambda *a, **k: None)
    monkeypatch.setenv('SIMFIN_API_KEY', 'K')
    con = duckdb.connect()
    provider.run(con, tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 90})
    cols = {c[0] for c in con.execute('DESCRIBE _simfin_companies').fetchall()}
    assert {'CompanyName', 'NumberEmployees', 'BusinessSummary', 'MainCurrency',
            'FinancialYearEndMonth'} <= cols
    assert not ({'Company Name', 'Number Employees', 'Business Summary', 'Main Currency',
                 'End of financial year (month)'} & cols)
