"""Tests for SimFin transform: extract, union+tag variants, companies join."""
import tempfile
import zipfile
from pathlib import Path

import duckdb

from investalyze.ingest.providers.simfin import fundamental_data as provider


def _zip(path: Path, member: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(member, text)


def _income_csv(ticker: str, fy: int, revenue: int) -> str:
    return (f'Ticker;SimFinId;Fiscal Year;Fiscal Period;Revenue\n'
            f'{ticker};101;{fy};FY;{revenue}\n')


def _place_income_variants(raw: Path) -> None:
    # one row per variant so we can see Period / IsRestated tagging
    _zip(raw / 'us-income-annual-full-asreported.zip', 'us-income-annual-full-asreported.csv',
         _income_csv('AAPL', 2023, 100))
    _zip(raw / 'us-income-quarterly-full-asreported.zip', 'us-income-quarterly-full-asreported.csv',
         _income_csv('AAPL', 2023, 25))
    _zip(raw / 'us-income-annual-full.zip', 'us-income-annual-full.csv',
         _income_csv('AAPL', 2023, 110))
    _zip(raw / 'us-income-quarterly-full.zip', 'us-income-quarterly-full.csv',
         _income_csv('AAPL', 2023, 28))


def test_read_statement_unions_and_tags(tmp_path: Path):
    raw = tmp_path / 'raw'
    _place_income_variants(raw)
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as td:
        df = provider._read_statement(con, raw, Path(td), 'income')
    assert len(df) == 4
    assert set(df['Market']) == {'us'}
    assert set(df['Period']) == {'A', 'Q'}
    assert set(df['IsRestated'].tolist()) == {True, False}
    assert set(df['Src']) == {'simfin'}
    assert 'SrcId' in df.columns and 'SimFinId' not in df.columns
    # the as-reported annual row keeps its raw value
    arow = df[(df['Period'] == 'A') & (~df['IsRestated'])]
    assert int(arow['Revenue'].iloc[0]) == 100


def test_read_statement_skips_missing_variants(tmp_path: Path):
    raw = tmp_path / 'raw'
    _zip(raw / 'us-income-annual-full.zip', 'us-income-annual-full.csv', _income_csv('AAPL', 2023, 110))
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as td:
        df = provider._read_statement(con, raw, Path(td), 'income')
    assert len(df) == 1                      # only the one present variant
    assert df['IsRestated'].iloc[0] is True or bool(df['IsRestated'].iloc[0]) is True


def test_read_statement_returns_empty_when_no_zips(tmp_path: Path):
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as td:
        df = provider._read_statement(con, tmp_path / 'raw', Path(td), 'balance')
    assert df.empty


def test_read_companies_joins_industries(tmp_path: Path):
    raw = tmp_path / 'raw'
    _zip(raw / 'us-companies.zip', 'us-companies.csv',
         'Ticker;SimFinId;Company Name;IndustryId;Market\nAAPL;101;Apple Inc;100001;us\n')
    _zip(raw / 'industries.zip', 'industries.csv',
         'IndustryId;Industry;Sector\n100001;Computer Hardware;Technology\n')
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as td:
        df = provider._read_companies(con, raw, Path(td))
    assert df['Industry'].iloc[0] == 'Computer Hardware'
    assert df['Sector'].iloc[0] == 'Technology'
    assert df['Market'].iloc[0] == 'us'
    assert 'SrcId' in df.columns and 'SimFinId' not in df.columns
