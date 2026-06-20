"""Tests for Stooq acquire helpers: which source categories are in scope."""
import zipfile
from pathlib import Path

import pytest

from investalyze.ingest.providers.stooq.provider import (
    _asset_class_from_category,
    _asset_class_from_ticker,
    _extract_bulk,
    _read_tree,
    _read_update_file,
)

_HEADER = '<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>'


def _write(path: Path, ticker: str) -> None:
    """Write a one-row Stooq ticker file at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{_HEADER}\n{ticker},D,20240321,000000,1.0,2.0,0.5,1.5,0,0\n')


def _sample_tree(root: Path) -> None:
    """A daily/ tree: in-scope bonds + nested currencies, plus skip-worthy dirs."""
    _write(root / 'world' / 'bonds' / '10yusy.b.txt', '10YUSY.B')
    _write(root / 'world' / 'currencies' / 'major' / 'eurusd.txt', 'EURUSD')   # nested subdir
    _write(root / 'world' / 'cryptocurrencies' / 'btc.txt', 'BTC.V')           # banned
    _write(root / 'us' / 'nasdaq stocks' / 'aapl.us.txt', 'AAPL.US')           # equity -> Yahoo


@pytest.mark.parametrize('category, expected', [
    ('bonds', 'bonds'),
    ('currencies', 'currencies'),
    ('indices', 'indices'),
])
def test_in_scope_categories_map_to_asset_class(category: str, expected: str):
    assert _asset_class_from_category(category) == expected


@pytest.mark.parametrize('category', [
    'nasdaq stocks', 'nyse stocks', 'nasdaq etfs', 'nyse etfs',  # equities -> Yahoo owns them
    'money market', 'cryptocurrencies',                          # excluded / banned
])
def test_out_of_scope_categories_return_none(category: str):
    assert _asset_class_from_category(category) is None


@pytest.mark.parametrize('ticker, expected', [
    ('10YUSY.B', 'bonds'),       # bond yield
    ('10YCAP.B', 'bonds'),       # bond price (also .B)
    ('^AEX', 'indices'),
    ('^SPX', 'indices'),
    ('EURUSD', 'currencies'),
    ('AUDCAD', 'currencies'),
])
def test_asset_class_from_ticker_classifies_in_scope(ticker: str, expected: str):
    assert _asset_class_from_ticker(ticker) == expected


@pytest.mark.parametrize('ticker', [
    'AAPL.US',     # US equity -> Yahoo
    'BTC.V',       # crypto
    'PLOPLN1M',    # money market (not 6 alpha)
    'GC.F',        # commodity future
])
def test_asset_class_from_ticker_drops_out_of_scope(ticker: str):
    assert _asset_class_from_ticker(ticker) is None


def test_read_tree_keeps_only_in_scope_categories(tmp_path: Path):
    root = tmp_path / 'daily'
    _sample_tree(root)
    out = _read_tree(root)
    assert set(out['Ticker']) == {'10YUSY', 'EURUSD'}                              # crypto + equity skipped
    assert list(out.columns) == ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'AssetClass']


def test_read_tree_tags_asset_class_from_category(tmp_path: Path):
    root = tmp_path / 'daily'
    _sample_tree(root)
    out = _read_tree(root).set_index('Ticker')
    assert out.loc['10YUSY', 'AssetClass'] == 'bonds'
    assert out.loc['EURUSD', 'AssetClass'] == 'currencies'                         # found in nested subdir


def test_read_tree_skips_empty_files(tmp_path: Path):
    root = tmp_path / 'daily'
    _write(root / 'world' / 'bonds' / '10yusy.b.txt', '10YUSY.B')
    (root / 'world' / 'bonds' / 'empty.b.txt').write_text('')   # zero-byte file in the real tree
    out = _read_tree(root)
    assert set(out['Ticker']) == {'10YUSY'}                     # empty file skipped, no crash


def test_extract_bulk_unzips_world_zip(tmp_path: Path):
    raw = tmp_path
    with zipfile.ZipFile(raw / 'd_world_txt.zip', 'w') as zf:
        zf.writestr('data/daily/world/bonds/10yusy.b.txt', f'{_HEADER}\n10YUSY.B,D,20240321,000000,5,5,5,5,0,0\n')
    _extract_bulk(raw)
    assert (raw / 'data' / 'daily' / 'world' / 'bonds' / '10yusy.b.txt').is_file()


def test_extract_bulk_is_noop_without_zip(tmp_path: Path):
    # No zip present: rely on an already-extracted tree, don't fail.
    _write(tmp_path / 'data' / 'daily' / 'world' / 'bonds' / '10yusy.b.txt', '10YUSY.B')
    _extract_bulk(tmp_path)
    assert (tmp_path / 'data' / 'daily' / 'world' / 'bonds' / '10yusy.b.txt').is_file()


def _update_file(path: Path) -> None:
    """A flat data_d.txt-style file with mixed in/out-of-scope instruments."""
    rows = [
        _HEADER,
        '10YUSY.B,D,20240321,000000,5.0,5.1,4.9,5.05,0,0',        # bond
        'EURUSD,D,20240321,000000,1.0,1.1,0.9,1.05,0,0',          # currency
        '^AEX,D,20240321,000000,800,810,790,805,0,0',             # index
        'AAPL.US,D,20240321,000000,180,181,179,180.5,0,0',        # equity -> drop
        'BTC.V,D,20240321,000000,60000,61000,59000,60500,0,0',    # crypto -> drop
    ]
    path.write_text('\n'.join(rows) + '\n')


def test_read_update_file_drops_out_of_scope_tickers(tmp_path: Path):
    f = tmp_path / 'data_d.txt'
    _update_file(f)
    out = _read_update_file(f)
    assert set(out['Ticker']) == {'10YUSY', 'EURUSD', '^AEX'}                       # equity + crypto dropped
    assert list(out.columns) == ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'AssetClass']


def test_read_update_file_tags_asset_class_by_pattern(tmp_path: Path):
    f = tmp_path / 'data_d.txt'
    _update_file(f)
    out = _read_update_file(f).set_index('Ticker')
    assert out.loc['10YUSY', 'AssetClass'] == 'bonds'
    assert out.loc['EURUSD', 'AssetClass'] == 'currencies'
    assert out.loc['^AEX', 'AssetClass'] == 'indices'
