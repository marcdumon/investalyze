"""End-to-end: stooq.run reads the input and loads market_data via storage.write."""
from pathlib import Path

import duckdb

from investalyze.ingest.providers.stooq.provider import run

_HEADER = '<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>'


def _write(path: Path, ticker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{_HEADER}\n{ticker},D,20240321,000000,1.0,2.0,0.5,1.5,0,0\n')


def test_run_loads_bulk_tree_into_market_data(tmp_path: Path):
    tree = tmp_path / 'stooq' / 'raw' / 'data'
    _write(tree / 'world' / 'bonds' / '10yusy.b.txt', '10YUSY.B')
    _write(tree / 'world' / 'currencies' / 'major' / 'eurusd.txt', 'EURUSD')
    con = duckdb.connect()
    n = run(con, tmp_path)
    assert n == 2
    rows = con.execute('SELECT Ticker, AssetClass FROM market_data ORDER BY Ticker').fetchall()
    assert rows == [('10YUSY', 'bonds'), ('EURUSD', 'currencies')]


def test_run_update_loads_update_file(tmp_path: Path):
    raw = tmp_path / 'stooq' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'data_d.txt').write_text(f'{_HEADER}\n10YUSY.B,D,20240321,000000,5.0,5.1,4.9,5.05,0,0\n')
    con = duckdb.connect()
    n = run(con, tmp_path, update=True)
    assert n == 1
    assert con.execute('SELECT Ticker FROM market_data').fetchall() == [('10YUSY',)]
