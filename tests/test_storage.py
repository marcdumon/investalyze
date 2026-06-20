"""Tests for storage.write — the single merge/upsert path into DuckDB."""
from datetime import date

import duckdb
import pandas as pd

from investalyze.ingest import storage

_KEY = ['Ticker', 'Date']


def _row(ticker: str, close: float) -> pd.DataFrame:
    """One market_data row."""
    return pd.DataFrame({
        'Ticker': [ticker], 'Date': [date(2024, 3, 21)],
        'O': [1.0], 'H': [2.0], 'L': [0.5], 'C': [close], 'AssetClass': ['currencies'],
    })


def test_write_creates_table_and_inserts_rows():
    con = duckdb.connect()
    n = storage.write(con, 'market_data', _row('EURUSD', 1.5), key=_KEY)
    assert n == 1
    assert con.execute('SELECT Ticker, C FROM market_data').fetchall() == [('EURUSD', 1.5)]


def test_write_upserts_on_key_without_duplicating():
    con = duckdb.connect()
    storage.write(con, 'market_data', _row('EURUSD', 1.5), key=_KEY)
    n = storage.write(con, 'market_data', _row('EURUSD', 9.9), key=_KEY)   # same (Ticker, Date)
    assert n == 1                                                          # upserted, not appended
    assert con.execute('SELECT C FROM market_data').fetchone()[0] == 9.9   # value updated


def test_write_inserts_distinct_keys():
    con = duckdb.connect()
    storage.write(con, 'market_data', _row('EURUSD', 1.5), key=_KEY)
    n = storage.write(con, 'market_data', _row('GBPUSD', 1.2), key=_KEY)
    assert n == 2


def test_connect_opens_db_file_at_data_root(tmp_path):
    con = storage.connect(tmp_path)
    con.execute('CREATE TABLE t (x INTEGER)')
    con.execute('INSERT INTO t VALUES (1)')
    assert con.execute('SELECT x FROM t').fetchone() == (1,)
    assert (tmp_path / 'investalyze.duckdb').exists()
