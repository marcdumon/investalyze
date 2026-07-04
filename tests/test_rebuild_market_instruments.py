"""Tests for the market_instruments housekeeping rebuild."""
import duckdb

from investalyze.ingest.housekeeping import rebuild_market_instruments


def _seed(con: duckdb.DuckDBPyConnection) -> None:
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, AssetClass VARCHAR)')
    con.execute("INSERT INTO market_data VALUES ('^SPX', '2026-01-02', 1, 1, 1, 1, 'indices')")
    con.execute("INSERT INTO market_data VALUES ('10YUSY', '2026-01-02', 1, 1, 1, 1, 'bonds')")
    con.execute("INSERT INTO market_data VALUES ('EURUSD', '2026-01-02', 1, 1, 1, 1, 'currencies')")
    con.execute("INSERT INTO market_data VALUES ('^NOTREAL', '2026-01-02', 1, 1, 1, 1, 'indices')")


def test_rebuild_decodes_each_asset_class(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed(con)

    result = rebuild_market_instruments(con, tmp_path, {})

    assert result == {'rows': 4, 'decoded': 3, 'undecoded': 1}
    spx = con.execute("SELECT Name, Country, AssetClass FROM market_instruments WHERE Ticker = '^SPX'").fetchone()
    assert spx == ('S&P 500', 'United States', 'indices')
    bond = con.execute("SELECT Name, Country, AssetClass FROM market_instruments WHERE Ticker = '10YUSY'").fetchone()
    assert bond == ('10-year government bond yield', 'United States', 'bonds')
    fx = con.execute("SELECT Name, Country, AssetClass FROM market_instruments WHERE Ticker = 'EURUSD'").fetchone()
    assert fx == ('Euro/US Dollar exchange rate', None, 'currencies')
    unknown = con.execute("SELECT Name, Country FROM market_instruments WHERE Ticker = '^NOTREAL'").fetchone()
    assert unknown == (None, None)


def test_rebuild_is_idempotent(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed(con)
    rebuild_market_instruments(con, tmp_path, {})
    second = rebuild_market_instruments(con, tmp_path, {})
    assert second == {'rows': 4, 'decoded': 3, 'undecoded': 1}


def test_rebuild_handles_empty_market_data(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, AssetClass VARCHAR)')
    result = rebuild_market_instruments(con, tmp_path, {})
    assert result == {'rows': 0, 'decoded': 0, 'undecoded': 0}
