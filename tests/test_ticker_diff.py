"""Tests for the cross-provider ticker diff."""
import duckdb

from investalyze.ingest.housekeeping import ticker_diff


def test_ticker_diff_splits_both_ways(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    con.execute('CREATE TABLE prices (Ticker VARCHAR)')
    con.execute('CREATE TABLE _simfin_companies (Ticker VARCHAR)')
    con.execute("INSERT INTO prices VALUES ('AAA'), ('YON')")             # YON only in yahoo
    con.execute("INSERT INTO _simfin_companies VALUES ('AAA'), ('SON')")  # SON only in simfin

    diff = ticker_diff(con)

    assert diff == {'simfin_only': ['SON'], 'yahoo_only': ['YON']}
