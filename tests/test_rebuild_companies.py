"""Tests for the combined-companies housekeeping rebuild."""
import duckdb

from investalyze.ingest.housekeeping import rebuild_companies


def _seed(con: duckdb.DuckDBPyConnection) -> None:
    con.execute('CREATE TABLE _yahoo_companies (Ticker VARCHAR, Industry VARCHAR, Sector VARCHAR, '
                'FullTimeEmployees BIGINT, Address1 VARCHAR, City VARCHAR, State VARCHAR, Zip VARCHAR, '
                'Country VARCHAR, Website VARCHAR, IRWebsite VARCHAR, BusinessSummary VARCHAR)')
    con.execute('CREATE TABLE _simfin_companies (Ticker VARCHAR, Industry VARCHAR, Sector VARCHAR, '
                'NumberEmployees BIGINT, CompanyName VARCHAR, ISIN VARCHAR, CIK BIGINT, BusinessSummary VARCHAR)')
    # shared AAA: yahoo and simfin disagree -> yahoo must win the overlaps
    con.execute("INSERT INTO _yahoo_companies VALUES ('AAA', 'Y-Ind', 'Y-Sec', 100, '1 Y St', 'Yville', "
                "'CA', '90001', 'US', 'y.com', 'ir.y.com', 'Y summary')")
    con.execute("INSERT INTO _simfin_companies VALUES ('AAA', 'S-Ind', 'S-Sec', 200, 'Apple Inc', 'US1', 320193, 'S summary')")
    # yahoo-only YYY
    con.execute("INSERT INTO _yahoo_companies VALUES ('YYY', 'Y2', 'Y2', 5, '2 St', 'Town', 'NY', '10001', 'US', 'y2.com', NULL, 'Y2 sum')")
    # simfin-only SSS
    con.execute("INSERT INTO _simfin_companies VALUES ('SSS', 'S2', 'S2', 9, 'Sss Corp', 'US2', 111, 'S2 sum')")


def test_rebuild_combines_with_yahoo_priority(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed(con)

    result = rebuild_companies(con, tmp_path, {})

    assert result == {'rows': 3, 'in_yahoo': 2, 'in_simfin': 2, 'both': 1}
    row = con.execute("SELECT InYahoo, InSimfin, Industry, Sector, NrEmployees, CompanyName, City, ISIN, CIK, "
                      "BusinessSummary FROM companies WHERE Ticker = 'AAA'").fetchone()
    assert row == (True, True, 'Y-Ind', 'Y-Sec', 100, 'Apple Inc', 'Yville', 'US1', 320193, 'Y summary')
    yyy = con.execute("SELECT InYahoo, InSimfin, CompanyName, ISIN FROM companies WHERE Ticker = 'YYY'").fetchone()
    assert yyy == (True, False, None, None)
    sss = con.execute("SELECT InYahoo, InSimfin, City, Website FROM companies WHERE Ticker = 'SSS'").fetchone()
    assert sss == (False, True, None, None)


def test_rebuild_is_idempotent(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed(con)
    rebuild_companies(con, tmp_path, {})
    second = rebuild_companies(con, tmp_path, {})   # CREATE OR REPLACE — safe re-run
    assert second == {'rows': 3, 'in_yahoo': 2, 'in_simfin': 2, 'both': 1}
