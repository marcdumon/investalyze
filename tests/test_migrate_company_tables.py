"""Tests for the one-off company-table rename + combined-build migration."""
import duckdb

from scripts.migrate_company_tables import migrate


def _seed_old(con: duckdb.DuckDBPyConnection) -> None:
    con.execute('CREATE TABLE company_profile (Ticker VARCHAR, Industry VARCHAR, Sector VARCHAR, '
                'FullTimeEmployees BIGINT, Address1 VARCHAR, City VARCHAR, State VARCHAR, Zip VARCHAR, '
                'Country VARCHAR, Website VARCHAR, IRWebsite VARCHAR, BusinessSummary VARCHAR)')
    con.execute('CREATE TABLE companies (Ticker VARCHAR, Industry VARCHAR, Sector VARCHAR, '
                'NumberEmployees BIGINT, CompanyName VARCHAR, ISIN VARCHAR, CIK BIGINT, BusinessSummary VARCHAR)')
    con.execute("INSERT INTO company_profile VALUES ('AAA','Y','Y',1,'a','c','s','z','US','w','ir','sum')")
    con.execute("INSERT INTO companies VALUES ('AAA','S','S',2,'Apple','US1',1,'sum')")


def test_migrate_renames_and_builds(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed_old(con)

    done = migrate(con)

    tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
    assert '_yahoo_companies' in tables and '_simfin_companies' in tables
    assert 'company_profile' not in tables and 'companies' in tables  # combined now owns the name
    assert con.execute("SELECT InYahoo, InSimfin FROM companies WHERE Ticker = 'AAA'").fetchone() == (True, True)
    assert any('company_profile' in d for d in done)


def test_migrate_is_idempotent(tmp_path):
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    _seed_old(con)
    migrate(con)
    second = migrate(con)             # already migrated -> no renames, rebuild still fine
    assert second == []
    assert '_simfin_companies' in {r[0] for r in con.execute('SHOW TABLES').fetchall()}
