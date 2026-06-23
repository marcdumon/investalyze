"""Tests for the one-off PascalCase column migration."""
import duckdb

from scripts.migrate_pascalcase_columns import migrate


def test_migrate_renames_old_columns(tmp_path):
    db = tmp_path / 'test.duckdb'
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE _yahoo_companies (Ticker VARCHAR, Src VARCHAR, "
                'city VARCHAR, fullTimeEmployees BIGINT, FetchedOn DATE)')
    con.execute("INSERT INTO _yahoo_companies VALUES ('AAA', 'yahoo', 'Cupertino', 100, DATE '2024-01-01')")

    done = migrate(con)

    cols = {c[0] for c in con.execute('DESCRIBE _yahoo_companies').fetchall()}
    assert 'City' in cols and 'FullTimeEmployees' in cols
    assert 'city' not in cols and 'fullTimeEmployees' not in cols
    assert any('city' in d for d in done)
    # data preserved
    assert con.execute('SELECT City FROM _yahoo_companies').fetchone() == ('Cupertino',)


def test_migrate_is_idempotent(tmp_path):
    db = tmp_path / 'test.duckdb'
    con = duckdb.connect(str(db))
    con.execute('CREATE TABLE _yahoo_companies (Ticker VARCHAR, Src VARCHAR, city VARCHAR, FetchedOn DATE)')
    migrate(con)
    second = migrate(con)              # already migrated
    assert second == []                # no-op, no error
    cols = {c[0] for c in con.execute('DESCRIBE _yahoo_companies').fetchall()}
    assert 'City' in cols


def test_migrate_skips_missing_table(tmp_path):
    db = tmp_path / 'test.duckdb'
    con = duckdb.connect(str(db))      # no tables at all
    assert migrate(con) == []          # does not raise
