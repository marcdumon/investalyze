"""Tests for the backtest CLI: argument parsing, printing a stored report and deleting stored runs."""

import argparse
from pathlib import Path

import duckdb
import pytest

from investalyze.backtest import __main__ as cli
from investalyze.backtest import writer


def _args(**overrides) -> argparse.Namespace:
    """A parsed-argument namespace with every flag defaulted, then the overrides applied."""
    defaults = dict(command='report', run_id=None)
    return argparse.Namespace(**(defaults | overrides))


def test_the_parser_accepts_every_command():
    for command in cli.COMMANDS:
        assert cli._parser().parse_args([command]).command == command


def test_the_parser_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(['backtest-everything'])


def test_deleting_without_a_run_id_fails_before_touching_the_database():
    with pytest.raises(SystemExit, match='delete needs --run-id'):
        cli._delete(None, _args(command='delete'), Path('data'))


def test_deleting_a_run_that_was_never_stored_says_so(tmp_path: Path):
    con = duckdb.connect()
    writer.ensure_tables(con)
    with pytest.raises(SystemExit, match='no stored run: nosuchrun'):
        cli._delete(con, _args(command='delete', run_id=['nosuchrun']), tmp_path)


def test_deleting_a_stored_run_clears_it_and_leaves_the_others(tmp_path: Path):
    con = duckdb.connect()
    writer.ensure_tables(con)
    for identifier in ('keepme00', 'dropme00'):
        con.execute("INSERT INTO backtest_equity VALUES (?, DATE '2020-01-02', 100.0, 0, 0, 0, NULL)",
                    [identifier])
    (tmp_path / 'backtests' / 'dropme00').mkdir(parents=True)
    (tmp_path / 'backtests' / 'dropme00' / 'report.md').write_text('gone')

    cli._delete(con, _args(command='delete', run_id=['dropme00']), tmp_path)

    remaining = con.execute('SELECT RunId FROM backtest_equity').fetchall()
    assert remaining == [('keepme00',)]
    assert not (tmp_path / 'backtests' / 'dropme00').exists()


def test_the_parser_reads_several_run_ids():
    args = cli._parser().parse_args(['delete', '--run-id', 'aaa', 'bbb', 'ccc'])
    assert args.run_id == ['aaa', 'bbb', 'ccc']


def test_deleting_several_runs_takes_them_all_in_one_pass(tmp_path: Path):
    con = duckdb.connect()
    writer.ensure_tables(con)
    for identifier in ('one', 'two', 'three'):
        con.execute("INSERT INTO backtest_equity VALUES (?, DATE '2020-01-02', 100.0, 0, 0, 0, NULL)",
                    [identifier])

    cli._delete(con, _args(command='delete', run_id=['one', 'three']), tmp_path)

    assert con.execute('SELECT RunId FROM backtest_equity').fetchall() == [('two',)]


def test_an_unknown_id_among_good_ones_still_deletes_the_good_ones(tmp_path: Path):
    con = duckdb.connect()
    writer.ensure_tables(con)
    con.execute("INSERT INTO backtest_equity VALUES ('real', DATE '2020-01-02', 100.0, 0, 0, 0, NULL)")

    with pytest.raises(SystemExit, match='no stored run: ghost'):
        cli._delete(con, _args(command='delete', run_id=['real', 'ghost']), tmp_path)

    assert con.execute('SELECT count(*) FROM backtest_equity').fetchone()[0] == 0


def test_reporting_without_a_run_id_fails_before_touching_the_disk():
    with pytest.raises(SystemExit, match='report needs one --run-id'):
        cli._report(_args(command='report'), Path('data'))


def test_reporting_more_than_one_run_is_refused():
    with pytest.raises(SystemExit, match='report needs one --run-id'):
        cli._report(_args(command='report', run_id=['aaa', 'bbb']), Path('data'))


def test_reporting_a_run_with_no_report_on_disk_says_where_it_looked(tmp_path: Path):
    with pytest.raises(SystemExit, match='no report at'):
        cli._report(_args(command='report', run_id=['ghost']), tmp_path)


def test_reporting_prints_what_the_run_wrote(tmp_path: Path, capsys):
    directory = tmp_path / 'backtests' / 'abc123def456'
    directory.mkdir(parents=True)
    (directory / 'report.md').write_text('# run abc123def456\nCAGR 11.6%')

    cli._report(_args(command='report', run_id=['abc123def456']), tmp_path)

    assert 'CAGR 11.6%' in capsys.readouterr().out
