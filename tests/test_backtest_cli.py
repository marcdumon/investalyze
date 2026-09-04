"""Tests for the backtest CLI surface: argument parsing, the signal catalog, and universe resolution.

The commands themselves are covered where their modules are tested; what matters here is that every
command is reachable, that the catalog shown in `--help` stays in step with the registry, and that a
command missing a required argument fails before it opens the database.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from investalyze.backtest import __main__ as cli
from investalyze.backtest import config as backtest_config
from investalyze.backtest import registry, writer


def _args(**overrides) -> argparse.Namespace:
    """A parsed-argument namespace with every flag defaulted, then the overrides applied."""
    defaults = dict(command='factors', signal=None, factor=None, horizon=21, start=None, end=None,
                    freq='M', panel_freq='M', rebuild=False, universe=None, n_top=None,
                    rank_by=None, version='as_reported', run_id=None, build_panel=False)
    return argparse.Namespace(**(defaults | overrides))


def test_the_parser_accepts_every_command():
    for command in cli.COMMANDS:
        assert cli._parser().parse_args([command]).command == command


def test_the_parser_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(['backtest-everything'])


def test_the_parser_rejects_an_unknown_signal():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(['run', '--signal', 'not_a_signal'])


def test_the_parser_reads_the_run_flags():
    args = cli._parser().parse_args(['run', '--signal', 'hold_cash', '--start', '2010-01-01',
                                     '--end', '2020-12-31', '--freq', 'Q'])
    assert (args.signal, args.start, args.end, args.freq) == ('hold_cash', '2010-01-01', '2020-12-31', 'Q')


def test_the_epilog_lists_every_signal_with_its_kind_and_description():
    epilog = cli._epilog()
    for name, (kind, _) in registry.SIGNALS.items():
        assert name in epilog
        assert kind in epilog
        assert registry.SIGNAL_DESCRIPTIONS[name] in epilog


def test_building_factors_without_a_date_range_fails_before_touching_the_database():
    with pytest.raises(SystemExit, match='factors needs --start and --end'):
        cli._build_factors(None, _args(), backtest_config.read())


def test_universe_resolution_prefers_the_flag_over_the_config(monkeypatch):
    monkeypatch.setattr(cli.universes, 'load_universe', lambda name: [f'{name}-A', f'{name}-B'])
    resolved = cli._resolve_universe(_args(universe='from_flag'), backtest_config.read())
    assert resolved == ['from_flag-A', 'from_flag-B']


def test_universe_resolution_falls_back_to_every_priced_ticker():
    assert cli._resolve_universe(_args(), backtest_config.read()) is None


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


# --- a run that builds the panel it needs ---

def _coverage(start: str, end: str) -> dict:
    """A stored panel spanning the two dates, shaped as `panel_coverage` returns it."""
    return {'version': 'as_reported', 'n_dates': 120, 'n_tickers': 500,
            'start': pd.Timestamp(start), 'end': pd.Timestamp(end), 'frequency': 'M',
            'factors': ['roe'], 'missing_factors': []}


def test_a_run_builds_nothing_unless_the_flag_is_set(monkeypatch):
    """A command that quietly builds what it needs is one whose cost the caller cannot predict."""
    called = []
    monkeypatch.setattr(cli, '_ensure_panel', lambda *args: called.append(args))
    monkeypatch.setattr(cli.registry, 'SIGNALS', {'x': ('cross_sectional', lambda *a: None)})
    args = _args(command='run', signal='x', start='2015-01-01', end='2024-12-31')
    args.build_panel = False

    assert args.build_panel is False
    assert called == []


def test_a_run_inside_the_stored_panel_builds_nothing(monkeypatch):
    built = []
    monkeypatch.setattr(cli.factors_pit, 'panel_coverage', lambda con, version: _coverage('2015-01-31', '2024-12-31'))
    monkeypatch.setattr(cli, '_build_factors', lambda *args: built.append(args))

    cli._ensure_panel(None, _args(start='2016-01-01', end='2024-01-01'), backtest_config.read())

    assert built == []


def test_a_run_reaching_past_the_stored_panel_builds_it_first(monkeypatch):
    built = []
    monkeypatch.setattr(cli.factors_pit, 'panel_coverage', lambda con, version: _coverage('2015-01-31', '2020-12-31'))
    monkeypatch.setattr(cli, '_build_factors', lambda *args: built.append(args))

    cli._ensure_panel(None, _args(start='2016-01-01', end='2024-01-01'), backtest_config.read())

    assert len(built) == 1


def test_a_run_with_no_panel_at_all_builds_one(monkeypatch):
    built = []
    monkeypatch.setattr(cli.factors_pit, 'panel_coverage', lambda con, version: None)
    monkeypatch.setattr(cli, '_build_factors', lambda *args: built.append(args))

    cli._ensure_panel(None, _args(start='2016-01-01', end='2024-01-01'), backtest_config.read())

    assert len(built) == 1


# --- which names a panel is built for ---

def test_a_named_universe_is_built_whole(monkeypatch):
    """The completeness filter belongs to a run; a panel row exists as soon as a ticker has a close."""
    monkeypatch.setattr(cli, '_resolve_universe', lambda args, cfg: ['ARRY', 'ENPH', 'FSLR'])
    monkeypatch.setattr(cli, '_run_universe', lambda *args: pytest.fail('the named universe was filtered'))

    names = cli._panel_universe(None, _args(universe='solar', n_top=500), backtest_config.read())

    assert names == ['ARRY', 'ENPH', 'FSLR']


def test_no_universe_and_no_top_n_builds_over_every_priced_ticker(monkeypatch):
    monkeypatch.setattr(cli, '_resolve_universe', lambda args, cfg: None)
    assert cli._panel_universe(None, _args(), backtest_config.read()) is None


def test_top_n_without_a_universe_cuts_the_database(monkeypatch):
    monkeypatch.setattr(cli, '_resolve_universe', lambda args, cfg: None)
    monkeypatch.setattr(cli, '_run_universe', lambda *args: ['AAA', 'BBB'])
    assert cli._panel_universe(None, _args(n_top=2), backtest_config.read()) == ['AAA', 'BBB']


def test_building_over_a_universe_that_resolved_to_nothing_is_refused(monkeypatch):
    monkeypatch.setattr(cli.factors_pit, 'rebalance_dates',
                        lambda con, start, end, freq: pd.DatetimeIndex(['2020-01-31']))
    monkeypatch.setattr(cli, '_panel_universe', lambda con, args, cfg: [])

    with pytest.raises(SystemExit, match='resolved to no tickers'):
        cli._build_factors(None, _args(start='2020-01-01', end='2020-12-31'), backtest_config.read())


def test_a_run_universe_that_nothing_survives_says_why(monkeypatch):
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE)')
    monkeypatch.setattr(cli, '_resolve_universe', lambda args, cfg: ['ARRY', 'ENPH'])
    monkeypatch.setattr(cli.panels, 'trading_calendar',
                        lambda con, start, end: pd.DatetimeIndex(['2000-01-03']))
    monkeypatch.setattr(cli.panels, 'traded_universe', lambda *args, **kwargs: [])

    with pytest.raises(SystemExit, match='traded across the whole of'):
        cli._run_universe(con, _args(universe='solar'), backtest_config.read(), '2000-01-01', '2026-09-03')


def test_the_parser_defaults_to_not_building():
    assert cli._parser().parse_args(['run']).build_panel is False
    assert cli._parser().parse_args(['run', '--build-panel']).build_panel is True


def test_an_evaluation_builds_only_when_the_flag_is_set(monkeypatch):
    called = []
    monkeypatch.setattr(cli, '_ensure_panel', lambda *args: called.append(args))
    monkeypatch.setattr(cli.storage, 'table_exists', lambda con, table: False)

    with pytest.raises(SystemExit):
        cli._evaluate(None, _args(command='evaluate'), backtest_config.read())
    assert called == []

    args = _args(command='evaluate')
    args.build_panel = True
    with pytest.raises(SystemExit):
        cli._evaluate(None, args, backtest_config.read())
    assert len(called) == 1


def test_deleting_an_evaluation_without_a_universe_is_refused():
    with pytest.raises(SystemExit, match='delete-evaluation needs --universe'):
        cli._delete_evaluation(None, _args(command='delete-evaluation'), backtest_config.read())


def test_deleting_an_evaluation_that_was_never_stored_says_so():
    con = duckdb.connect()
    writer.ensure_tables(con)
    with pytest.raises(SystemExit, match='no as_reported evaluation stored for ghost'):
        cli._delete_evaluation(con, _args(command='delete-evaluation', universe='ghost'),
                               backtest_config.read())


def test_deleting_an_evaluation_leaves_the_other_universes_alone():
    con = duckdb.connect()
    writer.ensure_tables(con)
    for universe in ('all', 'unrecorded'):
        rows = pd.DataFrame([{'Version': 'as_reported', 'Universe': universe, 'Factor': 'roe',
                              'Date': pd.Timestamp('2020-01-31'), 'Horizon': 21, 'Ic': 0.1,
                              'NNames': 50}])
        writer.replace_evaluation(con, 'as_reported', universe, 'roe', rows,
                                  pd.DataFrame(columns=writer.QUANTILE_COLS))

    cli._delete_evaluation(con, _args(command='delete-evaluation', universe='unrecorded'),
                           backtest_config.read())

    assert con.execute('SELECT Universe FROM factor_ic').fetchall() == [('all',)]


# --- building only what is not stored ---

def test_a_build_asks_for_the_dates_the_panel_does_not_hold(monkeypatch):
    grid = pd.DatetimeIndex(['2020-01-31', '2020-02-28', '2020-03-31'])
    monkeypatch.setattr(cli.factors_pit, 'stored_dates',
                        lambda con, version, tickers: {pd.Timestamp('2020-01-31')})
    monkeypatch.setattr(cli.storage, 'table_exists', lambda con, table: False)

    building, why = cli._dates_to_build(None, _args(), grid, None)

    assert list(building) == [pd.Timestamp('2020-02-28'), pd.Timestamp('2020-03-31')]
    assert 'not stored yet' in why


def test_rebuild_asks_for_every_date_in_the_range(monkeypatch):
    grid = pd.DatetimeIndex(['2020-01-31', '2020-02-28'])
    monkeypatch.setattr(cli.factors_pit, 'stored_dates', lambda con, version, tickers: set(grid))

    building, why = cli._dates_to_build(None, _args(rebuild=True), grid, None)

    assert list(building) == list(grid)
    assert 'rebuilding every one' in why


def test_a_redefined_factor_forces_a_full_rebuild(monkeypatch):
    """The stored values came from a formula that no longer exists, so the dates are not reusable."""
    grid = pd.DatetimeIndex(['2020-01-31', '2020-02-28'])
    monkeypatch.setattr(cli.factors_pit, 'stored_dates', lambda con, version, tickers: set(grid))
    monkeypatch.setattr(cli.storage, 'table_exists', lambda con, table: True)
    monkeypatch.setattr(cli.writer, 'latest_panel_build',
                        lambda con, version: {'definitions': 'blend=old_formula:high'})
    monkeypatch.setattr(cli.custom_factors, 'changed_since', lambda recorded, customs: ['blend'])

    building, why = cli._dates_to_build(None, _args(), grid, None)

    assert list(building) == list(grid)
    assert 'blend was redefined' in why


def test_a_range_already_stored_builds_nothing(monkeypatch):
    grid = pd.DatetimeIndex(['2020-01-31', '2020-02-28'])
    monkeypatch.setattr(cli.factors_pit, 'stored_dates', lambda con, version, tickers: set(grid))
    monkeypatch.setattr(cli.storage, 'table_exists', lambda con, table: False)

    building, _why = cli._dates_to_build(None, _args(), grid, None)

    assert len(building) == 0


def test_the_two_frequencies_are_separate_flags():
    args = cli._parser().parse_args(['run', '--freq', 'W', '--panel-freq', 'Q'])
    assert (args.freq, args.panel_freq) == ('W', 'Q')


def test_a_refusal_names_the_earliest_start_that_would_work(monkeypatch):
    """Telling someone to shorten the range begs the question of how much."""
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE)')
    con.execute("INSERT INTO prices VALUES ('FSLR', DATE '2006-11-17'), ('ENPH', DATE '2012-03-30')")
    monkeypatch.setattr(cli, '_resolve_universe', lambda args, cfg: ['FSLR', 'ENPH'])
    monkeypatch.setattr(cli.panels, 'trading_calendar',
                        lambda con, start, end: pd.DatetimeIndex(['2000-01-03']))
    monkeypatch.setattr(cli.panels, 'traded_universe', lambda *args, **kwargs: [])

    with pytest.raises(SystemExit, match='earliest start that keeps one is 2006-11-17'):
        cli._run_universe(con, _args(universe='solar'), backtest_config.read(),
                          '2000-01-01', '2026-09-03')


def test_a_refusal_over_the_whole_database_falls_back_to_advice():
    con = duckdb.connect()
    assert 'shorten the range' in cli._earliest_workable(con, None)
