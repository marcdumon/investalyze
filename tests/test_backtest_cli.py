"""Tests for the backtest CLI surface: argument parsing, the signal catalog, and universe resolution.

The commands themselves are covered where their modules are tested; what matters here is that every
command is reachable, that the catalog shown in `--help` stays in step with the registry, and that a
command missing a required argument fails before it opens the database.
"""

import argparse

import pytest

from investalyze.backtest import __main__ as cli
from investalyze.backtest import config as backtest_config
from investalyze.backtest import registry


def _args(**overrides) -> argparse.Namespace:
    """A parsed-argument namespace with every flag defaulted, then the overrides applied."""
    defaults = dict(command='factors', signal=None, factor=None, horizon=21, start=None, end=None,
                    freq=None, universe=None, vintage='as_reported', run_id=None)
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
