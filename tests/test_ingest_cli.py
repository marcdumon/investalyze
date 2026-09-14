"""Tests for the ingest CLI's argument checks and command dispatch."""

import pytest

from investalyze.ingest import __main__ as cli


def _problem(*argv: str) -> str | None:
    return cli._argument_problem(cli._parser().parse_args(argv))


def test_a_blacklist_names_a_reason_and_one_ticker_source():
    assert _problem('blacklist-ticker', '--reason', 'delisted', '--ticker', 'AAA') is None
    assert _problem('blacklist-ticker', '--reason', 'delisted', '--universe', 'nasdaq_100') is None
    assert 'requires --reason' in _problem('blacklist-ticker', '--ticker', 'AAA')
    assert 'exactly one of' in _problem('blacklist-ticker', '--reason', 'r', '--ticker', 'AAA', '--universe', 'u')


def test_adding_listings_needs_the_reviewed_rows():
    assert _problem('add-listings') == 'add-listings requires --rows'
    assert _problem('add-listings', '--rows', '[]') is None


@pytest.mark.parametrize('command', [None, 'setup', 'housekeeping', 'scan-listings', 'blacklist-ticker', 'add-listings'])
def test_every_command_has_a_runner(command: str | None):
    assert callable(cli._COMMANDS[command])
