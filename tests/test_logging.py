"""Tests for console logging setup."""
import logging

from investalyze.ingest.logging import _ColorFormatter, configure_logging


def test_configure_installs_single_colored_handler():
    configure_logging('DEBUG')
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, _ColorFormatter)


def test_configure_silences_yfinance():
    configure_logging('INFO')
    assert logging.getLogger('yfinance').level == logging.CRITICAL
