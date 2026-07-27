"""Tests for the control-panel page's pure argv-building helpers."""

from investalyze.apps.control_panel import page


def test_build_listings_scan_argv():
    title, argv = page.build_listings_scan_argv()
    assert title == 'scan listings'
    assert argv == ['-m', 'investalyze.ingest', 'scan-listings']
