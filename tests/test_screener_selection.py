"""Tests for the screener's curation state machine: checkbox edits, bulk buttons, universe load/save."""

from pathlib import Path

import pytest

from investalyze.apps import universes
from investalyze.apps.screener import page


@pytest.fixture
def universe_dir(tmp_path, monkeypatch) -> Path:
    """Redirect the page's universe helpers at a temp dir, holding one universe of AAA and BBB."""
    universes.save_universe('existing', ['AAA', 'BBB'], tmp_path)
    monkeypatch.setattr(page, 'list_universes', lambda: universes.list_universes(tmp_path))
    monkeypatch.setattr(page, 'load_universe', lambda name: universes.load_universe(name, tmp_path))
    monkeypatch.setattr(page, 'save_universe', lambda name, tickers: universes.save_universe(name, tickers, tmp_path))
    return tmp_path


@pytest.fixture
def saved(monkeypatch) -> None:
    """Serve one fixed universe, so the tests do not depend on what sits in data/universes/."""
    monkeypatch.setattr(page, 'load_universe', lambda name: ['AAA', 'BBB', 'CCC'])


def _checkbox(ticker: str, value: bool) -> dict:
    """One AG Grid cellValueChanged event for the selection checkbox column."""
    return {'colId': 'sel', 'value': value, 'data': {'Ticker': ticker}}


def test_picking_a_universe_replaces_the_selection(saved):
    selection, _ = page.apply_action('universe-dd', ['ZZZ'], [], [], 'anything')
    assert selection == ['AAA', 'BBB', 'CCC']


def test_picking_a_universe_reports_what_it_loaded(saved):
    _, message = page.apply_action('universe-dd', [], [], [], 'anything')
    assert message == "loaded 'anything' (3 tickers)"


def test_clearing_the_dropdown_keeps_the_selection(saved):
    selection, _ = page.apply_action('universe-dd', ['ZZZ', 'YYY'], [], [], None)
    assert selection == ['YYY', 'ZZZ']


def test_checking_a_row_adds_it():
    selection, _ = page.apply_action('grid', [], [_checkbox('AAA', True)], [], None)
    assert selection == ['AAA']


def test_unchecking_a_row_removes_it():
    selection, _ = page.apply_action('grid', ['AAA', 'BBB'], [_checkbox('AAA', False)], [], None)
    assert selection == ['BBB']


def test_an_edit_outside_the_checkbox_column_is_ignored():
    event = {'colId': 'sector', 'value': 'Tech', 'data': {'Ticker': 'AAA'}}
    selection, _ = page.apply_action('grid', ['BBB'], [event], [], None)
    assert selection == ['BBB']


def test_select_all_adds_every_filtered_ticker():
    selection, _ = page.apply_action('btn-select-all', ['AAA'], [], ['BBB', 'CCC'], None)
    assert selection == ['AAA', 'BBB', 'CCC']


def test_deselect_all_empties_the_selection():
    selection, _ = page.apply_action('btn-deselect-all', ['AAA', 'BBB'], [], [], None)
    assert selection == []


def test_the_selected_grid_only_removes():
    """Unchecking in the condensed table drops a ticker; it is the one grid that cannot add one."""
    selection, _ = page.apply_action('sel-grid', ['AAA', 'BBB'], [_checkbox('AAA', False)], [], None)
    assert selection == ['BBB']


def test_loading_a_universe_fills_the_save_box():
    assert page.name_the_loaded_universe('existing') == 'existing'


def test_clearing_the_dropdown_leaves_the_save_box_alone():
    """dash.no_update, so a stray clear cannot blank a name the user is about to save under."""
    assert page.name_the_loaded_universe(None) is not None


def test_adding_a_ticker_and_saving_writes_back_to_the_same_universe(universe_dir):
    selection, _ = page.apply_action('universe-dd', [], [], [], 'existing')
    selection, _ = page.apply_action('grid', selection, [_checkbox('CCC', True)], [], None)
    page.save_current(1, selection, page.name_the_loaded_universe('existing'))
    assert universes.load_universe('existing', universe_dir) == ['AAA', 'BBB', 'CCC']


def test_saving_over_an_existing_universe_creates_no_second_file(universe_dir):
    page.save_current(1, ['AAA', 'BBB', 'CCC'], 'existing')
    assert universes.list_universes(universe_dir) == ['existing']


def test_saving_over_an_existing_universe_says_it_updated_one(universe_dir):
    _, message = page.save_current(1, ['AAA', 'CCC'], 'existing')
    assert message == "updated 'existing' (2 tickers)"


def test_saving_under_a_new_name_says_it_saved(universe_dir):
    _, message = page.save_current(1, ['AAA'], 'fresh')
    assert message == "saved 'fresh' (1 tickers)"


def test_saving_without_a_name_is_refused(universe_dir):
    _, message = page.save_current(1, ['AAA'], '   ')
    assert message == 'enter a universe name first'


def test_saving_an_empty_selection_is_refused(universe_dir):
    _, message = page.save_current(1, [], 'existing')
    assert message == 'selection is empty, nothing saved'


def test_a_refused_save_leaves_the_existing_universe_untouched(universe_dir):
    page.save_current(1, [], 'existing')
    assert universes.load_universe('existing', universe_dir) == ['AAA', 'BBB']
