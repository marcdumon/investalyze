"""The app folder (the hand-edited settings and .env, next to the Python environment) and the data folder (the database,
the provider downloads, the backtests, the factor library and the quality log; data_dir of investalyze.toml)."""

import sys
import time
from pathlib import Path

from investalyze import paths
from investalyze.apps.control_panel.jobs import JobManager
from investalyze.apps.data_quality import page as quality_page
from investalyze.backtest import config as backtest_config
from investalyze.ingest import config as ingest_config
from investalyze.ingest.providers.stooq import instrument_names


def _settings(tmp_path: Path, monkeypatch, text: str) -> None:
    """Point the app to an investalyze.toml with this text."""
    settings = tmp_path / 'investalyze.toml'
    settings.write_text(text)
    monkeypatch.setattr(paths, 'SETTINGS_FILE', settings)


def test_the_app_folder_holds_the_python_environment():
    assert Path(sys.prefix).parent == paths.APP_DIR


def test_the_data_folder_is_data_dir_of_the_settings(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, f"data_dir = '{tmp_path / 'backed_up'}'\n")
    assert paths.data_dir() == tmp_path / 'backed_up'


def test_a_data_dir_that_starts_with_a_tilde_is_in_the_home_folder(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, "data_dir = '~/investalyze'\n")
    assert paths.data_dir() == Path.home() / 'investalyze'


def test_without_data_dir_the_data_folder_is_in_the_app_folder(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, "data_dir = ''\n")
    assert paths.data_dir() == paths.APP_DIR / 'data'
    monkeypatch.setattr(paths, 'SETTINGS_FILE', tmp_path / 'missing.toml')
    assert paths.data_dir() == paths.APP_DIR / 'data'


def test_the_server_settings_have_a_port_and_debug_mode(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, "[server]\nport = 18050\ndebug = true\n")
    assert paths.server() == (18050, True)
    monkeypatch.setattr(paths, 'SETTINGS_FILE', tmp_path / 'missing.toml')
    assert paths.server() == (8050, False)


def test_the_factor_library_and_the_quality_log_are_in_the_data_folder():
    assert paths.FACTORS_DIR == paths.DATA_ROOT / 'factors'
    assert quality_page.LOG_PATH == paths.DATA_ROOT / 'quality_log.toml'


def test_the_hand_edited_settings_are_in_the_app_folder():
    assert backtest_config.CONFIG_PATH == paths.APP_DIR / 'backtest.toml'
    assert instrument_names._REFERENCE_FILE == paths.APP_DIR / 'stooq_tickers.toml'
    assert paths.INGEST_CONFIG == paths.APP_DIR / 'ingest.toml'
    assert paths.CLEANING_CONFIG == paths.APP_DIR / 'cleaning.toml'


def test_an_ingest_config_without_data_root_uses_the_data_folder(tmp_path):
    settings = tmp_path / 'ingest.toml'
    settings.write_text("db = 'investalyze.duckdb'\n")
    assert ingest_config.read(settings).data_root == paths.DATA_ROOT
    assert ingest_config.read(None).data_root == paths.DATA_ROOT


def test_a_control_panel_job_runs_in_the_app_folder():
    manager = JobManager()
    assert manager.start('where', ['-c', 'import os; print(os.getcwd())'])
    for _ in range(100):
        if manager.history:
            break
        time.sleep(0.05)
    assert manager.history[0].lines == [str(paths.APP_DIR)]


def test_the_example_settings_have_an_empty_data_dir():
    assert "\ndata_dir = ''\n" in (Path(__file__).parents[1] / 'investalyze.example.toml').read_text()
