"""Tests for backtest configuration reading: defaults and overrides.

Every key is optional, so the tests that matter are the ones proving a partial file still resolves
Frequencies are deliberately not read from here: a cadence that lived in the file would change a
result without appearing on the page that asked for it, so both are stated by the caller.
"""

from pathlib import Path

import pytest

from investalyze.backtest import config


def _write(tmp_path: Path, text: str) -> Path:
    """Write a backtest TOML into the temporary directory and return its path."""
    path = tmp_path / 'backtest.toml'
    path.write_text(text)
    return path


def test_config_falls_back_to_defaults_when_the_toml_is_absent(tmp_path: Path):
    cfg = config.read(tmp_path / 'missing.toml')
    assert cfg.starting_cash == config._DEFAULT_STARTING_CASH
    assert cfg.benchmark == config._DEFAULT_BENCHMARK
    assert cfg.long_only is config._DEFAULT_LONG_ONLY
    assert cfg.universe.n_top == config._DEFAULT_N_TOP
    assert cfg.costs.slippage_bps == config._DEFAULT_SLIPPAGE_BPS
    assert cfg.limits.max_weight == config._DEFAULT_MAX_WEIGHT


def test_config_falls_back_to_defaults_when_no_path_is_given():
    cfg = config.read()
    assert cfg.benchmark == config._DEFAULT_BENCHMARK


def test_config_reads_every_cost_and_limit_key(tmp_path: Path):
    path = _write(tmp_path, """
        [costs]
        slippage_bps = 7.5
        commission_bps = 3.0
        min_commission = 2.0
        impact_coef = 0.2
        borrow_bps_annual = 75.0
        cash_bps_annual = 400.0

        [limits]
        max_weight = 0.1
        max_gross = 2.0
        max_net = 0.5
        max_position_adv = 0.3
        participation = 0.02
        max_order_age = 3
    """)
    cfg = config.read(path)
    assert (cfg.costs.slippage_bps, cfg.costs.commission_bps, cfg.costs.min_commission) == (7.5, 3.0, 2.0)
    assert (cfg.costs.impact_coef, cfg.costs.borrow_bps_annual, cfg.costs.cash_bps_annual) == (0.2, 75.0, 400.0)
    assert (cfg.limits.max_weight, cfg.limits.max_gross, cfg.limits.max_net) == (0.1, 2.0, 0.5)
    assert (cfg.limits.max_position_adv, cfg.limits.participation, cfg.limits.max_order_age) == (0.3, 0.02, 3)


def test_a_partial_file_keeps_the_defaults_for_everything_it_omits(tmp_path: Path):
    path = _write(tmp_path, "benchmark = '^SPX'\n\n[costs]\nslippage_bps = 1.0\n")
    cfg = config.read(path)
    assert cfg.benchmark == '^SPX'
    assert cfg.costs.slippage_bps == 1.0
    assert cfg.costs.commission_bps == config._DEFAULT_COMMISSION_BPS
    assert cfg.limits.max_weight == config._DEFAULT_MAX_WEIGHT


def test_an_empty_universe_name_means_rank_the_whole_database(tmp_path: Path):
    path = _write(tmp_path, "[universe]\nname = ''\n")
    assert config.read(path).universe.name is None


def test_a_named_universe_is_kept_as_written(tmp_path: Path):
    path = _write(tmp_path, "[universe]\nname = 'mid_large_mega_non_fin_realest'\n")
    assert config.read(path).universe.name == 'mid_large_mega_non_fin_realest'


def test_signal_parameters_are_returned_per_signal(tmp_path: Path):
    path = _write(tmp_path, '[signals.factor_composite]\nfactors = ["ret_12_1", "roe"]\ntop_n = 50\n')
    cfg = config.read(path)
    assert cfg.signal('factor_composite') == {'factors': ['ret_12_1', 'roe'], 'top_n': 50}
    assert cfg.signal('never_configured') == {}


def test_the_repo_backtest_toml_reads_cleanly():
    cfg = config.read(Path('backtest.toml'))
    assert cfg.costs.slippage_bps > 0


def test_no_frequency_is_read_from_the_file():
    """A cadence here would change a run without showing on the card that asked for it."""
    cfg = config.read()
    assert not hasattr(cfg, 'rebalance')
    assert config.REBALANCE_FREQUENCIES == ('D', 'W', 'M', 'Q')
