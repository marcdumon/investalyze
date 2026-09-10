"""Tests for the backtest writer's evaluation tables: the period+horizon key, and the pre-migration backfill."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from investalyze.backtest import writer


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with every backtest table created."""
    con = duckdb.connect()
    writer.ensure_tables(con)
    return con


def _ic_rows(dates: list[date], horizon: int, ic: float = 0.1, factor: str = 'roe') -> pd.DataFrame:
    return pd.DataFrame({'Version': 'as_reported', 'Universe': 'all', 'Factor': factor, 'Date': dates,
                         'Horizon': horizon, 'Ic': ic, 'NNames': 10}, columns=writer.IC_COLS)


def _quantile_rows(dates: list[date], horizon: int, factor: str = 'roe') -> pd.DataFrame:
    rows = []
    for d in dates:
        for quantile in (0, 1):
            rows.append({'Version': 'as_reported', 'Universe': 'all', 'Factor': factor, 'Date': d,
                        'Horizon': horizon, 'Quantile': quantile, 'Return': 0.01, 'MedianReturn': 0.01,
                        'NNames': 5})
    return pd.DataFrame(rows, columns=writer.QUANTILE_COLS)


JAN = [date(2020, 1, 31), date(2020, 2, 28)]
JUN = [date(2020, 6, 30), date(2020, 7, 31)]


# --- schema ---

def test_ensure_tables_adds_period_columns_to_factor_ic(con):
    columns = {row[0] for row in con.execute('DESCRIBE factor_ic').fetchall()}
    assert {'StartDate', 'EndDate'} <= columns


def test_ensure_tables_adds_period_columns_to_factor_quantiles(con):
    columns = {row[0] for row in con.execute('DESCRIBE factor_quantiles').fetchall()}
    assert {'StartDate', 'EndDate'} <= columns


def test_ensure_tables_backfills_period_from_a_pre_migration_row(con):
    """A row written before StartDate/EndDate existed gets them from its own Date column."""
    con.execute("""
        INSERT INTO factor_ic (Version, Universe, Factor, Date, Horizon, Ic, NNames)
        VALUES ('as_reported', 'all', 'roe', '2020-01-31', 21, 0.1, 10),
              ('as_reported', 'all', 'roe', '2020-02-28', 21, 0.2, 10)
    """)
    writer.ensure_tables(con)
    row = con.execute("SELECT StartDate, EndDate FROM factor_ic WHERE Factor = 'roe'").fetchone()
    assert row == (date(2020, 1, 31), date(2020, 2, 28))


# --- replace_evaluation: keyed by horizon and period ---

def test_replace_evaluation_keeps_rows_from_a_different_horizon(con):
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JAN, horizon=21), _quantile_rows(JAN, horizon=21),
                              JAN[0], JAN[1])
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JAN, horizon=5), _quantile_rows(JAN, horizon=5),
                              JAN[0], JAN[1])
    horizons = {row[0] for row in con.execute('SELECT DISTINCT Horizon FROM factor_ic').fetchall()}
    assert horizons == {21, 5}


def test_replace_evaluation_keeps_rows_from_a_different_period(con):
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JAN, horizon=21), _quantile_rows(JAN, horizon=21),
                              JAN[0], JAN[1])
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JUN, horizon=21), _quantile_rows(JUN, horizon=21),
                              JUN[0], JUN[1])
    n = con.execute('SELECT count(*) FROM factor_ic').fetchone()[0]
    assert n == 4


def test_replace_evaluation_overwrites_the_same_horizon_and_period(con):
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JAN, horizon=21, ic=0.1),
                              _quantile_rows(JAN, horizon=21), JAN[0], JAN[1])
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', _ic_rows(JAN, horizon=21, ic=0.9),
                              _quantile_rows(JAN, horizon=21), JAN[0], JAN[1])
    rows = con.execute('SELECT Ic FROM factor_ic ORDER BY Date').fetchall()
    assert [r[0] for r in rows] == [0.9, 0.9]


def test_replace_evaluation_stamps_the_period_the_caller_names(con):
    """The period is what the whole evaluate run covered, not what this one factor's own rows span.

    A factor needing warm-up data (a growth ratio with fewer comparable quarters, say) reads over a
    narrower range than the rest of the run; stamping from its own rows would fragment one run into
    several near-duplicate periods in the picker instead of the one it actually was.
    """
    narrow = _ic_rows(JAN[1:], horizon=21)   # only the later date, unlike the whole run's JAN..JUN span
    writer.replace_evaluation(con, 'as_reported', 'all', 'roe', narrow, _quantile_rows(JAN[1:], horizon=21),
                              JAN[0], JUN[1])
    row = con.execute('SELECT StartDate, EndDate FROM factor_ic LIMIT 1').fetchone()
    assert row == (JAN[0], JUN[1])


# --- factor_ic_decay ---

def _decay_rows(horizons: tuple[int, ...] = (5, 21)) -> pd.DataFrame:
    return pd.DataFrame({'horizon': horizons, 'mean_ic': [0.1] * len(horizons),
                         't_stat': [2.0] * len(horizons), 'n_periods': [50] * len(horizons)})


def test_ensure_tables_creates_factor_ic_decay(con):
    columns = [row[0] for row in con.execute('DESCRIBE factor_ic_decay').fetchall()]
    assert columns == ['Version', 'Universe', 'Factor', 'StartDate', 'EndDate', 'Horizon',
                       'MeanIc', 'TStat', 'NPeriods']


def test_replace_ic_decay_stores_one_row_per_horizon(con):
    writer.replace_ic_decay(con, 'as_reported', 'all', 'roe', JAN[0], JAN[1], _decay_rows((5, 21)))
    horizons = {row[0] for row in con.execute('SELECT Horizon FROM factor_ic_decay').fetchall()}
    assert horizons == {5, 21}


def test_replace_ic_decay_overwrites_the_same_factor_and_period(con):
    writer.replace_ic_decay(con, 'as_reported', 'all', 'roe', JAN[0], JAN[1], _decay_rows((5, 21)))
    writer.replace_ic_decay(con, 'as_reported', 'all', 'roe', JAN[0], JAN[1], _decay_rows((63,)))
    rows = con.execute('SELECT Horizon FROM factor_ic_decay').fetchall()
    assert [r[0] for r in rows] == [63]


def test_replace_ic_decay_keeps_rows_from_a_different_period(con):
    writer.replace_ic_decay(con, 'as_reported', 'all', 'roe', JAN[0], JAN[1], _decay_rows((5,)))
    writer.replace_ic_decay(con, 'as_reported', 'all', 'roe', JUN[0], JUN[1], _decay_rows((5,)))
    n = con.execute('SELECT count(*) FROM factor_ic_decay').fetchone()[0]
    assert n == 2


# --- delete_evaluation ---

def _seed_evaluation(con, universe: str, dates: list[date], horizon: int = 21) -> None:
    ic = _ic_rows(dates, horizon).assign(Universe=universe)
    quantiles = _quantile_rows(dates, horizon).assign(Universe=universe)
    writer.replace_evaluation(con, 'as_reported', universe, 'roe', ic, quantiles, dates[0], dates[-1])
    writer.replace_ic_decay(con, 'as_reported', universe, 'roe', dates[0], dates[-1], _decay_rows((5,)))


def test_delete_evaluation_also_removes_ic_decay_rows(con):
    _seed_evaluation(con, 'all', JAN)
    removed = writer.delete_evaluation(con, 'as_reported', 'all')
    assert removed['factor_ic_decay'] == 1
    assert con.execute('SELECT count(*) FROM factor_ic_decay').fetchone()[0] == 0


def test_delete_evaluation_without_a_period_removes_every_period(con):
    _seed_evaluation(con, 'all', JAN)
    _seed_evaluation(con, 'all', JUN)
    removed = writer.delete_evaluation(con, 'as_reported', 'all')
    assert removed == {'factor_ic': 4, 'factor_quantiles': 8, 'factor_ic_decay': 2}
    assert con.execute('SELECT count(*) FROM factor_ic').fetchone()[0] == 0


def test_delete_evaluation_with_a_period_leaves_other_periods_alone(con):
    _seed_evaluation(con, 'all', JAN)
    _seed_evaluation(con, 'all', JUN)
    removed = writer.delete_evaluation(con, 'as_reported', 'all', JAN[0], JAN[1])
    assert removed == {'factor_ic': 2, 'factor_quantiles': 4, 'factor_ic_decay': 1}
    remaining = con.execute('SELECT DISTINCT StartDate FROM factor_ic').fetchall()
    assert remaining == [(JUN[0],)]


def test_delete_evaluation_with_a_period_leaves_other_universes_alone(con):
    _seed_evaluation(con, 'all', JAN)
    _seed_evaluation(con, 'unrecorded', JAN)
    writer.delete_evaluation(con, 'as_reported', 'all', JAN[0], JAN[1])
    remaining = {row[0] for row in con.execute('SELECT Universe FROM factor_ic').fetchall()}
    assert remaining == {'unrecorded'}
