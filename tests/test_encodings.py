"""Unit tests for the pure vector encodings."""
import numpy as np
import pytest

from investalyze.analysis import encodings

_ALL = [encodings.rebase_to_100, encodings.log_returns, encodings.demean, encodings.zscore, encodings.minmax]


def test_rebase_to_100_starts_at_100_and_math():
    windows = np.array([[10.0, 11.0, 12.0], [20.0, 25.0, 30.0]])
    out = encodings.rebase_to_100(windows)
    assert np.allclose(out[:, 0], 100.0)
    assert np.allclose(out[0], [100.0, 110.0, 120.0])
    assert np.allclose(out[1], [100.0, 125.0, 150.0])


def test_rebase_to_100_rows_are_independent():
    # each row rebased by its OWN first value, not a shared one
    windows = np.array([[5.0, 10.0], [50.0, 100.0]])
    out = encodings.rebase_to_100(windows)
    assert np.allclose(out, [[100.0, 200.0], [100.0, 200.0]])


def test_log_returns_shape_and_values():
    windows = np.array([[1.0, np.e, np.e ** 2]])
    out = encodings.log_returns(windows)
    assert out.shape == (1, 2)
    assert np.allclose(out, [[1.0, 1.0]])


def test_log_returns_value_is_log_ratio():
    windows = np.array([[2.0, 6.0]])
    out = encodings.log_returns(windows)
    assert np.allclose(out, [[np.log(6.0 / 2.0)]])


def test_handles_empty_input():
    empty = np.empty((0, 4))
    assert encodings.rebase_to_100(empty).shape == (0, 4)
    assert encodings.log_returns(empty).shape == (0, 3)


def test_demean_subtracts_row_mean():
    windows = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = encodings.demean(windows)
    assert np.allclose(out.mean(axis=1), 0.0)
    assert np.allclose(out[0], [-1.0, 0.0, 1.0])
    assert np.allclose(out[1], [-10.0, 0.0, 10.0])


def test_zscore_mean_zero_std_one():
    windows = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = encodings.zscore(windows)
    assert np.allclose(out.mean(axis=1), 0.0)
    assert np.allclose(out.std(axis=1), 1.0)
    # same shape -> identical z-scores regardless of scale
    assert np.allclose(out[0], out[1])


def test_minmax_maps_to_unit_range():
    windows = np.array([[10.0, 12.0, 14.0], [5.0, 0.0, 10.0]])
    out = encodings.minmax(windows)
    assert np.allclose(out.min(axis=1), 0.0)
    assert np.allclose(out.max(axis=1), 1.0)
    assert np.allclose(out[0], [0.0, 0.5, 1.0])


# --- future transformed by the history's parameters ---------------------------
def test_rebase_future_continuous_with_history():
    history = np.array([[10.0, 11.0]])
    future = np.array([[12.0]])
    enc_hist, enc_fut = encodings.rebase_to_100(history, future)
    assert np.allclose(enc_hist, [[100.0, 110.0]])
    assert np.allclose(enc_fut, [[120.0]])  # same base as the history


def test_demean_future_uses_history_mean():
    history = np.array([[1.0, 2.0, 3.0]])  # mean 2
    future = np.array([[5.0]])
    _, enc_fut = encodings.demean(history, future)
    assert np.allclose(enc_fut, [[3.0]])  # 5 - 2


def test_minmax_future_may_exceed_unit_range():
    history = np.array([[10.0, 20.0]])  # range [10, 20]
    future = np.array([[30.0]])
    _, enc_fut = encodings.minmax(history, future)
    assert np.allclose(enc_fut, [[2.0]])  # (30 - 10) / 10, outside [0, 1] by design


def test_log_returns_future_includes_bridge_step():
    history = np.array([[1.0, np.e]])      # last price e
    future = np.array([[np.e ** 2]])
    enc_hist, enc_fut = encodings.log_returns(history, future)
    assert np.allclose(enc_hist, [[1.0]])
    assert np.allclose(enc_fut, [[1.0]])  # log(e**2 / e) — measured against the history end


@pytest.mark.parametrize('fn', _ALL)
def test_no_future_leak_into_history(fn):
    history = np.array([[10.0, 11.0, 12.0, 13.0]])
    future = np.array([[14.0, 15.0]])
    enc_a, _ = fn(history, future)
    enc_b, _ = fn(history, np.array([[999.0, 0.001]]))  # wreck the future
    assert np.array_equal(enc_a, enc_b)                 # encoded history is unchanged
