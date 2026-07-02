"""Unit tests for the column-oriented encoders (windows in columns)."""
import numpy as np

from investalyze.analysis import encodings


def test_zscore_standardises_each_column():
    windows = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # 2 windows, 3 time steps
    out = encodings.zscore().fit(windows).transform(windows)
    assert np.allclose(out.mean(axis=0), 0.0)
    assert np.allclose(out.std(axis=0), 1.0)
    assert np.allclose(out[:, 0], out[:, 1])  # same shape -> same z-scores regardless of scale


def test_minmax_maps_each_column_to_unit_range():
    windows = np.array([[10.0, 5.0], [12.0, 0.0], [14.0, 10.0]])
    out = encodings.minmax().fit(windows).transform(windows)
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)
    assert np.allclose(out[:, 0], [0.0, 0.5, 1.0])


def test_demean_centres_each_column():
    windows = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out = encodings.demean().fit(windows).transform(windows)
    assert np.allclose(out.mean(axis=0), 0.0)
    assert np.allclose(out[:, 0], [-1.0, 0.0, 1.0])
    assert np.allclose(out[:, 1], [-10.0, 0.0, 10.0])


def test_zscore_flat_window_encodes_to_zero_not_nan():
    flat = np.array([[5.0], [5.0], [5.0]])  # constant window, std 0
    out = encodings.zscore().fit(flat).transform(flat)
    assert not np.isnan(out).any()
    assert np.allclose(out, 0.0)


def test_minmax_flat_window_encodes_to_zero_not_nan():
    flat = np.array([[5.0], [5.0], [5.0]])  # constant window, range 0
    out = encodings.minmax().fit(flat).transform(flat)
    assert not np.isnan(out).any()
    assert np.allclose(out, 0.0)


def test_scaler_future_uses_history_params():
    history = np.array([[1.0], [2.0], [3.0]])  # mean 2
    future = np.array([[5.0]])
    scaler = encodings.demean().fit(history)
    assert np.allclose(scaler.transform(future), [[3.0]])  # 5 - 2


def test_minmax_future_may_exceed_unit_range():
    history = np.array([[10.0], [20.0]])  # range [10, 20]
    future = np.array([[30.0]])
    scaler = encodings.minmax().fit(history)
    assert np.allclose(scaler.transform(future), [[2.0]])  # (30 - 10) / 10, outside [0, 1] by design


def test_rebase_starts_at_100_per_column():
    windows = np.array([[10.0, 20.0], [11.0, 25.0], [12.0, 30.0]])
    out = encodings.RebaseTo100().fit(windows).transform(windows)
    assert np.allclose(out[0, :], 100.0)
    assert np.allclose(out[:, 0], [100.0, 110.0, 120.0])
    assert np.allclose(out[:, 1], [100.0, 125.0, 150.0])


def test_rebase_future_continuous_with_history():
    history = np.array([[10.0], [11.0]])
    future = np.array([[12.0]])
    rebase = encodings.RebaseTo100().fit(history)
    assert np.allclose(rebase.transform(history), [[100.0], [110.0]])
    assert np.allclose(rebase.transform(future), [[120.0]])  # same base as the history


def test_log_returns_history_shape_and_values():
    windows = np.array([[1.0], [np.e], [np.e ** 2]])
    out = encodings.LogReturns().fit(windows).transform(windows)
    assert out.shape == (2, 1)
    assert np.allclose(out, [[1.0], [1.0]])


def test_log_returns_future_includes_bridge_step():
    history = np.array([[1.0], [np.e]])   # last price e
    future = np.array([[np.e ** 2]])
    logret = encodings.LogReturns().fit(history)
    assert np.allclose(logret.transform(history), [[1.0]])
    assert np.allclose(logret.transform_future(future), [[1.0]])  # log(e**2 / e), measured against history end


def test_rebase_handles_empty_windows():
    empty = np.empty((4, 0))
    out = encodings.RebaseTo100().fit(empty).transform(empty)
    assert out.shape == (4, 0)


def test_encoded_history_independent_of_future():
    history = np.array([[10.0], [11.0], [12.0], [13.0]])
    scaler = encodings.zscore().fit(history)
    enc = scaler.transform(history)
    scaler.transform(np.array([[999.0]]))  # transforming a future must not alter the fitted params
    assert np.array_equal(enc, scaler.transform(history))
