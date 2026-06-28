"""Unit tests for the pure vector encodings."""
import numpy as np

from investalyze.analysis import encodings


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
