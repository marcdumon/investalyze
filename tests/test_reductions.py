"""Unit tests for the column-oriented PCA reducer (windows in columns)."""
import numpy as np
import pytest

from investalyze.analysis import reductions


def test_pca_reduce_finds_minimal_components_for_collinear_windows():
    shape = np.array([1.0, 2.0, 3.0, 4.0])
    windows = np.column_stack([shape * k for k in (1.0, 2.0, 3.0, 4.0, 5.0)])  # all one direction
    reduced = reductions.PCAReduce(variance=0.999).fit(windows).transform(windows)
    assert reduced.shape == (1, windows.shape[1])


def test_pca_reduce_lower_variance_threshold_uses_fewer_or_equal_components():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(10, 50))  # independent noise, no dominant direction
    strict = reductions.PCAReduce(variance=0.999).fit(windows)
    loose = reductions.PCAReduce(variance=0.5).fit(windows)
    assert loose.pca_.n_components_ <= strict.pca_.n_components_


def test_pca_reduce_preserves_window_count():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(20, 30))
    out = reductions.PCAReduce(variance=0.95).fit(windows).transform(windows)
    assert out.shape[1] == windows.shape[1]


def test_pca_reduce_accepts_fixed_component_count():
    rng = np.random.default_rng(2)
    windows = rng.normal(size=(20, 30))
    out = reductions.PCAReduce(n_components=5).fit(windows).transform(windows)
    assert out.shape == (5, windows.shape[1])


def test_pca_reduce_rejects_both_arguments():
    with pytest.raises(ValueError):
        reductions.PCAReduce(variance=0.99, n_components=5)


def test_pca_reduce_rejects_neither_argument():
    with pytest.raises(ValueError):
        reductions.PCAReduce()


def test_pca_reduce_rejects_int_passed_as_variance():
    with pytest.raises(ValueError):
        reductions.PCAReduce(variance=1)


def test_pca_reduce_rejects_float_passed_as_n_components():
    with pytest.raises(ValueError):
        reductions.PCAReduce(n_components=5.0)
