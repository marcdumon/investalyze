"""Column-oriented dimensionality reduction: same windows-in-columns convention as `encodings`.

Each input is a 2-D array with one window per column and time (or feature) down the rows.
Reducing dimensionality before an exact search (e.g. faiss `IndexFlatL2`) keeps the search exact
but cuts its cost.
"""

from typing import Self

import numpy as np
from sklearn.decomposition import PCA


class PCAReduce:
    """Keep either a fixed number of components, or the fewest explaining a given variance share.

    sklearn's PCA takes both as a single `n_components` argument and infers which one you meant
    from its Python type (int vs float), so passing an int where you meant a variance share (or
    vice versa) is silently reinterpreted instead of raising. `variance` and `n_components` are
    kept as separate, validated keywords here to rule that out.
    """

    def __init__(self, variance: float | None = None, n_components: int | None = None) -> None:
        if (variance is None) == (n_components is None):
            raise ValueError('pass exactly one of variance or n_components')
        if variance is not None and not (isinstance(variance, float) and 0.0 < variance < 1.0):
            raise ValueError('variance must be a float in (0, 1)')
        if n_components is not None and not (isinstance(n_components, int) and n_components >= 1):
            raise ValueError('n_components must be a positive int')
        self.variance = variance
        self.n_components = n_components

    def fit(self, x: np.ndarray) -> Self:
        """Fit PCA on `x`'s windows (columns), keeping `n_components` components or enough for `variance`."""
        target = self.n_components if self.n_components is not None else self.variance
        self.pca_ = PCA(n_components=target, svd_solver='full').fit(x.T)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project `x`'s windows onto the fitted components; output has one row per component."""
        return self.pca_.transform(x.T).T
