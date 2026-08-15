"""PCA on a correlation matrix of standardized returns: loadings, Marcenko-Pastur denoising, eigenportfolios.

Follows Lopez de Prado, *Machine Learning for Asset Managers*, ch. 2. Every function here is pure
numpy/pandas and takes matrices, never a database connection: `panels.py` owns the DuckDB side, so
the math can be verified against 3x3 inputs that are checkable by hand.

Vocabulary used throughout:
    loadings    columns of the eigenvector matrix, unit norm; these are the "PCA weights"
    PC series   the projection of standardized returns onto a loading vector, one value per date
    eigenportfolio  a loading converted to tradable weights by dividing out each asset's own volatility

Distinct from `investalyze.analysis.reductions.PCAReduce`, which compresses price *windows* before
FAISS indexing: that one is windows-in-columns, centers without scaling, and exposes no spectrum.
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

_SIGMA2_FLOOR = 1e-4  # smallest noise variance the Marcenko-Pastur fit will consider


class Eigen(NamedTuple):
    """Eigendecomposition of a correlation matrix, ordered by descending eigenvalue.

    `values` is shape (n,) descending; `vectors` is (n, n) with column k the unit-norm, sign-fixed
    loading vector for component k; `names` labels the rows of `vectors`.
    """

    values: np.ndarray
    vectors: np.ndarray
    names: pd.Index


def correlation(z: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the columns, which for standardized returns is also their covariance."""
    if z.isna().to_numpy().any():
        raise ValueError('z contains NaN; build the panel with complete observations before correlating')
    return z.corr()


def fix_signs(vectors: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude entry is positive, making the decomposition reproducible.

    An eigenvector is only defined up to sign, so two runs on near-identical data can return opposite
    vectors and make a stability comparison read as a total regime change. This is the rule sklearn's
    `svd_flip` applies, so results line up with anything else in the repo built on sklearn's PCA.

    The winning entry is the first one within a tolerance of the maximum rather than plain `argmax`:
    for an exactly symmetric pair such as (0.7071, -0.7071, 0), floating-point noise alone would
    otherwise decide the sign of the whole vector.
    """
    fixed = vectors.copy()
    for k in range(fixed.shape[1]):
        magnitudes = np.abs(fixed[:, k])
        leader = int(np.argmax(magnitudes >= magnitudes.max() - 1e-12))
        if fixed[leader, k] < 0:
            fixed[:, k] *= -1
    return fixed


def eigen_decompose(corr: pd.DataFrame | np.ndarray) -> Eigen:
    """Eigendecompose a correlation matrix, sorted by descending eigenvalue with signs fixed.

    Uses `eigh` rather than `eig` because a correlation matrix is symmetric: `eigh` guarantees real
    eigenvalues and an orthonormal basis, which `eig` only delivers up to numerical noise.
    """
    names = corr.columns if isinstance(corr, pd.DataFrame) else pd.RangeIndex(corr.shape[0])
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'corr must be square, got shape {matrix.shape}')
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError('corr must be symmetric')

    values, vectors = np.linalg.eigh(matrix)
    order = values.argsort()[::-1]
    return Eigen(values=values[order], vectors=fix_signs(vectors[:, order]), names=pd.Index(names))


def variance_explained(values: np.ndarray) -> np.ndarray:
    """Share of total variance carried by each component."""
    return values / values.sum()


def loadings_frame(eig: Eigen, *, k: int | None = None) -> pd.DataFrame:
    """The first `k` loading vectors as a labelled frame, assets down the rows and PC1..PCk across."""
    limit = eig.vectors.shape[1] if k is None else k
    if not 1 <= limit <= eig.vectors.shape[1]:
        raise ValueError(f'k must be between 1 and {eig.vectors.shape[1]}, got {k}')
    return pd.DataFrame(eig.vectors[:, :limit], index=eig.names, columns=[f'PC{i + 1}' for i in range(limit)])


# --- Marcenko-Pastur: which part of the spectrum is noise ---


def max_noise_eigenvalue(*, q: float, sigma2: float) -> float:
    """Upper edge of the Marcenko-Pastur support, above which an eigenvalue is too large to be noise."""
    _check_q(q)
    return sigma2 * (1.0 + math.sqrt(1.0 / q)) ** 2


def min_noise_eigenvalue(*, q: float, sigma2: float) -> float:
    """Lower edge of the Marcenko-Pastur support."""
    _check_q(q)
    return sigma2 * (1.0 - math.sqrt(1.0 / q)) ** 2


def marcenko_pastur_pdf(x: np.ndarray, *, q: float, sigma2: float) -> np.ndarray:
    """Density of the noise eigenvalues of a correlation matrix built from q = T/N iid observations.

    Zero outside the support [lambda_minus, lambda_plus]; inside it the density is
    q / (2 pi sigma2 x) * sqrt((lambda_plus - x)(x - lambda_minus)).
    """
    lo, hi = min_noise_eigenvalue(q=q, sigma2=sigma2), max_noise_eigenvalue(q=q, sigma2=sigma2)
    x = np.asarray(x, dtype=float)
    inside = (x >= lo) & (x <= hi) & (x > 0)
    density = np.zeros_like(x)
    density[inside] = q / (2.0 * np.pi * sigma2 * x[inside]) * np.sqrt((hi - x[inside]) * (x[inside] - lo))
    return density


def fit_sigma2(values: np.ndarray, *, q: float, bandwidth: float = 0.01, points: int = 1000, n_grid: int = 40) -> float:
    """Noise variance implied by the eigenvalue spectrum, by matching the Marcenko-Pastur density to it.

    The signal eigenvalues absorb part of the trace, so the noise bulk of a real correlation matrix sits
    below the variance 1.0 that a pure-noise matrix would show. Fitting sigma2 rather than assuming 1.0
    is what stops the threshold from being set too high and every genuine factor from being discarded.

    de Prado minimizes the same objective with `scipy.optimize.minimize`. A nested grid search over the
    single bounded scalar reaches the same answer without reaching for a dependency the project does not
    declare, and the kernel density is fitted once and reused across every candidate.
    """
    _check_q(q)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.asarray(values, dtype=float).reshape(-1, 1))

    lo, hi = _SIGMA2_FLOOR, 1.0
    best = hi
    for _ in range(4):
        grid = np.linspace(lo, hi, n_grid)
        errors = np.array([_mp_fit_error(candidate, kde, q=q, points=points) for candidate in grid])
        best = float(grid[int(errors.argmin())])
        step = float(grid[1] - grid[0])
        lo, hi = max(_SIGMA2_FLOOR, best - step), min(1.0, best + step)
    return best


def _mp_fit_error(sigma2: float, kde: KernelDensity, *, q: float, points: int) -> float:
    """Squared error between the fitted kernel density and the Marcenko-Pastur density for one sigma2."""
    lo, hi = min_noise_eigenvalue(q=q, sigma2=sigma2), max_noise_eigenvalue(q=q, sigma2=sigma2)
    support = np.linspace(lo, hi, points)
    theoretical = marcenko_pastur_pdf(support, q=q, sigma2=sigma2)
    empirical = np.exp(kde.score_samples(support.reshape(-1, 1)))
    return float(np.sum((empirical - theoretical) ** 2))


def fit_sigma2_iterative(values: np.ndarray, *, q: float, max_iter: int = 100, tol: float = 1e-10) -> float:
    """Noise variance from the requirement that it equal the mean of the eigenvalues it labels noise.

    A fixed point of sigma2 -> mean(values <= lambda_plus(sigma2)), started from 1.0. It needs no kernel
    density and so no bandwidth, which makes it the check on `fit_sigma2` rather than a replacement.

    On a spectrum whose noise bulk really is Marcenko-Pastur the two agree closely. On one with a long
    tail of near-zero eigenvalues they part company: every tiny eigenvalue drags the mean down, which
    lowers the threshold, which admits the next batch. A large gap between the two estimates says the
    bulk is not Marcenko-Pastur shaped, which is itself worth knowing before trusting either count.
    """
    _check_q(q)
    values = np.asarray(values, dtype=float)
    sigma2 = 1.0
    for _ in range(max_iter):
        noise = values[values <= max_noise_eigenvalue(q=q, sigma2=sigma2)]
        if len(noise) == 0:
            return sigma2
        updated = float(noise.mean())
        if abs(updated - sigma2) < tol:
            return updated
        sigma2 = updated
    return sigma2


def n_signal_factors(values: np.ndarray, *, q: float, sigma2: float) -> int:
    """How many eigenvalues sit above the Marcenko-Pastur noise edge."""
    threshold = max_noise_eigenvalue(q=q, sigma2=sigma2)
    return int((np.asarray(values) > threshold).sum())


def parallel_analysis_threshold(z: pd.DataFrame, *, n_trials: int = 100, seed: int = 0) -> float:
    """Largest eigenvalue reachable by chance, from Horn's parallel analysis.

    Shuffles each column independently in time, which destroys the cross-sectional correlation while
    leaving every marginal distribution intact, and reports the largest eigenvalue seen across trials.
    It assumes nothing about the shape of the return distribution, so it is an independent second
    opinion on the Marcenko-Pastur count, which does assume iid entries with a finite fourth moment.
    """
    rng = np.random.default_rng(seed)
    matrix = z.to_numpy(dtype=float)
    n_rows, n_cols = matrix.shape
    peaks = []
    for _ in range(n_trials):
        shuffled = np.empty_like(matrix)
        for column in range(n_cols):
            shuffled[:, column] = matrix[rng.permutation(n_rows), column]
        peaks.append(np.linalg.eigvalsh(np.corrcoef(shuffled, rowvar=False)).max())
    return float(np.max(peaks))


# --- shrinkage ---


def cov_to_corr(matrix: np.ndarray) -> np.ndarray:
    """Rescale a covariance-like matrix so its diagonal is exactly one."""
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def denoise(eig: Eigen, *, n_facts: int) -> np.ndarray:
    """Rebuild the correlation matrix with every noise eigenvalue replaced by the average of that group.

    de Prado's constant-residual-eigenvalue method. The top `n_facts` eigenpairs pass through untouched;
    flattening the rest keeps the trace, and so the unit diagonal, while removing the spurious spread
    among eigenvalues that sampling noise alone produced.
    """
    total = len(eig.values)
    if not 0 <= n_facts < total:
        raise ValueError(f'n_facts must be between 0 and {total - 1}, got {n_facts}')

    values = eig.values.copy()
    values[n_facts:] = values[n_facts:].mean()
    return cov_to_corr(eig.vectors @ np.diag(values) @ eig.vectors.T)


def detone(eig: Eigen, *, n_market: int = 1) -> np.ndarray:
    """Remove the top `n_market` components, then rescale back to a unit diagonal.

    The market component loads on nearly every name and dominates any distance or clustering built on the
    raw matrix, which mostly recovers beta. Projecting it out leaves the relative-value structure that
    PC2 and beyond are supposed to describe.

    The result is singular by construction, so it serves interpretation and clustering; a mean-variance
    solve that inverts it will fail.
    """
    total = len(eig.values)
    if not 1 <= n_market < total:
        raise ValueError(f'n_market must be between 1 and {total - 1}, got {n_market}')

    market = eig.vectors[:, :n_market] @ np.diag(eig.values[:n_market]) @ eig.vectors[:, :n_market].T
    full = eig.vectors @ np.diag(eig.values) @ eig.vectors.T
    return cov_to_corr(full - market)


# --- projections ---


def pc_series(z: pd.DataFrame, eig: Eigen, *, k: int) -> pd.DataFrame:
    """Project standardized returns onto the first `k` loadings, giving one time series per component.

    This is PC_k = w_1k x_1 + ... + w_pk x_p evaluated for every date. The columns come out with
    variance equal to their eigenvalue rather than one, which is the identity the tests check.
    """
    if not 1 <= k <= eig.vectors.shape[1]:
        raise ValueError(f'k must be between 1 and {eig.vectors.shape[1]}, got {k}')
    projected = z.to_numpy(dtype=float) @ eig.vectors[:, :k]
    return pd.DataFrame(projected, index=z.index, columns=[f'PC{i + 1}' for i in range(k)])


def eigenportfolios(eig: Eigen, vol: pd.Series, *, k: int, normalize: str = 'gross') -> pd.DataFrame:
    """Convert loadings into tradable asset weights by dividing out each asset's own volatility.

    A loading multiplies a *standardized* return, so it is not a position. Holding w_ik / sigma_i of each
    asset reproduces the component from raw returns, up to scale.

    Gross normalization sets the sum of absolute weights to one and is well defined for every component.
    Net normalization sets the signed sum to one and is unusable past PC1, whose successors are close to
    market neutral by construction and so have a near-zero net.
    """
    if normalize not in ('gross', 'net'):
        raise ValueError(f"normalize must be 'gross' or 'net', got {normalize!r}")
    if not 1 <= k <= eig.vectors.shape[1]:
        raise ValueError(f'k must be between 1 and {eig.vectors.shape[1]}, got {k}')

    weights = pd.DataFrame(eig.vectors[:, :k], index=eig.names, columns=[f'PC{i + 1}' for i in range(k)])
    weights = weights.div(vol.reindex(eig.names), axis=0)
    scale = weights.abs().sum() if normalize == 'gross' else weights.sum()
    return weights.div(scale, axis=1)


def orient_to_reference(eig: Eigen, pcs: pd.DataFrame, reference: pd.Series) -> tuple[Eigen, pd.DataFrame]:
    """Flip components that move against a reference series, returning re-signed copies of both inputs.

    Kept apart from `eigen_decompose` on purpose: pointing PC1 the same way as the market is a reading
    of the result, not a property of the decomposition, and folding it in would make the numerical sign
    rule depend on which benchmark happened to be passed.
    """
    aligned = pcs.join(reference.rename('_reference'), how='inner').dropna()
    if aligned.empty:
        raise ValueError('reference shares no dates with the PC series')

    flips = np.ones(eig.vectors.shape[1])
    for position, column in enumerate(pcs.columns):
        if aligned[column].corr(aligned['_reference']) < 0:
            flips[position] = -1.0
    return (
        Eigen(values=eig.values, vectors=eig.vectors * flips, names=eig.names),
        pcs.mul(flips[: pcs.shape[1]], axis=1),
    )


# --- diagnostics ---


def loading_homogeneity(loadings: pd.Series) -> float:
    """Share of loadings carrying the majority sign; one means every asset moves the same way."""
    signs = np.sign(loadings.to_numpy(dtype=float))
    positive = float((signs > 0).sum())
    return max(positive, len(signs) - positive) / len(signs)


def loading_similarity(a: Eigen, b: Eigen, *, k: int) -> pd.Series:
    """Absolute cosine between matching components of two fits, over the assets they share.

    Absolute because the sign rule can legitimately land on opposite entries when a near-tie in
    magnitudes moves between the two windows.
    """
    shared = a.names.intersection(b.names)
    if len(shared) < 2:
        raise ValueError(f'the two decompositions share {len(shared)} assets; at least 2 are needed')

    left = loadings_frame(a, k=k).loc[shared]
    right = loadings_frame(b, k=k).loc[shared]
    cosines = []
    for column in left.columns:
        u = left[column].to_numpy() / np.linalg.norm(left[column].to_numpy())
        v = right[column].to_numpy() / np.linalg.norm(right[column].to_numpy())
        cosines.append(abs(float(u @ v)))
    return pd.Series(cosines, index=left.columns, name='similarity')


def subspace_similarity(a: Eigen, b: Eigen, *, k: int) -> float:
    """Mean cosine of the principal angles between the top-k subspaces of two fits.

    More trustworthy than matching components one by one: when two eigenvalues are nearly equal their
    individual eigenvectors rotate freely inside the plane they span, so per-component similarity dips
    even though the subspace itself is unchanged.
    """
    shared = a.names.intersection(b.names)
    if len(shared) < 2:
        raise ValueError(f'the two decompositions share {len(shared)} assets; at least 2 are needed')

    left = np.linalg.qr(loadings_frame(a, k=k).loc[shared].to_numpy())[0]
    right = np.linalg.qr(loadings_frame(b, k=k).loc[shared].to_numpy())[0]
    return float(np.linalg.svd(left.T @ right, compute_uv=False).mean())


def sector_eta_squared(loadings: pd.Series, sectors: pd.Series) -> float:
    """Share of the spread in a component's loadings that sector membership accounts for.

    The one-way ANOVA effect size: between-group sum of squares over total sum of squares. Zero means
    sectors say nothing about the loadings, one means the loading is a pure function of sector. This is
    the number that decides whether a component deserves to be called sector-specific.
    """
    aligned = pd.DataFrame({'loading': loadings, 'sector': sectors.reindex(loadings.index)}).dropna()
    if aligned['sector'].nunique() < 2:
        raise ValueError('need at least 2 sectors to attribute loading variance')

    grand_mean = aligned['loading'].mean()
    total_ss = float(((aligned['loading'] - grand_mean) ** 2).sum())
    if total_ss == 0:
        return 0.0
    group = aligned.groupby('sector', observed=True)['loading']
    between_ss = float((group.count() * (group.mean() - grand_mean) ** 2).sum())
    return between_ss / total_ss


def sector_profile(loadings: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    """Mean loading per sector with its standard error and t statistic, ranked by absolute mean."""
    aligned = pd.DataFrame({'loading': loadings, 'sector': sectors.reindex(loadings.index)}).dropna()
    group = aligned.groupby('sector', observed=True)['loading']
    profile = pd.DataFrame({'n': group.count(), 'mean': group.mean(), 'sd': group.std(ddof=1)})
    profile['t'] = profile['mean'] / (profile['sd'] / np.sqrt(profile['n']))
    return profile.sort_values('mean', key=lambda s: s.abs(), ascending=False)


def sector_eta_squared_null(loadings: pd.Series, sectors: pd.Series, *, n_perm: int = 999, seed: int = 0,
                            quantile: float = 0.95) -> float:
    """Eta squared reachable by chance, by reshuffling the sector labels.

    Eta squared rises mechanically as the number of groups grows relative to the number of assets, so a
    raw value means little on its own. This gives it a bar to clear.
    """
    rng = np.random.default_rng(seed)
    aligned = pd.DataFrame({'loading': loadings, 'sector': sectors.reindex(loadings.index)}).dropna()
    labels = aligned['sector'].to_numpy()
    draws = []
    for _ in range(n_perm):
        shuffled = pd.Series(rng.permutation(labels), index=aligned.index)
        draws.append(sector_eta_squared(aligned['loading'], shuffled))
    return float(np.quantile(draws, quantile))


def _check_q(q: float) -> None:
    """Reject a shape ratio the Marcenko-Pastur law cannot describe."""
    if q <= 1:
        raise ValueError(
            f'q = T/N must exceed 1 for the Marcenko-Pastur fit, got {q:.4f}; '
            'shorten the ticker list or lengthen the window so observations outnumber assets'
        )
