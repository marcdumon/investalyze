"""Unit tests for the correlation-matrix PCA: algebraic identities, Marcenko-Pastur, shrinkage, diagnostics.

Correctness rests on synthetic models whose answer is known in advance: a hand-checkable 3x3 correlation
matrix, a one-factor model that must return its own betas as PC1, and planted-block matrices whose factor
count and residual variance the Marcenko-Pastur fit has to recover.
"""

import numpy as np
import pandas as pd
import pytest

from notebooks import eigen

# A and B correlate 0.8 with each other and 0.4 with C; eigenvalues and loadings verified by hand
TOY_CORR = pd.DataFrame(
    [[1.0, 0.8, 0.4], [0.8, 1.0, 0.4], [0.4, 0.4, 1.0]],
    index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
)


def _noise_panel(n_obs: int = 2000, n_assets: int = 200, seed: int = 0) -> pd.DataFrame:
    """Independent standard-normal returns: no cross-sectional structure at all."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_obs, n_assets))
    return pd.DataFrame(values, columns=[f'T{i:03d}' for i in range(n_assets)])


def _factor_panel(betas: np.ndarray, n_obs: int = 4000, seed: int = 0) -> pd.DataFrame:
    """One common factor plus idiosyncratic noise, so PC1 has a known target: the betas."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_obs)
    noise = rng.standard_normal((n_obs, len(betas)))
    values = np.outer(factor, betas) + noise
    return pd.DataFrame(values, columns=[f'T{i:03d}' for i in range(len(betas))])


def _block_panel(n_blocks: int = 3, per_block: int = 60, n_obs: int = 2000, loading: float = 0.6,
                 seed: int = 1) -> pd.DataFrame:
    """Disjoint equally-loaded blocks, so the planted factor count and residual variance are both known.

    Loadings are homogeneous on purpose: with heterogeneous residuals the Marcenko-Pastur fit lands on a
    weighted average of them, which legitimately pushes the factor count above the number planted.
    """
    rng = np.random.default_rng(seed)
    residual = np.sqrt(1.0 - loading**2)
    columns = []
    for block in range(n_blocks):
        common = rng.standard_normal(n_obs)
        for member in range(per_block):
            columns.append(loading * common + residual * rng.standard_normal(n_obs))
    values = np.column_stack(columns)
    return pd.DataFrame(values, columns=[f'B{i:03d}' for i in range(values.shape[1])])


# --- algebraic identities ---


def test_eigenvectors_are_orthonormal():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert np.allclose(eig.vectors.T @ eig.vectors, np.eye(3))


def test_eigenvalues_come_back_descending():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert np.all(np.diff(eig.values) <= 0)


def test_trace_of_a_correlation_matrix_equals_its_width():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eig.values.sum() == pytest.approx(3.0)


def test_decomposition_reconstructs_the_input_matrix():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert np.allclose(eig.vectors @ np.diag(eig.values) @ eig.vectors.T, TOY_CORR.to_numpy())


def test_variance_explained_sums_to_one():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eigen.variance_explained(eig.values).sum() == pytest.approx(1.0)


def test_toy_example_eigenvalues_match_hand_arithmetic():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eig.values == pytest.approx([2.09282032, 0.70717968, 0.2], abs=1e-6)


def test_toy_example_first_component_is_an_all_positive_market():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eigen.loadings_frame(eig)['PC1'].to_numpy() == pytest.approx([0.62796, 0.62796, 0.45970], abs=1e-4)


def test_toy_example_second_component_is_the_spread_of_c_against_a_and_b():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eigen.loadings_frame(eig)['PC2'].to_numpy() == pytest.approx([-0.32505, -0.32505, 0.88808], abs=1e-4)


def test_toy_example_third_component_is_the_pure_a_versus_b_pair_spread():
    eig = eigen.eigen_decompose(TOY_CORR)
    assert eigen.loadings_frame(eig)['PC3'].to_numpy() == pytest.approx([0.70711, -0.70711, 0.0], abs=1e-4)


# --- sign convention ---


def test_sign_rule_makes_the_largest_loading_positive_in_every_component():
    eig = eigen.eigen_decompose(TOY_CORR)
    leaders = np.abs(eig.vectors).argmax(axis=0)
    assert np.all(eig.vectors[leaders, np.arange(3)] > 0)


def test_sign_rule_survives_negating_an_input_series():
    panel = _factor_panel(np.array([1.0, 0.8, 1.2, 0.5]))
    flipped = panel.copy()
    flipped['T001'] *= -1
    base = eigen.eigen_decompose(eigen.correlation(panel)).vectors
    other = eigen.eigen_decompose(eigen.correlation(flipped)).vectors
    assert np.allclose(np.abs(base), np.abs(other), atol=1e-8)


def test_sign_rule_is_stable_for_an_exactly_symmetric_pair():
    # (0.7071, -0.7071, 0) is a fp coin-flip under a plain argmax; the tolerance-based rule pins it
    vectors = np.array([[0.7071, 0.0], [-0.7071, 0.0], [0.0, 1.0]])
    assert np.allclose(eigen.fix_signs(vectors), eigen.fix_signs(eigen.fix_signs(vectors) * -1))


# --- projections ---


def test_pc_series_variance_equals_the_eigenvalues():
    panel = _block_panel(per_block=20, n_obs=4000)
    z = (panel - panel.mean()) / panel.std(ddof=1)
    eig = eigen.eigen_decompose(eigen.correlation(z))
    pcs = eigen.pc_series(z, eig, k=5)
    assert pcs.var(ddof=1).to_numpy() == pytest.approx(eig.values[:5], rel=0.02)


def test_pc_series_components_are_mutually_uncorrelated():
    panel = _block_panel(per_block=20, n_obs=4000)
    z = (panel - panel.mean()) / panel.std(ddof=1)
    pcs = eigen.pc_series(z, eigen.eigen_decompose(eigen.correlation(z)), k=5)
    off_diagonal = np.corrcoef(pcs.to_numpy(), rowvar=False) - np.eye(5)
    assert np.abs(off_diagonal).max() < 1e-10


def test_one_factor_model_returns_each_asset_correlation_with_the_factor_as_the_first_component():
    # standardizing strips scale, so the loading tracks corr(asset, factor) = beta / sqrt(beta^2 + 1),
    # a concave function of beta rather than beta itself. Asserting against beta directly only reaches 0.92.
    betas = np.array([0.4, 0.7, 1.0, 1.3, 1.6, 1.9])
    panel = _factor_panel(betas, n_obs=20_000)
    eig = eigen.eigen_decompose(eigen.correlation(panel))
    factor_corr = betas / np.sqrt(betas**2 + 1)
    assert np.corrcoef(eigen.loadings_frame(eig)['PC1'].to_numpy(), factor_corr)[0, 1] > 0.99


# --- Marcenko-Pastur ---


def test_marcenko_pastur_density_integrates_to_one():
    lo = eigen.min_noise_eigenvalue(q=4.0, sigma2=1.0)
    hi = eigen.max_noise_eigenvalue(q=4.0, sigma2=1.0)
    support = np.linspace(lo, hi, 200_000)
    assert np.trapezoid(eigen.marcenko_pastur_pdf(support, q=4.0, sigma2=1.0), support) == pytest.approx(1.0, abs=1e-3)


def test_noise_edges_match_the_closed_form():
    assert eigen.max_noise_eigenvalue(q=4.0, sigma2=0.5) == pytest.approx(0.5 * (1 + 0.5) ** 2)


def test_pure_noise_eigenvalues_stay_inside_the_marcenko_pastur_support():
    panel = _noise_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    q = panel.shape[0] / panel.shape[1]
    assert values.max() <= 1.02 * eigen.max_noise_eigenvalue(q=q, sigma2=1.0)


def test_fit_sigma2_recovers_unit_variance_on_pure_noise():
    panel = _noise_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    assert eigen.fit_sigma2(values, q=panel.shape[0] / panel.shape[1]) == pytest.approx(1.0, abs=0.05)


def test_pure_noise_yields_no_signal_factors():
    panel = _noise_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    q = panel.shape[0] / panel.shape[1]
    assert eigen.n_signal_factors(values, q=q, sigma2=eigen.fit_sigma2(values, q=q)) == 0


def test_fit_sigma2_recovers_the_planted_residual_variance():
    panel = _block_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    # loading 0.6 leaves a residual variance of 1 - 0.36 = 0.64
    assert eigen.fit_sigma2(values, q=panel.shape[0] / panel.shape[1]) == pytest.approx(0.64, abs=0.05)


def test_marcenko_pastur_counts_the_planted_blocks():
    panel = _block_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    q = panel.shape[0] / panel.shape[1]
    assert eigen.n_signal_factors(values, q=q, sigma2=eigen.fit_sigma2(values, q=q)) == 3


def test_iterative_sigma2_recovers_unit_variance_on_pure_noise():
    panel = _noise_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    assert eigen.fit_sigma2_iterative(values, q=panel.shape[0] / panel.shape[1]) == pytest.approx(1.0, abs=0.02)


def test_iterative_sigma2_recovers_the_planted_residual_variance():
    panel = _block_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    assert eigen.fit_sigma2_iterative(values, q=panel.shape[0] / panel.shape[1]) == pytest.approx(0.64, abs=0.02)


def test_the_two_sigma2_estimators_agree_on_a_marcenko_pastur_shaped_bulk():
    # they diverge badly on real equity spectra, which is the signal that the bulk is not MP shaped
    panel = _block_panel()
    values = eigen.eigen_decompose(eigen.correlation(panel)).values
    q = panel.shape[0] / panel.shape[1]
    assert eigen.fit_sigma2_iterative(values, q=q) == pytest.approx(eigen.fit_sigma2(values, q=q), abs=0.02)


def test_parallel_analysis_agrees_with_the_marcenko_pastur_edge_on_noise():
    panel = _noise_panel(n_obs=800, n_assets=100)
    threshold = eigen.parallel_analysis_threshold(panel, n_trials=20, seed=0)
    assert threshold == pytest.approx(eigen.max_noise_eigenvalue(q=8.0, sigma2=1.0), rel=0.15)


# --- shrinkage ---


def test_denoise_preserves_the_unit_diagonal():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel()))
    assert np.allclose(np.diag(eigen.denoise(eig, n_facts=3)), 1.0)


def test_denoise_collapses_the_spread_among_the_residual_eigenvalues():
    # the tail is flattened exactly, then cov_to_corr rescales the diagonal back to one and
    # reintroduces a little spread; the measured shrink is ~20x, so assert the ratio, not equality
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel()))
    after = eigen.eigen_decompose(eigen.denoise(eig, n_facts=3)).values[3:]
    assert after.std() / eig.values[3:].std() < 0.1


def test_denoise_moves_the_sample_matrix_closer_to_the_population_matrix():
    # de Prado's own Monte Carlo: the point of denoising is accuracy, not just a smoother spectrum
    panel = _block_panel(n_blocks=3, per_block=60, n_obs=400)
    sample = eigen.correlation(panel).to_numpy()
    population = _block_population(n_blocks=3, per_block=60, loading=0.6)
    denoised = eigen.denoise(eigen.eigen_decompose(sample), n_facts=3)
    assert np.linalg.norm(denoised - population) < np.linalg.norm(sample - population)


def test_detone_collapses_variance_along_the_old_first_component():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel()))
    market = eig.vectors[:, 0]
    before = market @ eig.vectors @ np.diag(eig.values) @ eig.vectors.T @ market
    after = market @ eigen.detone(eig) @ market
    assert after / before < 0.01


def test_detone_leaves_a_singular_matrix():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel()))
    assert np.linalg.eigvalsh(eigen.detone(eig)).min() == pytest.approx(0.0, abs=1e-8)


def test_detone_preserves_the_unit_diagonal():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel()))
    assert np.allclose(np.diag(eigen.detone(eig)), 1.0)


# --- eigenportfolios ---


def test_eigenportfolio_halves_the_weight_of_a_twice_as_volatile_asset():
    eig = eigen.Eigen(values=np.array([2.0, 1.0]), vectors=np.array([[0.7071, 0.7071], [0.7071, -0.7071]]),
                      names=pd.Index(['A', 'B']))
    weights = eigen.eigenportfolios(eig, pd.Series({'A': 0.01, 'B': 0.02}), k=1)
    assert weights.loc['B', 'PC1'] / weights.loc['A', 'PC1'] == pytest.approx(0.5)


def test_eigenportfolio_gross_exposure_sums_to_one():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel(per_block=10)))
    vol = pd.Series(0.02, index=eig.names)
    assert eigen.eigenportfolios(eig, vol, k=4).abs().sum().to_numpy() == pytest.approx(np.ones(4))


def test_eigenportfolio_return_tracks_its_component_series():
    panel = _block_panel(per_block=20, n_obs=2000)
    standardized = (panel - panel.mean()) / panel.std(ddof=1)
    eig = eigen.eigen_decompose(eigen.correlation(standardized))
    weights = eigen.eigenportfolios(eig, panel.std(ddof=1), k=1)
    realised = panel.to_numpy() @ weights['PC1'].to_numpy()
    assert abs(np.corrcoef(realised, eigen.pc_series(standardized, eig, k=1)['PC1'])[0, 1]) > 0.99


# --- diagnostics ---


def test_subspace_similarity_of_a_fit_with_itself_is_one():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel(per_block=10)))
    assert eigen.subspace_similarity(eig, eig, k=5) == pytest.approx(1.0)


def test_subspace_similarity_of_orthogonal_subspaces_is_zero():
    names = pd.Index(['A', 'B', 'C', 'D'])
    a = eigen.Eigen(values=np.arange(4, 0, -1.0), vectors=np.eye(4), names=names)
    b = eigen.Eigen(values=np.arange(4, 0, -1.0), vectors=np.eye(4)[:, ::-1].copy(), names=names)
    assert eigen.subspace_similarity(a, b, k=2) == pytest.approx(0.0, abs=1e-12)


def test_subspace_similarity_ignores_rotation_inside_the_same_span():
    # the property that makes this the trustworthy stability metric when eigenvalues nearly tie
    names = pd.Index(['A', 'B', 'C'])
    a = eigen.Eigen(values=np.array([3.0, 2.0, 1.0]), vectors=np.eye(3), names=names)
    angle = np.pi / 5
    rotated = np.eye(3)
    rotated[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    b = eigen.Eigen(values=a.values, vectors=rotated, names=names)
    assert eigen.subspace_similarity(a, b, k=2) == pytest.approx(1.0)


def test_loading_similarity_ignores_a_flipped_component():
    eig = eigen.eigen_decompose(eigen.correlation(_block_panel(per_block=10)))
    flipped = eigen.Eigen(values=eig.values, vectors=-eig.vectors, names=eig.names)
    assert eigen.loading_similarity(eig, flipped, k=3).to_numpy() == pytest.approx(np.ones(3))


def test_loading_homogeneity_is_one_when_every_asset_moves_together():
    assert eigen.loading_homogeneity(pd.Series([0.3, 0.5, 0.2, 0.4])) == pytest.approx(1.0)


def test_loading_homogeneity_is_a_half_for_an_even_long_short_split():
    assert eigen.loading_homogeneity(pd.Series([0.5, -0.5, 0.4, -0.4])) == pytest.approx(0.5)


def test_eta_squared_is_one_when_sector_fully_determines_the_loading():
    loadings = pd.Series([0.5, 0.5, -0.5, -0.5], index=['A', 'B', 'C', 'D'])
    sectors = pd.Series(['tech', 'tech', 'energy', 'energy'], index=['A', 'B', 'C', 'D'])
    assert eigen.sector_eta_squared(loadings, sectors) == pytest.approx(1.0)


def test_eta_squared_is_near_zero_when_labels_carry_no_information():
    rng = np.random.default_rng(0)
    index = pd.Index([f'T{i}' for i in range(400)])
    loadings = pd.Series(rng.standard_normal(400), index=index)
    sectors = pd.Series(rng.choice(['a', 'b', 'c'], 400), index=index)
    assert eigen.sector_eta_squared(loadings, sectors) < 0.05


def test_eta_squared_null_gives_random_labels_a_bar_to_clear():
    rng = np.random.default_rng(0)
    index = pd.Index([f'T{i}' for i in range(200)])
    loadings = pd.Series(rng.standard_normal(200), index=index)
    sectors = pd.Series(rng.choice(['a', 'b', 'c'], 200), index=index)
    observed = eigen.sector_eta_squared(loadings, sectors)
    assert observed < eigen.sector_eta_squared_null(loadings, sectors, n_perm=200)


def test_orient_to_reference_flips_a_component_that_runs_against_the_benchmark():
    dates = pd.date_range('2026-01-01', periods=50, freq='B')
    reference = pd.Series(np.linspace(-1, 1, 50), index=dates)
    pcs = pd.DataFrame({'PC1': -reference.to_numpy()}, index=dates)
    eig = eigen.Eigen(values=np.array([1.0]), vectors=np.array([[1.0]]), names=pd.Index(['A']))
    _, oriented = eigen.orient_to_reference(eig, pcs, reference)
    assert oriented['PC1'].corr(reference) > 0


# --- guards ---


@pytest.mark.parametrize('q', [0.5, 1.0])
def test_marcenko_pastur_rejects_more_assets_than_observations(q):
    with pytest.raises(ValueError, match='q = T/N'):
        eigen.max_noise_eigenvalue(q=q, sigma2=1.0)


def test_decompose_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match='square'):
        eigen.eigen_decompose(np.ones((3, 4)))


def test_decompose_rejects_an_asymmetric_matrix():
    with pytest.raises(ValueError, match='symmetric'):
        eigen.eigen_decompose(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_correlation_rejects_a_panel_with_gaps():
    panel = _noise_panel(n_obs=50, n_assets=4)
    panel.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        eigen.correlation(panel)


def test_denoise_rejects_keeping_every_component():
    eig = eigen.eigen_decompose(TOY_CORR)
    with pytest.raises(ValueError, match='n_facts'):
        eigen.denoise(eig, n_facts=3)


def test_detone_rejects_removing_every_component():
    eig = eigen.eigen_decompose(TOY_CORR)
    with pytest.raises(ValueError, match='n_market'):
        eigen.detone(eig, n_market=3)


def test_pc_series_rejects_more_components_than_assets():
    eig = eigen.eigen_decompose(TOY_CORR)
    z = pd.DataFrame(np.zeros((5, 3)), columns=['A', 'B', 'C'])
    with pytest.raises(ValueError, match='k must be'):
        eigen.pc_series(z, eig, k=4)


def test_eigenportfolios_rejects_an_unknown_normalization():
    eig = eigen.eigen_decompose(TOY_CORR)
    with pytest.raises(ValueError, match='normalize'):
        eigen.eigenportfolios(eig, pd.Series(0.1, index=eig.names), k=1, normalize='beta')


def _block_population(*, n_blocks: int, per_block: int, loading: float) -> np.ndarray:
    """The correlation matrix `_block_panel` samples from: `loading**2` inside a block, zero across blocks."""
    size = n_blocks * per_block
    population = np.eye(size)
    for block in range(n_blocks):
        lo, hi = block * per_block, (block + 1) * per_block
        population[lo:hi, lo:hi] = loading**2
    np.fill_diagonal(population, 1.0)
    return population
