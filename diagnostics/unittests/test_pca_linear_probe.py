import numpy as np

from diagnostics.policy_probes.pca_linear_probe import (
    _stabilize_component_signs,
    compare_linear_models,
)


def test_component_signs_make_largest_loading_positive():
    coords = np.array([[1.0, 2.0], [-3.0, 4.0]])
    components = np.array([[-0.8, 0.2, 0.1], [0.1, -0.9, 0.3]])

    stable_coords, stable_components = _stabilize_component_signs(
        coords, components)

    np.testing.assert_allclose(stable_components, -components)
    np.testing.assert_allclose(stable_coords, -coords)


def test_two_pc_model_recovers_weighted_sum_better_than_single_pcs():
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(400, 2))
    target = 0.65 + 0.18 * coords[:, 0] - 0.31 * coords[:, 1]

    fits = compare_linear_models(coords, target, cv_splits=5, seed=11)

    assert fits["pc1_pc2_affine"].r2_full > 0.999999
    assert fits["pc1_pc2_affine"].r2_cv > 0.999999
    assert fits["pc1_pc2_affine"].r2_cv > fits["pc1_affine"].r2_cv
    assert fits["pc1_pc2_affine"].r2_cv > fits["pc2_affine"].r2_cv
    np.testing.assert_allclose(
        fits["pc1_pc2_affine"].coefficients, [0.18, -0.31], atol=1e-12)
    assert abs(fits["pc1_pc2_affine"].intercept - 0.65) < 1e-12


def test_centered_weighted_sum_matches_affine_fit():
    rng = np.random.default_rng(13)
    coords = rng.normal(size=(300, 2))
    coords -= coords.mean(axis=0, keepdims=True)  # PCA coordinates are centered.
    target = 0.72 - 0.24 * coords[:, 0] + 0.09 * coords[:, 1]

    fits = compare_linear_models(coords, target, cv_splits=5, seed=5)
    affine = fits["pc1_pc2_affine"]
    centered = fits["pc1_pc2_centered"]
    through_origin = fits["pc1_pc2_origin"]

    np.testing.assert_allclose(centered.predictions, affine.predictions, atol=1e-12)
    np.testing.assert_allclose(centered.coefficients, affine.coefficients, atol=1e-12)
    assert centered.intercept == np.mean(target)
    assert centered.r2_full > 0.999999
    assert through_origin.r2_full < 0.0
