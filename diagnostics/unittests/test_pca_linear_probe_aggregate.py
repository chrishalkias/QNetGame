import numpy as np

from diagnostics.policy_probes.pca_linear_probe_aggregate import (
    analyze_size,
    leave_one_size_out,
)


def _make_dataset(seed: int, n: int = 500):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    mixing = rng.normal(size=(2, 8))
    embeddings = latent @ mixing + 0.01 * rng.normal(size=(n, 8))
    fidelity = 0.6 + 0.16 * latent[:, 0] - 0.09 * latent[:, 1]
    occupancy = 0.5 - 0.11 * latent[:, 0] + 0.21 * latent[:, 1]
    return embeddings, fidelity, occupancy


def test_analyze_size_finds_two_pc_encoding():
    embeddings, fidelity, occupancy = _make_dataset(3)

    result = analyze_size(
        embeddings, fidelity, occupancy, cv_splits=5, seed=9)

    assert result["targets"]["mean_fidelity"]["pc1_pc2_affine"].r2_cv > 0.99
    assert result["targets"]["avg_occupancy"]["pc1_pc2_affine"].r2_cv > 0.99
    assert result["full_embedding_cv_r2"]["mean_fidelity"] > 0.99
    assert result["full_embedding_cv_r2"]["avg_occupancy"] > 0.99


def test_leave_one_size_out_recovers_shared_relation():
    datasets = {
        size: _make_dataset(seed=size, n=400)
        for size in (5, 8, 10)
    }
    # Use a common embedding basis so the same representation relation exists
    # across sizes, while each size still has independently sampled states.
    common_mixing = np.random.default_rng(99).normal(size=(2, 8))
    rebuilt = {}
    for size in datasets:
        rng = np.random.default_rng(size)
        latent = rng.normal(size=(400, 2))
        embeddings = latent @ common_mixing + 0.01 * rng.normal(size=(400, 8))
        fidelity = 0.6 + 0.16 * latent[:, 0] - 0.09 * latent[:, 1]
        occupancy = 0.5 - 0.11 * latent[:, 0] + 0.21 * latent[:, 1]
        rebuilt[size] = (embeddings, fidelity, occupancy)

    result = leave_one_size_out(rebuilt, ridge_alpha=1.0)

    for size in rebuilt:
        assert result[size]["mean_fidelity"]["two_pc_r2"] > 0.99
        assert result[size]["avg_occupancy"]["two_pc_r2"] > 0.99
        assert result[size]["mean_fidelity"]["full_embedding_r2"] > 0.99
        assert result[size]["avg_occupancy"]["full_embedding_r2"] > 0.99
