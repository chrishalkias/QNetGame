"""Test whether PC1 and PC2 linearly encode fidelity and occupancy.

The script reproduces the synthetic observation sweep from ``PCA_viz.py``,
projects conv3 embeddings onto the first two principal components, and fits:

    target = intercept + a * PC1 + b * PC2

It also reports PC1-only, PC2-only, centered weighted-sum, and literal
through-origin fits. PCA coordinates are centered, while fidelity and occupancy
have nonzero means, so the centered equation is the meaningful interpretation
of ``target = a * PC1 + b * PC2``:

    target = mean(target) + a * PC1 + b * PC2

Usage:
    PYTHONPATH=. .venv311/bin/python \
        diagnostics/policy_probes/pca_linear_probe.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from rl_stack.agent import _obs_to_data
from rl_stack.env_wrapper import N_ACTIONS
from rl_stack.model import QNetwork, load_qnet


DEFAULT_MODEL = "checkpoints/cluster/cluster_004/policy.pth"
DEFAULT_SAVE_DIR = "checkpoints/cluster/cluster_004/diagnostics"
DEFAULT_RESOLUTION = 25
DEFAULT_N_NODES = 10
DEFAULT_PROBE = 5
DEFAULT_T_REM = 0.5


def _make_obs(
    fidelity: float,
    occupancy: float,
    time_remaining: float,
    n_nodes: int,
    probe: int,
    node_dim: int,
) -> dict[str, np.ndarray]:
    """Build the same synthetic chain sweep used by the original PCA plot."""
    if node_dim not in (8, 10):
        raise ValueError(f"unsupported checkpoint node dimension: {node_dim}")
    features = np.zeros((n_nodes, node_dim), dtype=np.float32)
    for node in range(n_nodes):
        if node == 0:
            values = [0.25, 0.70, 1, 0, 0.25, 0, 0, DEFAULT_T_REM]
        elif node == n_nodes - 1:
            values = [0.25, 0.70, 0, 1, 0.25, 0, 0, DEFAULT_T_REM]
        elif node == probe:
            can_swap = 1.0 if occupancy >= 0.5 else 0.0
            values = [
                occupancy, fidelity, 0, 0, occupancy, can_swap, 0,
                time_remaining,
            ]
        else:
            values = [0.50, 0.70, 0, 0, 0.50, 1, 0, DEFAULT_T_REM]
        features[node, :8] = values
        if node_dim == 10:
            features[node, 8:] = [0.5, 0.7]

    src, dst = [], []
    for node in range(n_nodes - 1):
        src.extend([node, node + 1])
        dst.extend([node + 1, node])
    return {
        "x": features,
        "edge_index": np.array([src, dst], dtype=np.int64),
    }


def collect_embeddings(
    model: QNetwork,
    device: str,
    resolution: int,
    n_nodes: int,
    probe: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep fidelity, occupancy, and time; capture probe-node conv3 output."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if not 0 < probe < n_nodes - 1:
        raise ValueError("probe must be an interior node")

    fidelity_values = np.linspace(0.25, 1.0, resolution)
    occupancy_values = np.linspace(0.0, 1.0, resolution)
    time_values = np.linspace(0.05, 1.0, resolution)
    node_dim = int(model.conv1.lin_l.in_channels)

    embeddings = []
    features_out = []
    actions = []
    hook_output: list[torch.Tensor] = []

    def capture_conv3(module, inputs, output):  # noqa: ARG001
        hook_output.append(output.detach().cpu())

    handle = model.conv3.register_forward_hook(capture_conv3)
    try:
        with torch.no_grad():
            for fidelity in fidelity_values:
                for occupancy in occupancy_values:
                    for time_remaining in time_values:
                        obs = _make_obs(
                            float(fidelity), float(occupancy),
                            float(time_remaining), n_nodes, probe, node_dim)
                        hook_output.clear()
                        q_values = model(_obs_to_data(obs, device))
                        embeddings.append(hook_output[0][probe].numpy())
                        features_out.append([fidelity, occupancy, time_remaining])
                        actions.append(int(q_values[probe].argmax()))
    finally:
        handle.remove()

    return (
        np.asarray(embeddings, dtype=np.float64),
        np.asarray(features_out, dtype=np.float64),
        np.asarray(actions, dtype=np.int32),
    )


@dataclass
class LinearFit:
    """Full-data fit plus shuffled K-fold validation statistics."""

    name: str
    coefficients: np.ndarray
    intercept: float
    predictions: np.ndarray
    r2_full: float
    r2_cv: float
    r2_cv_std: float
    rmse: float
    correlation: float


@dataclass
class PCAProjection:
    """Two-component PCA projection computed with NumPy SVD."""

    coords: np.ndarray
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray


def _stabilize_component_signs(
    coords: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the scikit-learn PCA convention: largest loading is positive."""
    max_loading = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), max_loading])
    signs[signs == 0.0] = 1.0
    return coords * signs, components * signs[:, None]


def fit_pca_2(embeddings: np.ndarray) -> PCAProjection:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be a 2D array with at least two samples")
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    coords, components = _stabilize_component_signs(coords, components)
    variances = singular_values ** 2
    explained = variances[:2] / variances.sum()
    return PCAProjection(coords, components, mean, explained)


def _r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - residual / total if total > 0.0 else float("nan")


def _least_squares(
    x: np.ndarray,
    target: np.ndarray,
    fit_intercept: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    if fit_intercept:
        design = np.column_stack([np.ones(x.shape[0]), x])
        params, *_ = np.linalg.lstsq(design, target, rcond=None)
        intercept = float(params[0])
        coefficients = params[1:]
    else:
        coefficients, *_ = np.linalg.lstsq(x, target, rcond=None)
        intercept = 0.0
    predictions = intercept + x @ coefficients
    return np.asarray(coefficients), intercept, predictions


def _folds(n_samples: int, cv_splits: int, seed: int):
    shuffled = np.random.default_rng(seed).permutation(n_samples)
    for test_idx in np.array_split(shuffled, cv_splits):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        yield np.flatnonzero(train_mask), test_idx


def _validate_inputs(coords: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n_samples, 2), got {coords.shape}")
    if target.ndim != 1 or target.shape[0] != coords.shape[0]:
        raise ValueError("target must be a vector with one value per PCA coordinate")
    if coords.shape[0] < 3:
        raise ValueError("at least three samples are required")
    if not np.isfinite(coords).all() or not np.isfinite(target).all():
        raise ValueError("coords and target must contain only finite values")
    return coords, target


def _fit_model(
    name: str,
    coords: np.ndarray,
    target: np.ndarray,
    columns: tuple[int, ...],
    fit_intercept: bool,
    cv_splits: int,
    seed: int,
) -> LinearFit:
    x = coords[:, columns]
    model_coef, model_intercept, predictions = _least_squares(
        x, target, fit_intercept)

    cv_scores = []
    for train_idx, test_idx in _folds(x.shape[0], cv_splits, seed):
        fold_coef, fold_intercept, _ = _least_squares(
            x[train_idx], target[train_idx], fit_intercept)
        fold_pred = fold_intercept + x[test_idx] @ fold_coef
        cv_scores.append(_r2_score(target[test_idx], fold_pred))

    coefficients = np.zeros(2, dtype=np.float64)
    coefficients[list(columns)] = model_coef
    correlation = float(np.corrcoef(target, predictions)[0, 1])
    return LinearFit(
        name=name,
        coefficients=coefficients,
        intercept=model_intercept,
        predictions=predictions,
        r2_full=_r2_score(target, predictions),
        r2_cv=float(np.mean(cv_scores)),
        r2_cv_std=float(np.std(cv_scores)),
        rmse=float(np.sqrt(np.mean((target - predictions) ** 2))),
        correlation=correlation,
    )


def _fit_centered_model(
    coords: np.ndarray,
    target: np.ndarray,
    cv_splits: int,
    seed: int,
) -> LinearFit:
    """Fit y - mean(y) from centered PC1/PC2 without an explicit intercept."""
    x_mean = coords.mean(axis=0)
    y_mean = float(target.mean())
    x_centered = coords - x_mean
    coefficients, _, centered_predictions = _least_squares(
        x_centered, target - y_mean, fit_intercept=False)
    predictions = y_mean + centered_predictions

    cv_scores = []
    for train_idx, test_idx in _folds(coords.shape[0], cv_splits, seed):
        fold_x_mean = coords[train_idx].mean(axis=0)
        fold_y_mean = float(target[train_idx].mean())
        fold_coef, _, _ = _least_squares(
            coords[train_idx] - fold_x_mean,
            target[train_idx] - fold_y_mean,
            fit_intercept=False,
        )
        fold_pred = fold_y_mean + (coords[test_idx] - fold_x_mean) @ fold_coef
        cv_scores.append(_r2_score(target[test_idx], fold_pred))

    intercept = y_mean - float(x_mean @ coefficients)
    return LinearFit(
        name="PC1 + PC2, centered target",
        coefficients=coefficients,
        intercept=intercept,
        predictions=predictions,
        r2_full=_r2_score(target, predictions),
        r2_cv=float(np.mean(cv_scores)),
        r2_cv_std=float(np.std(cv_scores)),
        rmse=float(np.sqrt(np.mean((target - predictions) ** 2))),
        correlation=float(np.corrcoef(target, predictions)[0, 1]),
    )


def compare_linear_models(
    coords: np.ndarray,
    target: np.ndarray,
    cv_splits: int = 5,
    seed: int = 42,
) -> dict[str, LinearFit]:
    """Compare single-PC and weighted two-PC linear encodings of a target."""
    coords, target = _validate_inputs(coords, target)
    if not 2 <= cv_splits <= coords.shape[0]:
        raise ValueError("cv_splits must be between 2 and the number of samples")
    return {
        "pc1_affine": _fit_model(
            "PC1 only", coords, target, (0,), True, cv_splits, seed),
        "pc2_affine": _fit_model(
            "PC2 only", coords, target, (1,), True, cv_splits, seed),
        "pc1_pc2_affine": _fit_model(
            "PC1 + PC2", coords, target, (0, 1), True, cv_splits, seed),
        "pc1_pc2_centered": _fit_centered_model(
            coords, target, cv_splits, seed),
        "pc1_pc2_origin": _fit_model(
            "PC1 + PC2, through origin", coords, target, (0, 1), False,
            cv_splits, seed),
    }


def _equation(target_name: str, fit: LinearFit) -> str:
    a, b = fit.coefficients
    return (
        f"{target_name} = {fit.intercept:.6f} "
        f"{a:+.6f}*PC1 {b:+.6f}*PC2"
    )


def _fit_to_dict(fit: LinearFit) -> dict:
    return {
        "name": fit.name,
        "intercept": fit.intercept,
        "a_pc1": float(fit.coefficients[0]),
        "b_pc2": float(fit.coefficients[1]),
        "r2_full": fit.r2_full,
        "r2_cv_mean": fit.r2_cv,
        "r2_cv_std": fit.r2_cv_std,
        "rmse": fit.rmse,
        "correlation_actual_predicted": fit.correlation,
    }


def write_reports(
    save_dir: str,
    stem: str,
    model_path: str,
    pca: PCAProjection,
    coords: np.ndarray,
    targets: dict[str, np.ndarray],
    results: dict[str, dict[str, LinearFit]],
) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{stem}.json")
    text_path = os.path.join(save_dir, f"{stem}.txt")

    payload = {
        "checkpoint": model_path,
        "n_samples": int(next(iter(targets.values())).shape[0]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        "targets": {},
    }
    lines = [
        "PC1/PC2 linear encoding probe",
        "=" * 72,
        f"Checkpoint: {model_path}",
        f"Samples: {payload['n_samples']}",
        "PCA variance: "
        f"PC1={pca.explained_variance_ratio[0] * 100:.3f}%, "
        f"PC2={pca.explained_variance_ratio[1] * 100:.3f}%",
        "",
        "Interpret the centered model as target = mean(target) + a*PC1 + b*PC2.",
        "The origin model is the literal target = a*PC1 + b*PC2 comparison.",
    ]

    for target_name, target in targets.items():
        fits = results[target_name]
        payload["targets"][target_name] = {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "corr_pc1": float(np.corrcoef(target, coords[:, 0])[0, 1]),
            "corr_pc2": float(np.corrcoef(target, coords[:, 1])[0, 1]),
            "models": {key: _fit_to_dict(fit) for key, fit in fits.items()},
        }
        lines.extend(["", target_name, "-" * len(target_name)])
        for key, fit in fits.items():
            lines.append(_equation(target_name, fit))
            lines.append(
                f"  {key}: R2(full)={fit.r2_full:.6f}, "
                f"R2(CV)={fit.r2_cv:.6f} +/- {fit.r2_cv_std:.6f}, "
                f"RMSE={fit.rmse:.6f}, corr(y,pred)={fit.correlation:.6f}"
            )

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return json_path, text_path


def plot_target_fits(
    coords: np.ndarray,
    targets: dict[str, np.ndarray],
    results: dict[str, dict[str, LinearFit]],
    pca_var: np.ndarray,
    output_path: str,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    indices = (
        rng.choice(n, size=max_points, replace=False)
        if n > max_points else np.arange(n)
    )
    plot_coords = coords[indices]

    fig, axes = plt.subplots(len(targets), 3, figsize=(16, 9), squeeze=False)
    for row, (target_name, target) in enumerate(targets.items()):
        fit = results[target_name]["pc1_pc2_affine"]
        actual = target[indices]
        predicted = fit.predictions[indices]
        residual = actual - predicted
        vmin, vmax = float(target.min()), float(target.max())

        actual_scatter = axes[row, 0].scatter(
            plot_coords[:, 0], plot_coords[:, 1], c=actual, cmap="viridis",
            vmin=vmin, vmax=vmax, s=5, alpha=0.55, linewidths=0)
        axes[row, 0].set_title(f"{target_name}: actual")
        fig.colorbar(actual_scatter, ax=axes[row, 0])

        span = np.ptp(plot_coords, axis=0)
        direction = fit.coefficients / max(np.linalg.norm(fit.coefficients), 1e-12)
        arrow_length = 0.22 * float(np.min(span))
        axes[row, 0].arrow(
            0.0, 0.0, direction[0] * arrow_length, direction[1] * arrow_length,
            width=0.015 * arrow_length, color="black", length_includes_head=True)
        axes[row, 0].text(
            0.02, 0.98, _equation(target_name, fit),
            transform=axes[row, 0].transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

        residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
        residual_scatter = axes[row, 1].scatter(
            plot_coords[:, 0], plot_coords[:, 1], c=residual, cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0,
                              vmax=residual_limit),
            s=5, alpha=0.55, linewidths=0)
        axes[row, 1].set_title(f"{target_name}: residual (actual - fitted)")
        fig.colorbar(residual_scatter, ax=axes[row, 1])

        axes[row, 2].scatter(actual, predicted, s=6, alpha=0.4, linewidths=0)
        lo = min(float(actual.min()), float(predicted.min()))
        hi = max(float(actual.max()), float(predicted.max()))
        axes[row, 2].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[row, 2].set_xlabel("Actual")
        axes[row, 2].set_ylabel("Predicted")
        axes[row, 2].set_title(
            f"Two-PC fit: R2={fit.r2_full:.4f}, CV R2={fit.r2_cv:.4f}")
        axes[row, 2].grid(alpha=0.25)

        for col in (0, 1):
            axes[row, col].set_xlabel(f"PC1 ({pca_var[0] * 100:.1f}%)")
            axes[row, col].set_ylabel(f"PC2 ({pca_var[1] * 100:.1f}%)")

    fig.suptitle("Linear decoding from the first two conv3 principal components")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, dict[str, LinearFit]],
    output_path: str,
) -> None:
    model_keys = ["pc1_affine", "pc2_affine", "pc1_pc2_affine", "pc1_pc2_origin"]
    labels = ["PC1", "PC2", "PC1 + PC2", "PC1 + PC2\nthrough origin"]
    fig, axes = plt.subplots(1, len(results), figsize=(12, 4.5), squeeze=False)

    for ax, (target_name, fits) in zip(axes[0], results.items()):
        full = [fits[key].r2_full for key in model_keys]
        cv = [fits[key].r2_cv for key in model_keys]
        x = np.arange(len(model_keys))
        width = 0.36
        ax.bar(x - width / 2, full, width, label="In-sample R2")
        ax.bar(x + width / 2, cv, width, label="5-fold CV R2")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("R2")
        ax.set_title(target_name)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Does a weighted PC1 + PC2 model improve linear decoding?")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_probe(args: argparse.Namespace) -> dict[str, str]:
    os.makedirs(args.save_dir, exist_ok=True)
    model = load_qnet(args.model, args.device)
    embeddings, features, _ = collect_embeddings(
        model,
        device=args.device,
        resolution=args.resolution,
        n_nodes=args.n_nodes,
        probe=args.probe,
    )

    pca = fit_pca_2(embeddings)
    coords = pca.coords
    targets = {
        "mean_fidelity": features[:, 0].astype(np.float64),
        "avg_occupancy": features[:, 1].astype(np.float64),
    }
    results = {
        name: compare_linear_models(
            coords, target, cv_splits=args.cv_splits, seed=args.seed)
        for name, target in targets.items()
    }

    stem = f"diag_pca_linear_probe_n{args.n_nodes}_p{args.probe}"
    json_path, text_path = write_reports(
        args.save_dir, stem, args.model, pca, coords, targets, results)
    fit_path = os.path.join(args.save_dir, f"{stem}_fits.png")
    comparison_path = os.path.join(args.save_dir, f"{stem}_r2.png")
    plot_target_fits(
        coords, targets, results, pca.explained_variance_ratio, fit_path,
        args.seed, args.max_plot_points)
    plot_model_comparison(results, comparison_path)

    print(
        f"PCA variance: PC1={pca.explained_variance_ratio[0] * 100:.2f}%, "
        f"PC2={pca.explained_variance_ratio[1] * 100:.2f}%")
    for target_name, fits in results.items():
        fit = fits["pc1_pc2_affine"]
        print(_equation(target_name, fit))
        print(
            f"  R2(full)={fit.r2_full:.6f}; "
            f"R2(CV)={fit.r2_cv:.6f} +/- {fit.r2_cv_std:.6f}; "
            f"PC1-only CV R2={fits['pc1_affine'].r2_cv:.6f}; "
            f"PC2-only CV R2={fits['pc2_affine'].r2_cv:.6f}")
    for path in (text_path, json_path, fit_path, comparison_path):
        print(f"Saved {path}")
    return {
        "text": text_path,
        "json": json_path,
        "fits_plot": fit_path,
        "r2_plot": comparison_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_nodes", type=int, default=DEFAULT_N_NODES)
    parser.add_argument("--probe", type=int, default=DEFAULT_PROBE)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_plot_points", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())
