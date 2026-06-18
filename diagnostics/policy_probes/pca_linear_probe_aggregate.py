"""Aggregate PC1/PC2 linear decoding analysis across chain sizes.

This uses the same rollout regime as ``interpretability_aggregate.py``:
greedy policy rollouts, all interior nodes, and at least 5,000 conv3 embeddings
for each N in {5, 8, 10, 12, 15}. For each size it asks whether PC1 and PC2 of
the standardized embedding pool linearly decode node mean fidelity and
fractional occupancy.

It also performs leave-one-size-out evaluation. PCA and the decoder are fitted
on four sizes and evaluated on the held-out fifth size. This distinguishes
"the relation exists separately at every size" from "one relation transfers
across sizes".
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnostics.policy_probes.pca_linear_probe import (
    LinearFit,
    _folds,
    _least_squares,
    _r2_score,
    compare_linear_models,
    fit_pca_2,
)
from rl_stack.model import load_qnet
from rl_stack.agent import _obs_to_data
from rl_stack.env_wrapper import QRNEnv


SIZES = (5, 8, 10, 12, 15)
TARGET_EMBEDDINGS = 5000
MAX_STEPS = 50
SEED = 12345
DEFAULT_MODEL = "checkpoints/cluster/cluster_004/policy.pth"
DEFAULT_SAVE_DIR = "checkpoints/cluster/cluster_004/diagnostics"
PARAM_POINT = {
    "p_gen": 0.5,
    "p_swap": 0.7,
    "cutoff": 15,
    "dt_seconds": 0.0,
    "topology": "chain",
    "n_ch": 4,
    "F0": 0.95,
    "channel_loss": 0.02,
}


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _standardize_apply(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (values - mean) / scale


def _ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    x_mean = features.mean(axis=0)
    y_mean = float(target.mean())
    x_centered = features - x_mean
    y_centered = target - y_mean
    gram = x_centered.T @ x_centered
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0]),
        x_centered.T @ y_centered,
    )
    intercept = y_mean - float(x_mean @ coefficients)
    return coefficients, intercept


def ridge_cv_r2(
    embeddings: np.ndarray,
    target: np.ndarray,
    cv_splits: int,
    seed: int,
    alpha: float = 1.0,
) -> float:
    """Match Figure 4's standardized full-embedding ridge reference."""
    standardized, _, _ = _standardize_fit(embeddings)
    scores = []
    for train_idx, test_idx in _folds(len(target), cv_splits, seed):
        coefficients, intercept = _ridge_fit(
            standardized[train_idx], target[train_idx], alpha)
        predictions = intercept + standardized[test_idx] @ coefficients
        scores.append(_r2_score(target[test_idx], predictions))
    return float(np.mean(scores))


def analyze_size(
    embeddings: np.ndarray,
    fidelity: np.ndarray,
    occupancy: np.ndarray,
    cv_splits: int = 5,
    seed: int = SEED,
    ridge_alpha: float = 1.0,
) -> dict:
    """Fit standardized per-size PCA and target decoders."""
    standardized, _, _ = _standardize_fit(embeddings)
    pca = fit_pca_2(standardized)
    targets = {
        "mean_fidelity": np.asarray(fidelity, dtype=np.float64),
        "avg_occupancy": np.asarray(occupancy, dtype=np.float64),
    }
    fits = {
        name: compare_linear_models(
            pca.coords, target, cv_splits=cv_splits, seed=seed)
        for name, target in targets.items()
    }
    full_r2 = {
        name: ridge_cv_r2(
            embeddings, target, cv_splits, seed, alpha=ridge_alpha)
        for name, target in targets.items()
    }
    return {
        "pca": pca,
        "coords": pca.coords,
        "targets": fits,
        "target_values": targets,
        "full_embedding_cv_r2": full_r2,
    }


def leave_one_size_out(
    datasets: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ridge_alpha: float = 1.0,
) -> dict[int, dict[str, dict[str, float]]]:
    """Fit representation decoders on all but one size and test held-out N."""
    output = {}
    for held_size, held_data in datasets.items():
        train_parts = [data for size, data in datasets.items() if size != held_size]
        train_embeddings = np.concatenate([data[0] for data in train_parts])
        train_fidelity = np.concatenate([data[1] for data in train_parts])
        train_occupancy = np.concatenate([data[2] for data in train_parts])
        test_embeddings, test_fidelity, test_occupancy = held_data

        z_train, mean, scale = _standardize_fit(train_embeddings)
        z_test = _standardize_apply(test_embeddings, mean, scale)
        pca = fit_pca_2(z_train)
        train_coords = pca.coords
        test_coords = (z_test - pca.mean) @ pca.components.T

        train_targets = {
            "mean_fidelity": train_fidelity,
            "avg_occupancy": train_occupancy,
        }
        test_targets = {
            "mean_fidelity": test_fidelity,
            "avg_occupancy": test_occupancy,
        }
        output[held_size] = {}
        for target_name in train_targets:
            pc_coefficients, pc_intercept, _ = _least_squares(
                train_coords, train_targets[target_name], fit_intercept=True)
            pc_predictions = pc_intercept + test_coords @ pc_coefficients

            full_coefficients, full_intercept = _ridge_fit(
                z_train, train_targets[target_name], ridge_alpha)
            full_predictions = full_intercept + z_test @ full_coefficients
            output[held_size][target_name] = {
                "two_pc_r2": _r2_score(test_targets[target_name], pc_predictions),
                "full_embedding_r2": _r2_score(
                    test_targets[target_name], full_predictions),
                "two_pc_rmse": float(np.sqrt(np.mean(
                    (test_targets[target_name] - pc_predictions) ** 2))),
                "full_embedding_rmse": float(np.sqrt(np.mean(
                    (test_targets[target_name] - full_predictions) ** 2))),
            }
    return output


class Conv3Hook:
    """Capture the post-ReLU conv3 representation consumed by the head."""

    def __init__(self, model):
        self.value = None
        self._handle = model.conv3.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):  # noqa: ARG002
        self.value = torch.relu(output).detach().cpu().numpy()

    def remove(self):
        self._handle.remove()


def _model_observation(obs: dict, node_dim: int) -> dict:
    """Adapt current 10-feature observations to historical 8-feature models."""
    if obs["x"].shape[1] < node_dim:
        raise ValueError(
            f"environment has {obs['x'].shape[1]} features, model needs {node_dim}")
    return {
        "x": obs["x"][:, :node_dim],
        "edge_index": obs["edge_index"],
    }


def collect_embeddings(
    model,
    n_nodes: int,
    target_count: int,
    device: str,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect Figure 4 rollout embeddings from all interior nodes."""
    hook = Conv3Hook(model)
    embeddings = []
    fidelities = []
    occupancies = []
    seed_rng = np.random.default_rng(rng_seed)
    episodes = 0
    node_dim = int(model.conv1.lin_l.in_channels)
    count = 0

    try:
        while count < target_count:
            env = QRNEnv(
                n_repeaters=n_nodes,
                max_steps=MAX_STEPS,
                rng=np.random.default_rng(int(seed_rng.integers(0, 2**32))),
                **PARAM_POINT,
            )
            obs = env.reset()
            interior = [node for node in range(env.N) if not env.is_target(node)]

            for _ in range(MAX_STEPS):
                mask = env.get_action_mask()
                model_obs = _model_observation(obs, node_dim)
                with torch.no_grad():
                    q_values = model(_obs_to_data(model_obs, device))

                for node in interior:
                    embeddings.append(hook.value[node:node + 1])
                    fidelities.append(float(obs["x"][node, 1]))
                    occupancies.append(float(obs["x"][node, 0]))
                    count += 1
                    if count >= target_count:
                        break
                if count >= target_count:
                    break

                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
                masked_q = q_values.clone()
                masked_q[~mask_tensor] = -float("inf")
                actions = masked_q.argmax(dim=1).cpu().numpy().astype(np.int32)
                obs, _, done, _ = env.step(actions)
                if done:
                    break
            episodes += 1
            if episodes > 100000:
                raise RuntimeError("embedding collection exceeded safety episode cap")
    finally:
        hook.remove()

    return (
        np.concatenate(embeddings, axis=0).astype(np.float64),
        np.asarray(fidelities, dtype=np.float64),
        np.asarray(occupancies, dtype=np.float64),
        episodes,
    )


def _fit_payload(fit: LinearFit) -> dict:
    return {
        "intercept": fit.intercept,
        "pc1_coefficient": float(fit.coefficients[0]),
        "pc2_coefficient": float(fit.coefficients[1]),
        "r2_full": fit.r2_full,
        "r2_cv_mean": fit.r2_cv,
        "r2_cv_std": fit.r2_cv_std,
        "rmse": fit.rmse,
    }


def _equation(target_name: str, fit: LinearFit) -> str:
    return (
        f"{target_name} = {fit.intercept:.6f} "
        f"{fit.coefficients[0]:+.6f}*PC1 "
        f"{fit.coefficients[1]:+.6f}*PC2"
    )


def write_reports(
    save_dir: str,
    model_path: str,
    target_count: int,
    datasets: dict,
    analyses: dict,
    transfer: dict,
) -> tuple[str, str]:
    stem = "diag_pca_linear_probe_aggregate"
    json_path = os.path.join(save_dir, f"{stem}.json")
    text_path = os.path.join(save_dir, f"{stem}.txt")
    payload = {
        "checkpoint": model_path,
        "sizes": list(datasets),
        "target_embeddings_per_size": target_count,
        "regime": PARAM_POINT,
        "per_size": {},
        "leave_one_size_out": transfer,
    }
    lines = [
        "Aggregate PC1/PC2 decoding across chain sizes",
        "=" * 78,
        f"Checkpoint: {model_path}",
        f"Sizes: {list(datasets)}",
        f"Embeddings per size: {target_count}",
        f"Regime: {PARAM_POINT}",
        "PCA: fitted separately to each standardized per-size embedding pool",
        "Full reference: standardized 64D ridge, alpha=1.0",
        "Transfer: train scaler/PCA/decoder on four sizes; test held-out size",
    ]

    for size in datasets:
        analysis = analyses[size]
        pca = analysis["pca"]
        payload["per_size"][str(size)] = {
            "n_embeddings": int(len(datasets[size][1])),
            "episodes": int(datasets[size][3]),
            "pca_variance_pc1": float(pca.explained_variance_ratio[0]),
            "pca_variance_pc2": float(pca.explained_variance_ratio[1]),
            "targets": {},
        }
        lines.extend([
            "",
            f"N={size}",
            "-" * 24,
            f"samples={len(datasets[size][1])}, episodes={datasets[size][3]}, "
            f"PC1={pca.explained_variance_ratio[0] * 100:.3f}%, "
            f"PC2={pca.explained_variance_ratio[1] * 100:.3f}%",
        ])
        for target_name, fits in analysis["targets"].items():
            payload["per_size"][str(size)]["targets"][target_name] = {
                "models": {key: _fit_payload(fit) for key, fit in fits.items()},
                "full_embedding_cv_r2": analysis["full_embedding_cv_r2"][target_name],
            }
            two_pc = fits["pc1_pc2_affine"]
            lines.append(_equation(target_name, two_pc))
            lines.append(
                f"  CV R2: PC1={fits['pc1_affine'].r2_cv:.6f}, "
                f"PC2={fits['pc2_affine'].r2_cv:.6f}, "
                f"PC1+PC2={two_pc.r2_cv:.6f}, "
                f"full64={analysis['full_embedding_cv_r2'][target_name]:.6f}"
            )

    lines.extend(["", "Leave-one-size-out transfer", "-" * 38])
    for size in datasets:
        for target_name, metrics in transfer[size].items():
            lines.append(
                f"held N={size:>2} {target_name:<14} "
                f"two-PC R2={metrics['two_pc_r2']:+.6f} "
                f"full64 R2={metrics['full_embedding_r2']:+.6f}"
            )

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return text_path, json_path


def plot_summary(analyses: dict, transfer: dict, output_path: str) -> None:
    sizes = list(analyses)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"pc1": "#7f7f7f", "pc2": "#bcbd22", "two": "#1f77b4", "full": "#d62728"}

    for ax, target_name, title in (
        (axes[0, 0], "mean_fidelity", "Mean fidelity: within-size decoding"),
        (axes[0, 1], "avg_occupancy", "Occupancy: within-size decoding"),
    ):
        ax.plot(sizes, [analyses[n]["targets"][target_name]["pc1_affine"].r2_cv
                        for n in sizes], "o--", color=colors["pc1"], label="PC1")
        ax.plot(sizes, [analyses[n]["targets"][target_name]["pc2_affine"].r2_cv
                        for n in sizes], "s--", color=colors["pc2"], label="PC2")
        ax.plot(sizes, [analyses[n]["targets"][target_name]["pc1_pc2_affine"].r2_cv
                        for n in sizes], "o-", color=colors["two"], label="PC1 + PC2")
        ax.plot(sizes, [analyses[n]["full_embedding_cv_r2"][target_name]
                        for n in sizes], "d-", color=colors["full"], label="Full 64D ridge")
        ax.set_title(title)
        ax.set_xlabel("Chain size N")
        ax.set_ylabel("5-fold CV R2")
        ax.set_xticks(sizes)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[1, 0].plot(
        sizes,
        [analyses[n]["pca"].explained_variance_ratio[0] * 100 for n in sizes],
        "o-", label="PC1")
    axes[1, 0].plot(
        sizes,
        [analyses[n]["pca"].explained_variance_ratio[:2].sum() * 100 for n in sizes],
        "s-", label="PC1 + PC2")
    axes[1, 0].set_title("Variance retained by first two PCs")
    axes[1, 0].set_xlabel("Chain size N")
    axes[1, 0].set_ylabel("Explained variance (%)")
    axes[1, 0].set_xticks(sizes)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    for target_name, label, color, marker in (
        ("mean_fidelity", "Fidelity: PC1 + PC2", "#1f77b4", "o"),
        ("avg_occupancy", "Occupancy: PC1 + PC2", "#2ca02c", "s"),
    ):
        axes[1, 1].plot(
            sizes, [transfer[n][target_name]["two_pc_r2"] for n in sizes],
            marker=marker, color=color, label=label)
        axes[1, 1].plot(
            sizes, [transfer[n][target_name]["full_embedding_r2"] for n in sizes],
            marker=marker, color=color, linestyle="--",
            label=label.replace("PC1 + PC2", "full 64D"))
    axes[1, 1].set_title("Leave-one-size-out transfer")
    axes[1, 1].set_xlabel("Held-out chain size N")
    axes[1, 1].set_ylabel("Held-out R2")
    axes[1, 1].set_xticks(sizes)
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Aggregate linear decoding from PC1 and PC2 across chain sizes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(analyses: dict, output_path: str, seed: int) -> None:
    sizes = list(analyses)
    fig, axes = plt.subplots(2, len(sizes), figsize=(18, 7), sharex="row", sharey="row")
    rng = np.random.default_rng(seed)
    for column, size in enumerate(sizes):
        for row, target_name in enumerate(("mean_fidelity", "avg_occupancy")):
            actual = analyses[size]["target_values"][target_name]
            predicted = analyses[size]["targets"][target_name]["pc1_pc2_affine"].predictions
            count = min(1500, len(actual))
            indices = rng.choice(len(actual), size=count, replace=False)
            ax = axes[row, column]
            ax.scatter(actual[indices], predicted[indices], s=5, alpha=0.25, linewidths=0)
            low = min(float(actual.min()), float(predicted.min()))
            high = max(float(actual.max()), float(predicted.max()))
            ax.plot([low, high], [low, high], "k--", lw=0.8)
            fit = analyses[size]["targets"][target_name]["pc1_pc2_affine"]
            ax.set_title(f"N={size}, CV R2={fit.r2_cv:.3f}")
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel(f"Predicted {target_name}")
            if row == 1:
                ax.set_xlabel("Actual")
    fig.suptitle("Within-size two-PC predictions on rollout-derived embeddings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(sizes) < 2 or any(size < 3 for size in sizes):
        raise argparse.ArgumentTypeError("provide at least two chain sizes >= 3")
    return sizes


def run(args: argparse.Namespace) -> dict[str, str]:
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_qnet(args.model, args.device)

    collected = {}
    analyses = {}
    transfer_datasets = {}
    for index, size in enumerate(args.sizes):
        print(f"Collecting N={size}: {args.target} rollout embeddings")
        embeddings, fidelity, occupancy, episodes = collect_embeddings(
            model, size, args.target, args.device, args.seed + 1000 * index)
        collected[size] = (embeddings, fidelity, occupancy, episodes)
        transfer_datasets[size] = (embeddings, fidelity, occupancy)
        analyses[size] = analyze_size(
            embeddings, fidelity, occupancy, args.cv_splits, args.seed,
            args.ridge_alpha)
        pca = analyses[size]["pca"]
        print(
            f"  episodes={episodes}; PC1={pca.explained_variance_ratio[0] * 100:.2f}%; "
            f"PC1+2={pca.explained_variance_ratio[:2].sum() * 100:.2f}%")
        for target_name in ("mean_fidelity", "avg_occupancy"):
            fits = analyses[size]["targets"][target_name]
            print(
                f"  {target_name}: two-PC CV R2={fits['pc1_pc2_affine'].r2_cv:.4f}; "
                f"full64 CV R2={analyses[size]['full_embedding_cv_r2'][target_name]:.4f}")

    transfer = leave_one_size_out(transfer_datasets, args.ridge_alpha)
    print("Leave-one-size-out:")
    for size in args.sizes:
        print(
            f"  N={size}: fidelity two-PC={transfer[size]['mean_fidelity']['two_pc_r2']:.4f}, "
            f"occupancy two-PC={transfer[size]['avg_occupancy']['two_pc_r2']:.4f}")

    text_path, json_path = write_reports(
        args.save_dir, args.model, args.target, collected, analyses, transfer)
    summary_path = os.path.join(
        args.save_dir, "diag_pca_linear_probe_aggregate_summary.png")
    predictions_path = os.path.join(
        args.save_dir, "diag_pca_linear_probe_aggregate_predictions.png")
    plot_summary(analyses, transfer, summary_path)
    plot_predictions(analyses, predictions_path, args.seed)
    for path in (text_path, json_path, summary_path, predictions_path):
        print(f"Saved {path}")
    return {
        "text": text_path,
        "json": json_path,
        "summary_plot": summary_path,
        "predictions_plot": predictions_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--sizes", type=parse_sizes, default=SIZES)
    parser.add_argument("--target", type=int, default=TARGET_EMBEDDINGS)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
