from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .adversary import AdversaryAgent, AdversaryFlavor
from .environment import AdversarialQRNEnv
from .train import load_defender


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    defender: str
    use_adversary: bool


SCENARIOS = (
    Scenario("pretrained_clean", "Pretrained, no sabotage", "pretrained", False),
    Scenario("finetuned_clean", "Fine-tuned, no sabotage", "finetuned", False),
    Scenario("pretrained_adversarial", "Pretrained vs adversary", "pretrained", True),
    Scenario("finetuned_adversarial", "Fine-tuned vs adversary", "finetuned", True),
)


def _make_env(n_repeaters: int, n_ch: int, max_steps: int, seed: int):
    return AdversarialQRNEnv(
        AdversaryFlavor.PHOTON_EATER,
        n_repeaters=n_repeaters,
        n_ch=n_ch,
        spacing=50.0,
        p_gen=0.8,
        p_swap=0.7,
        cutoff=30,
        F0=0.95,
        channel_loss=0.02,
        dt_seconds=1e-3,
        max_steps=max_steps,
        rng=np.random.default_rng(seed),
        topology="chain",
        backend="legacy",
    )


def _load_adversary(checkpoint: Path, n_ch: int) -> AdversaryAgent:
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=n_ch,
        hidden=64,
        epsilon=0.0,
        k=1,
        device="cpu",
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(state_dict, strict=True)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def _empty_raw(n_range):
    return {
        scenario.key: {
            str(n): {
                "success": [],
                "return": [],
                "steps": [],
                "trigger_rate": [],
                "selection_rate": [],
            }
            for n in n_range
        }
        for scenario in SCENARIOS
    }


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    stderr = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {"mean": mean, "stderr": stderr}


def evaluate(
    run_dir: str | Path,
    *,
    pretrained_checkpoint: str | Path = "checkpoints/inhomo_001/policy_final.pth",
    episodes: int = 50,
    n_range=(4, 5, 6, 7),
    n_ch: int = 4,
    max_steps: int = 50,
    seed: int = 2026,
) -> dict:
    run_dir = Path(run_dir)
    pretrained_checkpoint = Path(pretrained_checkpoint)
    defender_checkpoint = run_dir / "defender_final.pth"
    adversary_checkpoint = run_dir / "adversary_final.pth"
    for checkpoint in (
        pretrained_checkpoint,
        defender_checkpoint,
        adversary_checkpoint,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if episodes < 1:
        raise ValueError("episodes must be positive")

    torch.manual_seed(seed)
    defenders = {
        "pretrained": load_defender(
            str(pretrained_checkpoint),
            lr=3e-4,
            rng=np.random.default_rng(seed + 1),
        ),
        "finetuned": load_defender(
            str(defender_checkpoint),
            lr=3e-4,
            rng=np.random.default_rng(seed + 2),
        ),
    }
    for defender in defenders.values():
        defender.epsilon = 0.0
        defender.policy_net.eval()
    adversary = _load_adversary(adversary_checkpoint, n_ch)

    n_range = tuple(int(n) for n in n_range)
    raw = _empty_raw(n_range)
    target_bins = 8
    target_counts = {
        scenario.key: np.zeros((target_bins, n_ch), dtype=np.int64)
        for scenario in SCENARIOS
        if scenario.use_adversary
    }
    seed_rng = np.random.default_rng(seed)

    with torch.no_grad():
        for n_repeaters in n_range:
            episode_seeds = seed_rng.integers(
                0,
                np.iinfo(np.int64).max,
                size=episodes,
                dtype=np.int64,
            )
            for episode_seed in episode_seeds:
                for scenario in SCENARIOS:
                    env = _make_env(
                        n_repeaters,
                        n_ch,
                        max_steps,
                        int(episode_seed),
                    )
                    observation = env.reset()
                    total_return = 0.0
                    trigger_count = 0
                    selection_count = 0
                    done = False
                    info = {"fidelity": 0.0}

                    for step in range(1, max_steps + 1):
                        targets = []
                        if scenario.use_adversary:
                            _, targets = adversary.select_actions(
                                env,
                                observation,
                                training=False,
                            )
                            selection_count += len(targets)
                            for target in targets:
                                position = target.node / max(n_repeaters - 1, 1)
                                bin_index = min(int(position * target_bins), target_bins - 1)
                                target_counts[scenario.key][bin_index, target.qubits[0]] += 1

                        defender = defenders[scenario.defender]
                        actions = defender.select_actions(
                            observation,
                            env.get_action_mask(),
                            training=False,
                        )
                        observation, reward, done, info = env.step_adversarial(
                            actions,
                            targets,
                        )
                        total_return += float(reward)
                        trigger_count += int(info["sabotage_triggered"])
                        if done:
                            break

                    entry = raw[scenario.key][str(n_repeaters)]
                    entry["success"].append(
                        bool(done and float(info.get("fidelity", 0.0)) > 0.0)
                    )
                    entry["return"].append(total_return)
                    entry["steps"].append(step)
                    entry["trigger_rate"].append(trigger_count / step)
                    entry["selection_rate"].append(selection_count / step)

    aggregate = {
        scenario.key: {
            str(n): {
                metric: _summary(values)
                for metric, values in raw[scenario.key][str(n)].items()
            }
            for n in n_range
        }
        for scenario in SCENARIOS
    }
    result = {
        "config": {
            "run_dir": str(run_dir),
            "pretrained_checkpoint": str(pretrained_checkpoint),
            "episodes": episodes,
            "n_range": list(n_range),
            "n_ch": n_ch,
            "max_steps": max_steps,
            "seed": seed,
            "defender_checkpoint": str(defender_checkpoint),
            "adversary_checkpoint": str(adversary_checkpoint),
        },
        "scenarios": {scenario.key: scenario.label for scenario in SCENARIOS},
        "aggregate": aggregate,
        "raw": raw,
        "target_counts": {
            key: counts.tolist() for key, counts in target_counts.items()
        },
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _plot_summary(result, run_dir / "evaluation_summary.png")
    _plot_targets(result, run_dir / "adversary_targets.png")
    return result


def _plot_summary(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_range = result["config"]["n_range"]
    metrics = (
        ("success", "Delivery success rate"),
        ("return", "Mean defender return"),
        ("steps", "Mean episode length"),
        ("trigger_rate", "Sabotage trigger rate"),
    )
    colors = ("#4477AA", "#228833", "#CC6677", "#AA3377")
    styles = ("--", "-", "--", "-")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for axis, (metric, title) in zip(axes.flat, metrics):
        for scenario, color, style in zip(SCENARIOS, colors, styles):
            values = [
                result["aggregate"][scenario.key][str(n)][metric]["mean"]
                for n in n_range
            ]
            errors = [
                result["aggregate"][scenario.key][str(n)][metric]["stderr"]
                for n in n_range
            ]
            axis.errorbar(
                n_range,
                values,
                yerr=errors,
                marker="o",
                color=color,
                linestyle=style,
                capsize=3,
                label=scenario.label,
            )
        axis.set_title(title)
        axis.set_xlabel("Repeaters (N)")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=14,
        markerscale=1.35,
        handlelength=2.8,
        columnspacing=2.2,
        labelspacing=0.8,
    )
    figure.suptitle("Stage III checkpoint evaluation (greedy policies)")
    figure.tight_layout(rect=(0, 0.13, 1, 0.96))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_targets(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("pretrained_adversarial", "finetuned_adversarial")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, key in zip(axes, keys):
        counts = np.asarray(result["target_counts"][key], dtype=np.float64)
        frequencies = counts / counts.sum() if counts.sum() else counts
        image = axis.imshow(frequencies, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(result["scenarios"][key])
        axis.set_xlabel("Target qubit")
        axis.set_xticks(range(result["config"]["n_ch"]))
        axis.set_yticks(range(counts.shape[0]))
        axis.set_yticklabels(
            [f"{i / counts.shape[0]:.2f}-{(i + 1) / counts.shape[0]:.2f}" for i in range(counts.shape[0])]
        )
        figure.colorbar(image, ax=axis, label="Fraction of selected targets")
    axes[0].set_ylabel("Normalized node-position bin")
    figure.suptitle("Learned PhotonEater target distribution")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Stage III checkpoints")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="checkpoints/inhomo_001/policy_final.pth",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-range", type=int, nargs="+", default=(4, 5, 6, 7))
    parser.add_argument("--n-ch", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    evaluate(
        args.run_dir,
        pretrained_checkpoint=args.pretrained_checkpoint,
        episodes=args.episodes,
        n_range=tuple(args.n_range),
        n_ch=args.n_ch,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
