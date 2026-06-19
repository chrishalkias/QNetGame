from __future__ import annotations

import argparse
import json
import math
import operator
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl_stack.agent import QRNAgent

from .adversary import AdversaryAgent, AdversaryFlavor
from .environment import AdversarialQRNEnv


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


@dataclass(frozen=True)
class StageIIIConfig:
    defender_checkpoint: str = "checkpoints/inhomo_001/policy_final.pth"
    flavor: str = "photon_eater"
    episodes: int = 3000
    max_steps: int = 50
    n_range: tuple[int, ...] = (4, 5, 6, 7)
    n_ch: int = 4
    k: int = 1
    p_gen: float = 0.8
    p_swap: float = 0.7
    cutoff: int = 30
    F0: float = 0.95
    channel_loss: float = 0.02
    dt_seconds: float = 1e-3
    hidden: int = 64
    defender_lr: float = 3e-4
    adversary_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 80_000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.9
    output_dir: str = "checkpoints/adversarial_game"
    seed: int = 0
    plot: bool = True

    def validate(self) -> "StageIIIConfig":
        flavor = AdversaryFlavor(self.flavor)
        if flavor is AdversaryFlavor.COSMIC_RAY:
            raise NotImplementedError("CosmicRay adversary is not implemented")
        if _integer(self.n_ch, "n_ch") < 2:
            raise ValueError("n_ch must be an integer >= 2")
        if _integer(self.k, "K") != 1:
            raise ValueError("this Stage III environment currently requires K=1")
        if not self.n_range:
            raise ValueError("n_range must contain chain sizes >= 3")
        try:
            chain_sizes = tuple(_integer(size, "n_range value") for size in self.n_range)
        except TypeError:
            raise ValueError("n_range must be an iterable of integers") from None
        if any(size < 3 for size in chain_sizes):
            raise ValueError("n_range must contain chain sizes >= 3")
        for name in ("episodes", "max_steps", "buffer_size", "batch_size"):
            if _integer(getattr(self, name), name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if _integer(self.hidden, "hidden") < 1:
            raise ValueError("hidden must be a positive integer")
        if _integer(self.cutoff, "cutoff") < 1:
            raise ValueError("cutoff must be a positive integer")
        _integer(self.seed, "seed")
        if not 0.0 <= self.p_gen <= 1.0 or not 0.0 <= self.p_swap <= 1.0:
            raise ValueError("p_gen and p_swap must be in [0, 1]")
        if self.defender_lr <= 0.0 or self.adversary_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not os.path.isfile(self.defender_checkpoint):
            raise FileNotFoundError(self.defender_checkpoint)
        return self


@dataclass
class TrainingState:
    config: StageIIIConfig
    env: AdversarialQRNEnv
    defender: QRNAgent
    adversary: AdversaryAgent
    episode_rng: np.random.Generator
    environment_seed_rng: np.random.Generator


def load_defender(
    checkpoint: str,
    lr: float,
    rng: np.random.Generator,
    *,
    hidden: int = 64,
    gamma: float = 0.99,
    buffer_size: int = 80_000,
    batch_size: int = 64,
    tau: float = 0.005,
    epsilon: float = 1.0,
) -> QRNAgent:
    defender = QRNAgent(
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        epsilon=epsilon,
        rng=rng,
    )
    state_dict = torch.load(
        checkpoint,
        map_location=defender.device,
        weights_only=True,
    )
    defender.policy_net.load_state_dict(state_dict, strict=True)
    defender.target_net.load_state_dict(defender.policy_net.state_dict())
    defender.target_net.eval()
    return defender


def _make_environment(
    config: StageIIIConfig,
    n_repeaters: int,
    rng: np.random.Generator,
) -> AdversarialQRNEnv:
    return AdversarialQRNEnv(
        AdversaryFlavor(config.flavor),
        n_repeaters=n_repeaters,
        n_ch=config.n_ch,
        spacing=50.0,
        p_gen=config.p_gen,
        p_swap=config.p_swap,
        cutoff=config.cutoff,
        F0=config.F0,
        channel_loss=config.channel_loss,
        dt_seconds=config.dt_seconds,
        max_steps=config.max_steps,
        rng=rng,
        topology="chain",
        gamma=config.gamma,
        backend="legacy",
    )


def build_training_state(config: StageIIIConfig) -> TrainingState:
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    seeds = np.random.SeedSequence(config.seed).spawn(5)
    episode_rng = np.random.default_rng(seeds[0])
    environment_seed_rng = np.random.default_rng(seeds[1])
    defender_rng = np.random.default_rng(seeds[2])
    adversary_rng = np.random.default_rng(seeds[3])
    initial_env_rng = np.random.default_rng(seeds[4])
    env = _make_environment(config, int(config.n_range[0]), initial_env_rng)
    defender = load_defender(
        config.defender_checkpoint,
        config.defender_lr,
        defender_rng,
        hidden=config.hidden,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        tau=config.tau,
        epsilon=config.epsilon_start,
    )
    adversary = AdversaryAgent(
        config.flavor,
        n_ch=config.n_ch,
        hidden=config.hidden,
        lr=config.adversary_lr,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        tau=config.tau,
        epsilon=config.epsilon_start,
        k=config.k,
        rng=adversary_rng,
        device=defender.device,
    )
    return TrainingState(
        config=config,
        env=env,
        defender=defender,
        adversary=adversary,
        episode_rng=episode_rng,
        environment_seed_rng=environment_seed_rng,
    )


def play_step(state: TrainingState, observation, training: bool = True):
    adversary_observation = state.adversary.observe(state.env, observation)
    adversary_actions, targets = state.adversary.select_actions(
        state.env,
        observation,
        training=training,
    )
    defender_actions = state.defender.select_actions(
        observation,
        state.env.get_action_mask(),
        training=training,
    )
    next_observation, defender_reward, done, info = state.env.step_adversarial(
        defender_actions,
        targets,
    )
    defender_reward = float(defender_reward)
    adversary_reward = -defender_reward
    state.defender.memory.add(
        observation,
        defender_actions,
        defender_reward,
        next_observation,
        done,
        state.env.get_action_mask(),
    )
    state.adversary.memory.add(
        adversary_observation,
        adversary_actions,
        adversary_reward,
        state.adversary.observe(state.env, next_observation),
        done,
        state.adversary.get_target_mask(state.env),
    )
    defender_loss = state.defender.train_step() if training else None
    adversary_loss = state.adversary.train_step() if training else None
    record = {
        "defender_reward": defender_reward,
        "adversary_reward": adversary_reward,
        "defender_loss": defender_loss,
        "adversary_loss": adversary_loss,
        "targets": [asdict(target) for target in targets],
        "sabotage_triggered": bool(info["sabotage_triggered"]),
    }
    return next_observation, defender_reward, bool(done), info, record


def _epsilon(config: StageIIIConfig, episode: int) -> float:
    decay_episodes = max(1, math.ceil(config.episodes * config.epsilon_decay_fraction))
    progress = min(episode / max(decay_episodes - 1, 1), 1.0)
    return config.epsilon_end + 0.5 * (
        config.epsilon_start - config.epsilon_end
    ) * (1.0 + math.cos(math.pi * progress))


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _save_plot(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(metrics["defender_return"]) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(episodes, metrics["defender_return"], label="defender")
    axes[0].plot(episodes, metrics["adversary_return"], label="adversary")
    axes[0].set_ylabel("return")
    axes[0].legend()
    axes[1].plot(episodes, metrics["epsilon"], label="epsilon")
    axes[1].plot(episodes, metrics["trigger_count"], label="triggers")
    axes[1].set_xlabel("episode")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(config: StageIIIConfig) -> dict:
    config.validate()
    state = build_training_state(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": asdict(config),
        "defender_return": [],
        "adversary_return": [],
        "defender_loss": [],
        "adversary_loss": [],
        "success": [],
        "delivery_steps": [],
        "epsilon": [],
        "selected_targets": [],
        "trigger_count": [],
    }
    for episode in range(config.episodes):
        n_repeaters = int(state.episode_rng.choice(config.n_range))
        env_seed = int(
            state.environment_seed_rng.integers(0, np.iinfo(np.int64).max)
        )
        state.env = _make_environment(
            config,
            n_repeaters,
            np.random.default_rng(env_seed),
        )
        epsilon = _epsilon(config, episode)
        state.defender.epsilon = epsilon
        state.adversary.epsilon = epsilon
        observation = state.env.reset()
        defender_return = 0.0
        defender_losses: list[float] = []
        adversary_losses: list[float] = []
        selected_targets = []
        trigger_count = 0
        done = False
        info = {"fidelity": 0.0}

        for _ in range(config.max_steps):
            observation, reward, done, info, record = play_step(
                state,
                observation,
                training=True,
            )
            defender_return += reward
            if record["defender_loss"] is not None:
                defender_losses.append(float(record["defender_loss"]))
            if record["adversary_loss"] is not None:
                adversary_losses.append(float(record["adversary_loss"]))
            selected_targets.append(record["targets"])
            trigger_count += int(record["sabotage_triggered"])
            if done:
                break

        success = bool(done and float(info.get("fidelity", 0.0)) > 0.0)
        metrics["defender_return"].append(defender_return)
        metrics["adversary_return"].append(-defender_return)
        metrics["defender_loss"].append(_mean_or_none(defender_losses))
        metrics["adversary_loss"].append(_mean_or_none(adversary_losses))
        metrics["success"].append(success)
        metrics["delivery_steps"].append(state.env.steps if success else None)
        metrics["epsilon"].append(epsilon)
        metrics["selected_targets"].append(selected_targets)
        metrics["trigger_count"].append(trigger_count)

    torch.save(
        state.defender.policy_net.state_dict(),
        output_dir / "defender_final.pth",
    )
    torch.save(
        state.adversary.policy_net.state_dict(),
        output_dir / "adversary_final.pth",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    if config.plot:
        _save_plot(metrics, output_dir / "training_metrics.png")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage III adversarial training")
    parser.add_argument(
        "--defender-checkpoint",
        default=StageIIIConfig.defender_checkpoint,
    )
    parser.add_argument(
        "--flavor",
        choices=[flavor.value for flavor in AdversaryFlavor],
        default=StageIIIConfig.flavor,
    )
    parser.add_argument("--episodes", type=int, default=StageIIIConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=StageIIIConfig.max_steps)
    parser.add_argument("--n-range", type=int, nargs="+", default=StageIIIConfig.n_range)
    parser.add_argument("--n-ch", type=int, default=StageIIIConfig.n_ch)
    parser.add_argument("--k", type=int, default=StageIIIConfig.k)
    parser.add_argument("--p-gen", type=float, default=StageIIIConfig.p_gen)
    parser.add_argument("--p-swap", type=float, default=StageIIIConfig.p_swap)
    parser.add_argument("--cutoff", type=int, default=StageIIIConfig.cutoff)
    parser.add_argument("--defender-lr", type=float, default=StageIIIConfig.defender_lr)
    parser.add_argument("--adversary-lr", type=float, default=StageIIIConfig.adversary_lr)
    parser.add_argument("--batch-size", type=int, default=StageIIIConfig.batch_size)
    parser.add_argument("--buffer-size", type=int, default=StageIIIConfig.buffer_size)
    parser.add_argument("--seed", type=int, default=StageIIIConfig.seed)
    parser.add_argument("--output-dir", default=StageIIIConfig.output_dir)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv=None) -> dict:
    args = _parser().parse_args(argv)
    config = StageIIIConfig(
        defender_checkpoint=args.defender_checkpoint,
        flavor=args.flavor,
        episodes=args.episodes,
        max_steps=args.max_steps,
        n_range=tuple(args.n_range),
        n_ch=args.n_ch,
        k=args.k,
        p_gen=args.p_gen,
        p_swap=args.p_swap,
        cutoff=args.cutoff,
        defender_lr=args.defender_lr,
        adversary_lr=args.adversary_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
        output_dir=args.output_dir,
        plot=not args.no_plot,
    )
    return train(config)


if __name__ == "__main__":
    main()
