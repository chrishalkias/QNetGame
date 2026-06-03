"""Run one curriculum phase by delegating to QRNAgent.train()."""
from __future__ import annotations
import os
from typing import Dict, List

from .phases import PhaseConfig


def run_phase(agent, cfg: PhaseConfig, save_dir: str, plot: bool = True) -> Dict[str, List]:
    """Train `agent` for one phase and save the checkpoint under `save_dir`.

    Returns the training metrics dict from QRNAgent.train()."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = agent.train(
        episodes=cfg.episodes,
        max_steps=cfg.max_steps,
        n_range=list(cfg.n_range),
        n_ch=list(cfg.n_ch),
        p_gen=cfg.p_gen,
        p_swap=cfg.p_swap,
        cutoff=cfg.cutoff,
        F0=cfg.F0,
        channel_loss=cfg.channel_loss,
        dt_seconds=cfg.dt_seconds,
        heterogeneous=cfg.heterogeneous,
        curriculum=True,
        topology=cfg.topology,
        backend=cfg.backend,
        fidelity_mode=cfg.fidelity_mode,
        save_path=save_dir,
        plot=plot,
    )
    return metrics
