"""
--------------------------------------------------------------------------------
Training-curve figures, the validation results table, and the action-timeline grid.

Split out of agent.py (2026-07-26) so the learning code does not carry
matplotlib at import time. `plot_training` renders training_metrics.png (return
/ loss / steps vs cumulative optimizer steps) and, when --compare logged paired
rollouts, training_compare.png.

Regenerate figures from a finished run's metrics.json without retraining:
  PYTHONPATH=src:. python experiments/training/replot.py --dir checkpoints/<id>/
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import os
import numpy as np

from rl_stack.env_wrapper import NOOP, SWAP, PURIFY

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


def _running_avg(vals, window=30):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(np.mean(vals[lo:i+1]))
    return out


def _repeater_colors(N: int):
    cmap = plt.cm.tab10 if N <= 10 else plt.cm.tab20
    return [to_rgba(cmap(i / max(N - 1, 1))) for i in range(N)]


_ACTION_HATCH = {NOOP: "", SWAP: "///", PURIFY: "..."}

                                       
                # ▄▄▄▄▄▄▄   ▄▄▄        ▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄ 
                # ███▀▀███▄ ███      ▄███████▄ ▀▀▀███▀▀▀ 
                # ███▄▄███▀ ███      ███   ███    ███    
                # ███▀▀▀▀   ███      ███▄▄▄███    ███    
                # ███       ████████  ▀█████▀     ███    

def config_caption(cfg):
    """One-line parameter caption for the figures (N, n_ch, p_gen, p_swap,
    tau=cutoff, H). Returns '' if no config was recorded (older runs)."""
    if not cfg:
        return ""
    def f(v):
        if isinstance(v, (list, tuple)):
            u = sorted(set(v))
            return str(u[0]) if len(u) == 1 else "{" + ",".join(map(str, u)) + "}"
        return str(v)
    parts = [f"N={f(cfg.get('N'))}", f"n_ch={f(cfg.get('n_ch'))}",
             f"p_gen={f(cfg.get('p_gen'))}", f"p_swap={f(cfg.get('p_swap'))}",
             f"τ(cutoff)={cfg.get('cutoff')}", f"max_steps={cfg.get('max_steps')}"]
    if 2 in (cfg.get("disable_actions") or []):
        parts.append("swap-only")
    return ", ".join(parts)


def plot_training(metrics, save_path='assets/', window=None):
    """`window`: rolling-mean window for ALL smoothed curves. None -> the
    per-panel adaptive defaults; pass an int (e.g. via replot.py --window)
    to make the figures smoother without retraining."""
    w_metric = int(window) if window else 30
    w_steps = int(window) if window else 50
    caption = config_caption(metrics.get("config"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Training Metrics" + (f"\n{caption}" if caption else ""),
                 fontsize=11, y=0.99)

    # x-axis = cumulative optimizer steps (the honest learning-progress axis
    # now that updates-per-episode vary under the serialized sweep). Falls back
    # to the episode index for old metrics.json that predate opt_steps.
    xs = metrics.get("opt_steps") or list(range(len(metrics["reward"])))
    xlabel = "Optimizer steps" if metrics.get("opt_steps") else "Episode"

    axes[0].fill_between(xs, metrics["reward"], alpha=0.15, color="royalblue")
    axes[0].plot(xs, _running_avg(metrics["reward"], w_metric), color="royalblue", lw=1.2)
    axes[0].set_ylabel("Episode Return")
    axes[0].axhline(0, color="grey", ls=":", lw=0.5)

    nonzero = [v for v in metrics["loss"] if v > 0]
    if nonzero:
        axes[1].plot(xs, metrics["loss"], alpha=0.2, color="red")
        axes[1].plot(xs, _running_avg(metrics["loss"], w_metric), color="red", lw=1.2)
        axes[1].set_ylabel("Loss")
        axes[1].set_yscale("log")

    axes[2].fill_between(xs, metrics["steps"], alpha=0.15, color="seagreen")
    axes[2].plot(xs, _running_avg(metrics["steps"], w_steps), color="seagreen",
                 lw=1.4)
    axes[2].set_ylabel("Avg Steps to Termination")
    axes[2].set_xlabel(xlabel)

    plt.tight_layout()
    fname = os.path.join(save_path, "training_metrics.png") if save_path else "training_metrics.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

    # --compare: dedicated crossover plot, same net/episode, GREEDY agent vs
    # baselines. Return mixes speed/fidelity/action-economy; the steps and
    # success panels are the pure TASK metrics — read crossovers off those.
    if metrics.get("cmp_agent"):
        # Fixed colour/label per known series; only those present are drawn,
        # so the optimal line appears automatically when compare_extra
        # supplies it. Add new named baselines here to colour them.
        _known = (("rand", "grey", "Random"),
                  ("swap", "darkorange", "SwapASAP"),
                  ("optimal", "seagreen", "Optimal (swap-only)"),
                  ("agent", "royalblue", "Agent (greedy)"))
        series = tuple(s for s in _known if metrics.get(f"cmp_{s[0]}"))
        n = len(metrics["cmp_agent"])
        # x-axis = cumulative optimizer steps (aligned per episode), fall back
        # to episode index if opt_steps is absent or length-mismatched.
        _opt = metrics.get("opt_steps")
        cep = _opt if (_opt and len(_opt) == n) else list(range(n))
        _cxlabel = "Optimizer steps" if (_opt and len(_opt) == n) else "Episode"
        # Wide smoothing: per-episode (p_gen,p_swap) randomisation injects huge
        # raw variance, so a small window can't reveal the trend. No raw fog —
        # at thousands of episodes it just buries the means. `window` (e.g. via
        # replot.py --window) overrides the adaptive default.
        win = int(window) if window else max(50, n // 25)

        # Paired GAP-to-optimal panel (the key readout): because every policy
        # runs on the SAME seeded net each episode, policy_steps - opt_steps
        # cancels the param-draw noise. Agent -> 0 means it reached optimal.
        has_opt = bool(metrics.get("cmp_optimal_steps"))
        panels = [("cmp_{}", "Episode Return"),
                  ("cmp_{}_steps", "Steps to Terminate"),
                  ("cmp_{}_succ", "Success")]
        nrows = len(panels) + (1 if has_opt else 0)
        fig2, axes2 = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)

        for i, (tmpl, ylabel) in enumerate(panels):
            for short, color, label in series:
                axes2[i].plot(cep, _running_avg(metrics[tmpl.format(short)], win),
                              color=color, lw=1.8, label=(label if i == 0 else None))
            axes2[i].set_ylabel(ylabel)
        axes2[0].axhline(0, color="grey", ls=":", lw=0.5)

        if has_opt:
            gax = axes2[len(panels)]
            opt = np.asarray(metrics["cmp_optimal_steps"], dtype=float)
            for short, color, label in series:
                if short == "optimal":
                    continue
                gap = (np.asarray(metrics[f"cmp_{short}_steps"], float) - opt).tolist()
                gax.plot(cep, _running_avg(gap, win), color=color, lw=1.8)
            gax.axhline(0, color="seagreen", ls="--", lw=1.4)  # optimal = 0
            gax.set_ylabel("Steps above Optimal\n(paired; 0 = optimal)")

        _title = (f"Per-episode paired comparison "
                  f"(same seeded network, rolling mean w={win})")
        if caption:
            _title += f"\n{caption}"
        axes2[0].set_title(_title, fontsize=10)
        axes2[0].legend(loc="best", fontsize=9)
        axes2[-1].set_xlabel(_cxlabel)
        plt.tight_layout()
        fname2 = (os.path.join(save_path, "training_compare.png")
                  if save_path else "training_compare.png")
        plt.savefig(fname2, dpi=200, bbox_inches="tight")
        plt.close()


def print_results_table(results, N, pg, ps, c):
    pm = "\u00B1"
    print(f"\n{'='*70}")
    print(f"Validation: N={N}, p_gen={pg}, p_swap={ps}, cutoff={c}")
    print(f"{'='*70}")
    print(f"{'Strategy':<14} | {'Avg Steps':>12} | {'Avg Fidelity':>14} | "
          f"{'Succ%':>6}")
    print("-" * 70)
    for name, data in results.items():
        ns   = len(data["steps"])   # only successful episodes
        tot  = data["total"]
        succ = ns / max(tot, 1) * 100
        avg_s = np.mean(data["steps"]) if ns else float("nan")
        std_s = np.std(data["steps"])  if ns else 0.0
        avg_f = np.mean(data["fidelities"]) if ns else 0.0
        std_f = np.std(data["fidelities"])  if ns else 0.0
        print(f"{name:<14} | {avg_s:>5.1f}{pm}{std_s:<5.1f} | "
              f"{avg_f:>6.4f}{pm}{std_f:<6.4f} | {succ:>5.0f}%")


def plot_timeline_grid(timelines, N, pg, ps, c, save_dir="."):
    """Plot action timeline.

    Each cell = one node at one timestep.
    - Solid colour (repeater ID) = NOOP (wait / background entangle).
    - Hatched ``///`` = SWAP.
    - Hatched ``...`` = PURIFY.
    """
    strats   = list(timelines.keys())
    n_strats = len(strats)
    max_steps = max((len(tl) for tl in timelines.values()), default=1)
    rep_colors = _repeater_colors(N)

    fig_w = min(max_steps * 0.3 + 3, 22)
    fig_h = n_strats * 1.4 + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    row_h = 1.0
    bar_h = row_h / N

    for si, sname in enumerate(strats):
        tl = timelines[sname]
        y_base = (n_strats - 1 - si) * (row_h + 0.3)

        for t, actions in enumerate(tl):
            for node in range(min(N, len(actions))):
                a = int(actions[node])
                y = y_base + node * bar_h
                color = rep_colors[node]
                hatch = _ACTION_HATCH.get(a, "")

                rect = mpatches.FancyBboxPatch(
                    (t - 0.45, y), 0.9, bar_h * 0.9,
                    boxstyle="square,pad=0",
                    facecolor=color, edgecolor="none", linewidth=0)
                ax.add_patch(rect)

                # Only overlay hatch for SWAP / PURIFY
                if a in (SWAP, PURIFY):
                    h_rect = mpatches.FancyBboxPatch(
                        (t - 0.45, y), 0.9, bar_h * 0.9,
                        boxstyle="square,pad=0",
                        facecolor="none", edgecolor="black",
                        hatch=hatch, linewidth=0, alpha=0.6)
                    ax.add_patch(h_rect)
        
        # Append black patch after the end of the timeline
        t_end = len(tl)
        black_patch = mpatches.FancyBboxPatch(
            (t_end - 0.45, y_base), 0.9, row_h - (bar_h * 0.1),
            boxstyle="square,pad=0",
            facecolor="black", edgecolor="none", linewidth=0, zorder=3)
        ax.add_patch(black_patch)

    y_positions = [(n_strats - 1 - i) * (row_h + 0.3) + row_h / 2
                   for i in range(n_strats)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(strats)
    
    # Extended xlim to ensure the appended patch is not cut off
    ax.set_xlim(-0.5, max_steps + 1.5)
    ax.set_ylim(-0.3, n_strats * (row_h + 0.3))
    ax.set_xlabel("Time Step")
    ax.set_title(f"Policy Actions — median episode (N={N}, pg={pg}, ps={ps}, c={c})")
    ax.grid(False)

    handles = []
    for i in range(N):
        handles.append(mpatches.Patch(
            facecolor=rep_colors[i], label=f"R{i}",
            edgecolor="grey", linewidth=0.5))
    handles.append(mpatches.Patch(
        facecolor="white", edgecolor="grey", label="Noop"))
    handles.append(mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="///", label="Swap"))
    handles.append(mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="...", label="Purify"))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1, 0.5), title="Legend", fontsize=7)

    plt.savefig(os.path.join(save_dir, "validation_actions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
