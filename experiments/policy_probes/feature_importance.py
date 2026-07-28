"""
--------------------------------------------------------------------------------
Permutation feature importance over greedy rollouts, multi-seed, multi-panel.

For each observation feature, permute its values across all interior-node decisions
collected from greedy rollouts, re-feed the (otherwise unchanged) states, and
measure the fraction of decisions whose greedy action changes. A larger value means
the policy relies more on that feature. Distribution-grounded and model-agnostic.

Note: can_swap / can_purify are permuted as observation features while the action
mask is held at its true value, so this isolates reliance on the feature beyond the
mask that already gates those actions.

One panel per --ranges entry (chain-size range "lo-hi"), one independent
collection per --seeds entry; bars show the mean over seeds, error bars the
standard deviation. Everything lands in ONE json + ONE pdf, re-renderable:

  compute: PYTHONPATH=src:. python experiments/policy_probes/feature_importance.py \
               --ckpt <path> --ranges 4-12 12-20 --seeds 0 1 2 3
  plot:    PYTHONPATH=src:. python experiments/policy_probes/feature_importance.py --plot
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from experiments.policy_probes import _collect as C

LABELS = {
    "occ": "Occupancy", "can_swap": "Can swap", "can_purify": "Can purify",
    "p_gen": r"$p_{\mathrm{gen}}$", "p_swap": r"$p_{\mathrm{swap}}$",
    "normalized_age": "Normalized age", "relative_position": "Relative position",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", action="store_true", help="render from existing json")
    p.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    p.add_argument("--episodes", type=int, default=200, help="per seed")
    p.add_argument("--n_ch", type=int, nargs="+", default=[2, 3, 4],
                   help="n_ch pool for rollouts (match the ckpt's training n_ch)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one independent collection per seed; bars = mean, "
                   "error bars = std across seeds")
    p.add_argument("--ranges", nargs="+", default=["4-12"],
                   help="chain-size ranges 'lo-hi', one panel each "
                   "(training range is 4-12)")
    p.add_argument("--notes", nargs="+", default=None,
                   help="panel titles, one per range (default: '$N=lo$--$hi$')")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--color", default="#4C72B0",
                   help="bar color (CC-delay agent uses purple by convention)")
    p.add_argument("--max_steps", type=int, default=200,
                   help="episode cap for rollout collection")
    p.add_argument("--xmax", type=float, default=None,
                   help="fixed x-axis limit (default: auto from the data)")
    return p.parse_args()


def _flip_fractions(ckpt, episodes, seed, sizes, max_steps, n_chs=(2, 3, 4)):
    """One collection -> {feature: flip fraction}."""
    d = C.collect(ckpt, episodes=episodes, seed=seed, sizes=sizes,
                  n_chs=n_chs, max_steps=max_steps)
    model, states, idx, base = d["model"], d["states"], d["idx"], d["A"]
    rng = np.random.default_rng(seed)
    flip = {}
    for j, name in enumerate(C.FEATURE_NAMES):
        orig = d["X"][:, j].copy()
        perm = rng.permutation(orig)
        for k, (si, node) in enumerate(idx):
            states[si]["x"][node, j] = perm[k]
        new = C.greedy_actions_for_states(model, states, d["device"])
        new_flat = np.array([new[si][node] for (si, node) in idx])
        flip[name] = float(np.mean(new_flat != base))
        for k, (si, node) in enumerate(idx):
            states[si]["x"][node, j] = orig[k]
        print(f"  {name:<11} {flip[name]:.3f}", flush=True)
    return flip


def compute(a, out_json):
    panels = []
    for ri, rng_str in enumerate(a.ranges):
        lo, hi = (int(v) for v in rng_str.split("-"))
        note = (a.notes[ri] if a.notes else rf"$N={lo}$--${hi}$")
        per_seed = {name: [] for name in C.FEATURE_NAMES}
        for seed in a.seeds:
            print(f"[range {lo}-{hi}, seed {seed}]", flush=True)
            flip = _flip_fractions(a.ckpt, a.episodes, seed, range(lo, hi + 1),
                                   a.max_steps, n_chs=tuple(a.n_ch))
            for name, v in flip.items():
                per_seed[name].append(v)
        panels.append(dict(n_lo=lo, n_hi=hi, note=note, flip=per_seed))
    data = dict(ckpt=a.ckpt, episodes=a.episodes, seeds=a.seeds, panels=panels)
    json.dump(data, open(out_json, "w"), indent=1)
    print(f"saved -> {out_json}")
    return data


def render(data, stem, color="#4C72B0", xmax_fixed=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = data["panels"]
    # one feature order for every panel (sorted by panel-0 mean) so rows align
    names = sorted(C.FEATURE_NAMES,
                   key=lambda n: float(np.mean(panels[0]["flip"][n])))

    def draw(usetex):
        plt.rcParams.update({
            "text.usetex": usetex, "font.family": "serif",
            "mathtext.fontset": "cm", "font.size": 12,
            "axes.labelsize": 13, "axes.titlesize": 13, "figure.dpi": 150,
        })
        P = len(panels)
        fig, axes = plt.subplots(1, P, figsize=(5.6 * P, 4.4),
                                 constrained_layout=True, sharex=True)
        axes = np.atleast_1d(axes)
        # one global limit: per-panel set_xlim under sharex would let the last
        # panel clip the others' longest bar
        xmax = xmax_fixed or max(float(np.mean(p["flip"][n]) + np.std(p["flip"][n]))
                                 for p in panels for n in names) * 1.22
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            means = np.array([np.mean(panel["flip"][n]) for n in names])
            stds = np.array([np.std(panel["flip"][n]) for n in names])
            ax.barh(range(len(names)), means, xerr=stds, color=color,
                    capsize=2.5, error_kw={"lw": 1.0})
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([LABELS[n] for n in names] if i == 0
                               else [""] * len(names))
            for y, (m, s) in enumerate(zip(means, stds)):
                ax.text(m + s + 0.003, y, f"{m:.3f}", va="center", ha="left",
                        fontsize=9)
            ax.set_title(panel["note"])
            lab = f"({'ABCDEFGH'[i]})"
            ax.text(-0.05, 1.07, rf"\textbf{{{lab}}}" if usetex else lab,
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=13, fontweight="bold")
            ax.set_xlim(0, xmax)
            ax.grid(alpha=0.3, axis="x")
        fig.supxlabel("Fraction of decisions altered under feature permutation",
                      fontsize=12)
        return fig

    try:
        fig = draw(True)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[usetex unavailable ({e}); falling back to mathtext]")
        plt.close("all")
        fig = draw(False)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)
    out_json = os.path.join(out, "feature_importance.json")
    data = json.load(open(out_json)) if a.plot else compute(a, out_json)
    render(data, os.path.join(out, "feature_importance"), a.color, a.xmax)


if __name__ == "__main__":
    main()
