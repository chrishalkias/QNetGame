"""
--------------------------------------------------------------------------------
Paired evaluation of the three SOTA training seeds (omni_v3_20k_s{1,2,3}).

Each episode draws random parameters (N, n_ch, p_gen, p_swap, cutoff, with
sigma=0.15 per-repeater inhomogeneity) from the training domain and evaluates
ALL THREE agents on the SAME seeded environment, so the comparison is paired
(the environment stochasticity is shared, not a confound). Delivery time T is
censored at the horizon; delivered fidelity F is averaged over delivered
episodes only.

Produces two side-by-side paper barplots (appendix):
  left  = mean delivery time T per agent   (error bar = standard error of mean)
  right = mean delivered fidelity F per agent (error bar = standard deviation)

Dual-mode + chunkable (each chunk is an independent seed, merge concatenates):
  eval:  PYTHONPATH=src:. python experiments/comparisons/agents_seed_barplot.py --seed 0
  plot:  PYTHONPATH=src:. python experiments/comparisons/agents_seed_barplot.py --plot
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import numpy as np
from experiments.comparisons import _common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpts", nargs="+", default=[
        "checkpoints/omni_v3_20k_s1/policy.pth",
        "checkpoints/omni_v3_20k_s2/policy.pth",
        "checkpoints/omni_v3_20k_s3/policy.pth"])
    ap.add_argument("--labels", nargs="+", default=["seed 1", "seed 2", "seed 3"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=3000, help="per run/chunk")
    ap.add_argument("--horizon", type=int, default=2000)
    # random-parameter domain (defaults = the omni_v3_20k training distribution)
    ap.add_argument("--n_lo", type=int, default=4)
    ap.add_argument("--n_hi", type=int, default=12)
    ap.add_argument("--n_ch", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--pg_lo", type=float, default=0.4)
    ap.add_argument("--pg_hi", type=float, default=0.9)
    ap.add_argument("--ps_lo", type=float, default=0.4)
    ap.add_argument("--ps_hi", type=float, default=0.9)
    ap.add_argument("--cutoff_lo", type=int, default=10)
    ap.add_argument("--cutoff_hi", type=int, default=50)
    ap.add_argument("--p_gen_std", type=float, default=0.15)
    ap.add_argument("--p_swap_std", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/comparisons/agents_seed_barplot.json")
    ap.add_argument("--fig", default="results/figures/agents_seed_barplot")
    return ap.parse_args()


def _rollout(fn, N, n_ch, p_gen, p_swap, cutoff, ep_seed, H, pg_std, ps_std):
    """One greedy episode on a freshly seeded env. Returns (T censored at H,
    F or None if not delivered, delivered flag)."""
    from rl_stack.env_wrapper import QRNEnv
    env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                 p_gen_std=pg_std, p_swap_std=ps_std,
                 F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=H,
                 topology="chain", rng=np.random.default_rng(ep_seed))
    obs = env.reset()
    step, done, info = 0, False, {}
    for step in range(H):
        obs, _, done, info = env.step(fn(env, obs))
        if done:
            break
    F = float(info.get("fidelity", 0.0))
    delivered = bool(done) and F > 0
    return (step + 1 if delivered else H), (F if delivered else None), int(delivered)


def run_eval(a):
    from experiments.mc_eval import make_agent_fn
    agents = {lab: make_agent_fn(ck, hidden=a.hidden)
              for lab, ck in zip(a.labels, a.ckpts)}
    master = np.random.default_rng(a.seed)
    print(f"{len(agents)} agents, {a.episodes} random episodes, H={a.horizon}, "
          f"N in [{a.n_lo},{a.n_hi}], n_ch in {a.n_ch}, "
          f"p_gen/p_swap in [{a.pg_lo},{a.pg_hi}]/[{a.ps_lo},{a.ps_hi}], "
          f"cutoff in [{a.cutoff_lo},{a.cutoff_hi}], sigma={a.p_gen_std}")
    rows = []
    for e in range(a.episodes):
        ep_seed = int(master.integers(2**32))
        prng = np.random.default_rng(ep_seed)   # param draws (same for all agents)
        N = int(prng.integers(a.n_lo, a.n_hi + 1))
        n_ch = int(prng.choice(a.n_ch))
        p_gen = float(prng.uniform(a.pg_lo, a.pg_hi))
        p_swap = float(prng.uniform(a.ps_lo, a.ps_hi))
        cutoff = int(prng.integers(a.cutoff_lo, a.cutoff_hi + 1))
        row = dict(episode=e, seed=ep_seed, N=N, n_ch=n_ch,
                   p_gen=p_gen, p_swap=p_swap, cutoff=cutoff)
        for lab, fn in agents.items():
            T, F, dv = _rollout(fn, N, n_ch, p_gen, p_swap, cutoff, ep_seed,
                                a.horizon, a.p_gen_std, a.p_swap_std)
            row[f"T_{lab}"], row[f"F_{lab}"], row[f"deliv_{lab}"] = T, F, dv
        rows.append(row)
        if e % 25 == 0:
            C.save_json(rows, a.out)
            print(f"  ep {e}/{a.episodes} N={N} n_ch={n_ch} "
                  f"pg={p_gen:.2f} ps={p_swap:.2f} co={cutoff}", flush=True)
    C.save_json(rows, a.out)
    C.write_meta(a, extra={"evaluator": "agents_seed_barplot (paired random-param)"})
    print(f"saved -> {a.out}")


def _stats(rows, labels):
    """Per agent: mean/std/se of censored T (all episodes) and of delivered F
    (delivered episodes only), plus delivery rate."""
    out = {}
    for lab in labels:
        T = np.array([r[f"T_{lab}"] for r in rows], float)
        F = np.array([r[f"F_{lab}"] for r in rows if r[f"F_{lab}"] is not None], float)
        dv = np.array([r[f"deliv_{lab}"] for r in rows], float)
        se = lambda x: float(np.std(x) / np.sqrt(len(x))) if len(x) else 0.0
        out[lab] = dict(
            T_mean=float(T.mean()), T_std=float(T.std()), T_se=se(T),
            F_mean=(float(F.mean()) if len(F) else float("nan")),
            F_std=(float(F.std()) if len(F) else 0.0), F_se=se(F),
            deliv_rate=float(dv.mean()), n=len(rows), n_deliv=int(len(F)))
    return out


def run_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = C.load_json(a.out)
    st = _stats(rows, a.labels)
    n = len(rows)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.titlesize": 11, "figure.dpi": 150})
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(a.labels))
    fig, (axT, axF) = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)

    # left: mean delivery time, error bar = SE of the mean
    T_m = [st[l]["T_mean"] for l in a.labels]
    T_e = [st[l]["T_se"] for l in a.labels]
    axT.bar(x, T_m, yerr=T_e, capsize=5, color=colors, edgecolor="black", lw=0.6)
    axT.set_ylabel(r"mean delivery time $T$ (steps, censored at $H$)")
    axT.set_title("Delivery time")
    for xi, (m, e) in enumerate(zip(T_m, T_e)):
        axT.text(xi, m + e, f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    # right: mean delivered fidelity, error bar = standard deviation
    F_m = [st[l]["F_mean"] for l in a.labels]
    F_s = [st[l]["F_std"] for l in a.labels]
    axF.bar(x, F_m, yerr=F_s, capsize=5, color=colors, edgecolor="black", lw=0.6)
    axF.axhline(0.5, color="firebrick", ls=":", lw=1.0)
    axF.text(len(a.labels) - 1, 0.5, " separable $F=1/2$", color="firebrick",
             fontsize=7, va="bottom", ha="right")
    axF.set_ylabel(r"mean delivered fidelity $\bar{F}$")
    axF.set_title("Delivered fidelity")
    for xi, (m, s) in enumerate(zip(F_m, F_s)):
        axF.text(xi, m + s, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    for ax, plab in ((axT, "(A)"), (axF, "(B)")):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n({st[l]['deliv_rate']*100:.0f}% delivered)"
                            for l in a.labels])
        ax.grid(alpha=0.3, axis="y")
        ax.text(-0.10, 1.03, plab, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="bottom", ha="left")   # just outside top-left
    fig.suptitle(rf"SOTA seeds on {n} random training-domain episodes "
                 rf"($N\in[{a.n_lo},{a.n_hi}]$, $n_\mathrm{{ch}}\in\{{{','.join(map(str,a.n_ch))}\}}$, "
                 rf"$H={a.horizon}$; error bars: (A) SE, (B) SD)", fontsize=9.5)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
