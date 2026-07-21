"""
--------------------------------------------------------------------------------
Shared helpers for the experiments/comparisons sweeps.

All evals reuse mc_eval.mc_eval (delivery time T, censored at the
horizon) and the same three policies, so every comparison is a paired,
identically-configured measurement. Delivery is topological (source connects
to dest); the cutoff already bounds delivered-link decoherence, so no
fidelity threshold gates the terminal state (dropped 2026-07-21).
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import json, math, os
import numpy as np


def build_policies(ckpt, hidden=64):
    """{name: policy_fn(env, obs)} for agent / swap-ASAP / purify-then-swap."""
    from experiments import mc_eval as ob
    from rl_stack import strategies
    return {
        "agent":       ob.make_agent_fn(ckpt, hidden=hidden),
        "swap_asap":   lambda env, obs: strategies.swap_asap(env),
        "purify_swap": lambda env, obs: strategies.purify_then_swap(env),
    }


def eval_T(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps,
           p_gen_std=0.0, p_swap_std=0.0):
    """(mean T, standard error) for one policy at one config. p_gen_std/
    p_swap_std > 0 -> per-repeater inhomogeneous chain."""
    from experiments import mc_eval as ob
    T, sd = ob.mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps,
                       p_gen_std=p_gen_std, p_swap_std=p_swap_std)
    return float(T), float(sd / math.sqrt(mc_eps))


def eval_stats(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, seed=42,
              p_gen_std=0.0, p_swap_std=0.0):
    """Canonical comparisons-suite evaluator: one rollout pass yielding delivery
    time T plus fidelity stats over delivered (topologically connected) episodes.

    Returns a dict: T, se, conn_rate, mean_F_conn, seF_conn.
    """
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    times, conn_fids = [], []
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     p_gen_std=p_gen_std, p_swap_std=p_swap_std,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        step, done, info = 0, False, {}
        for step in range(H):
            obs, _, done, info = env.step(policy_fn(env, obs))
            if done:
                break
        F = info.get("fidelity", 0.0)
        connected = bool(done) and F > 0
        times.append(step + 1 if connected else H)
        if connected:
            conn_fids.append(float(F))
    _se = lambda x: float(np.std(x) / math.sqrt(len(x))) if len(x) else 0.0
    n = float(mc_eps)
    return dict(
        T=float(np.mean(times)), se=_se(times),
        conn_rate=len(conn_fids) / n,
        mean_F_conn=(float(np.mean(conn_fids)) if conn_fids else None),
        seF_conn=_se(conn_fids),
    )


def eval_T_and_F(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, seed=42):
    """(T mean, T se, F mean, F se). T = delivery time censored at H (matches
    mc_eval.mc_eval). F = mean terminal fidelity over delivered episodes only
    (censored/truncated episodes carry no delivery fidelity)."""
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    times, fids = [], []
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        step, done, info = 0, False, {}
        for step in range(H):
            obs, _, done, info = env.step(policy_fn(env, obs))
            if done:
                break
        F = info.get("fidelity", 0.0)
        delivered = bool(done) and F > 0
        times.append(step + 1 if delivered else H)
        if delivered:
            fids.append(float(F))
    T = np.asarray(times, float)
    _se = lambda x: float(np.std(x) / math.sqrt(len(x))) if len(x) else 0.0
    F = float(np.mean(fids)) if fids else float("nan")
    return float(np.mean(T)), _se(T), F, _se(fids)


def action_fractions(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, seed=42):
    """Fraction of [NOOP, SWAP, PURIFY] among interior-node decisions over
    mc_eps greedy rollouts (source/dest excluded; the three sum to 1)."""
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    counts = np.zeros(3, dtype=np.int64)
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        for _ in range(H):
            acts = policy_fn(env, obs)
            for i in range(env.N):
                if i not in (env.source, env.dest):
                    counts[int(acts[i])] += 1
            obs, _, done, _ = env.step(acts)
            if done:
                break
    return (counts / max(counts.sum(), 1)).tolist()   # [f_noop, f_swap, f_purify]


def write_meta(args, extra=None):
    """Paper-provenance sidecar `<out>.meta.json`: the resolved CLI args plus
    git commit, date, host, and library versions. Written once at eval start so
    every figure JSON records exactly how it was produced."""
    import subprocess, sys, time, platform
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:   # cluster copy is not a git repo
        commit = "unknown"
    meta = dict(vars(args))
    meta.update(git_commit=commit, date=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                host=platform.node(), python=sys.version.split()[0],
                numpy=np.__version__, evaluator="_common.eval_stats(seed=42, paired)")
    if extra:
        meta.update(extra)
    path = os.path.splitext(args.out)[0] + ".meta.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(meta, open(path, "w"), indent=2, default=str)
    print(f"meta -> {path}")


def save_json(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(rows, open(path, "w"), indent=2)


def load_json(path):
    return json.load(open(path))


PLOT_RC = {"font.size": 10, "figure.dpi": 150}
COLORS = {"agent": "tab:blue", "swap_asap": "tab:orange", "purify_swap": "tab:green"}
LABELS = {"agent": "Agent", "swap_asap": "Swap-ASAP", "purify_swap": "Purify-then-swap"}
ACTION_COLORS = ["#bdbdbd", "tab:blue", "tab:green"]   # noop, swap, purify
ACTION_LABELS = ["NOOP", "SWAP", "PURIFY"]


def savefig(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")
