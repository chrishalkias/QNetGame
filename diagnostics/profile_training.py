"""
Profile where training wall-clock actually goes: env step vs GNN inference vs
replay update. Answers "are we env-bound or compute-bound?" before anyone
reaches for JAX / multiprocessing / a bigger GPU.

Three phases timed + cProfiled separately:
  1. env  : env.step() + get_action_mask()      (pure NumPy simulator + obs build)
  2. infer: select_actions(training=False)       (obs->Data + GNN forward, greedy)
  3. update: train_step()                         (batch assembly + fwd/bwd)

Run:  PYTHONPATH=. python diagnostics/profile_training.py --n_repeaters 10 --steps 300 --updates 200
"""
from __future__ import annotations
import argparse
import cProfile
import io
import pstats
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_repeaters", type=int, default=10)
    p.add_argument("--n_ch", type=int, default=4)
    p.add_argument("--p_gen", type=float, default=0.8)
    p.add_argument("--p_swap", type=float, default=0.7)
    p.add_argument("--cutoff", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=300, help="env/infer iterations")
    p.add_argument("--updates", type=int, default=200, help="train_step calls")
    p.add_argument("--topn", type=int, default=12, help="hottest functions to print")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _rand_actions(env, rng):
    mask = env.get_action_mask()
    a = np.zeros(env.N, dtype=np.int32)
    for i in range(env.N):
        valid = np.flatnonzero(mask[i])
        a[i] = rng.choice(valid) if len(valid) else 0
    return a


def _profile(fn, label, topn):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    n = fn()
    pr.disable()
    dt = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(topn)
    # keep only the function rows (drop pstats preamble)
    rows = [ln for ln in s.getvalue().splitlines() if ln.strip()][5:]
    print(f"\n===== {label}: {n} calls, {dt*1e3:.1f} ms total, "
          f"{dt/max(n,1)*1e3:.3f} ms/call =====")
    print("\n".join(rows[:topn + 1]))
    return dt / max(n, 1)


def main():
    args = parse_args()
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack.agent import QRNAgent

    rng = np.random.default_rng(args.seed)
    env = QRNEnv(n_repeaters=args.n_repeaters, n_ch=args.n_ch,
                 p_gen=args.p_gen, p_swap=args.p_swap, cutoff=args.cutoff,
                 max_steps=args.max_steps, topology="chain",
                 rng=np.random.default_rng(args.seed))
    agent = QRNAgent(hidden=args.hidden, rng=np.random.default_rng(args.seed))
    print(f"device={agent.device}  N={env.N}  hidden={args.hidden}  "
          f"batch_size={agent.batch_size}")

    obs = env.reset()

    # ---- warm up the replay buffer so train_step does real work ----
    while agent.memory.size() <= agent.batch_size + args.updates:
        mask = env.get_action_mask()
        a = _rand_actions(env, rng)
        nobs, r, done, info = env.step(a)
        agent.memory.add(obs, a, r, nobs, done, env.get_action_mask())
        obs = nobs if not done else env.reset()

    # ---- phase 1: env step (simulator + observation build) ----
    def env_phase():
        o = env.reset()
        for _ in range(args.steps):
            a = _rand_actions(env, rng)
            o, r, d, _ = env.step(a)
            if d:
                o = env.reset()
        return args.steps
    t_env = _profile(env_phase, "ENV step (sim + obs)", args.topn)

    # ---- phase 2: greedy inference (obs->Data + GNN forward) ----
    o2 = env.reset()
    def infer_phase():
        nonlocal o2
        for _ in range(args.steps):
            m = env.get_action_mask()
            agent.select_actions(o2, m, training=False)
            a = _rand_actions(env, rng)
            o2, r, d, _ = env.step(a)
            if d:
                o2 = env.reset()
        return args.steps
    # subtract the env portion already measured so this is ~pure inference
    t_infer_raw = _profile(infer_phase, "INFER select_actions (greedy GNN)", args.topn)
    t_infer = max(t_infer_raw - t_env, 0.0)

    # ---- phase 3: replay update ----
    def update_phase():
        for _ in range(args.updates):
            agent.train_step()
        return args.updates
    t_update = _profile(update_phase, "UPDATE train_step (batch + fwd/bwd)", args.topn)

    # ---- headline split: one real training iteration ----
    # train() does per step: get_action_mask x2 + select + env.step + train_step
    per_iter = t_env + t_infer + t_update
    print("\n" + "=" * 60)
    print("PER-TRAINING-ITERATION WALL-CLOCK SPLIT (ms, lower is faster)")
    print("=" * 60)
    for label, t in (("env.step + obs", t_env),
                     ("GNN inference", t_infer),
                     ("train_step (update)", t_update)):
        print(f"  {label:24s} {t*1e3:7.3f} ms   {t/per_iter*100:5.1f}%")
    print(f"  {'TOTAL / iter':24s} {per_iter*1e3:7.3f} ms")
    sim_share = (t_env + t_infer) / per_iter * 100
    print(f"\n  simulator+inference (NumPy/Python) = {sim_share:.0f}% of a step")
    print("  -> env-bound: multiprocessing.Pool of envs is the cheap win."
          if sim_share >= 55 else
          "  -> update-bound: GNN/optimizer dominates; batching helps.")


if __name__ == "__main__":
    main()
