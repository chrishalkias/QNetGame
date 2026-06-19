"""Compare-smoke against the EXACT optimal, n_ch=2 chain.

Trains a SWAP-ONLY agent (PURIFY masked) on a homogeneous N=4 chain where each
episode draws (p_gen, p_swap) from the discrete grid {0.3,0.5,0.7,0.9}^2, and
logs — per episode, on the SAME seeded network — the greedy agent, swap-asap,
random AND the exact DP-optimal (swap-only) policy. The optimal exists only as
precomputed pickles at that grid / cutoff=5 / horizon=30 (on-the-fly DP is ~71s
per point), so we sample params from the grid and dispatch to the matching
pickle. Expectation: the agent starts below swap-asap, overtakes it, then
converges toward the optimal line (steps/success panels of training_compare.png).

  PYTHONPATH=$(pwd) python experiments/compare_optimal.py --episodes 500 \
      --run_id compare_optimal_smoke
"""
from __future__ import annotations
import argparse
import os
import pickle

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

from experiments import optimal_baseline as ob  # noqa: E402
from rl_stack import QRNAgent           # noqa: E402
from rl_stack.env_wrapper import PURIFY  # noqa: E402

GRID = (0.30, 0.50, 0.70, 0.90)
N, N_CH, CUTOFF, HORIZON = 4, 2, 5, 30
POLICY_DIR = os.path.join(_ROOT, "results", "optimal_policies")


def build_optimal_dispatch():
    """Load the 16 precomputed optimal policies and return a single
    optimal_dispatch(env, obs) that picks the one matching the episode's
    (p_gen, p_swap) read off the (homogeneous) network."""
    opt_fns = {}
    for pg in GRID:
        for ps in GRID:
            fname = (f"optimal_policy_N{N}_ch{N_CH}_co{CUTOFF}_h{HORIZON}"
                     f"_pg{pg:.2f}_ps{ps:.2f}.pkl")
            path = os.path.join(POLICY_DIR, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"missing optimal pickle: {path}\n"
                    "Generate it via experiments/optimal_baseline.py first.")
            with open(path, "rb") as f:
                payload = pickle.load(f)
            acts = [np.asarray(a, dtype=int) for a in payload["acts"]]
            opt_fns[(pg, ps)] = ob.optimal_policy_fn(payload["policy"], acts)

    def optimal_dispatch(env, obs):
        ns = env.backend.node_state(0)        # homogeneous -> node 0 carries both
        key = (round(float(ns.p_gen), 2), round(float(ns.p_swap), 2))
        return opt_fns[key](env, obs)

    return optimal_dispatch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--run_id", type=str, default="compare_optimal_smoke")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_base_dir", type=str, default="checkpoints")
    args = p.parse_args(argv)

    save_path = os.path.join(args.save_base_dir, args.run_id)
    os.makedirs(save_path, exist_ok=True)

    optimal_dispatch = build_optimal_dispatch()
    print(f"[ok] loaded {len(GRID)**2} optimal policies "
          f"(N={N}, n_ch={N_CH}, cutoff={CUTOFF}, grid={GRID})")

    agent = QRNAgent(lr=args.lr, hidden=args.hidden, buffer_size=80_000,
                     gamma=0.99, tau=0.005, epsilon=1.0)

    agent.train(
        episodes=args.episodes,
        max_steps=HORIZON,
        n_range=[N],
        n_ch=N_CH,
        p_gen=set(GRID),          # per-episode discrete grid draw
        p_swap=set(GRID),
        p_gen_std=0.0, p_swap_std=0.0,
        cutoff=CUTOFF,
        F0=1.0, channel_loss=0.0, dt_seconds=0.0,
        curriculum=False,
        topology="chain",
        backend="legacy",
        disable_actions=(PURIFY,),          # SWAP-ONLY agent (vs swap-only optimum)
        compare=True,
        compare_extra={"optimal": optimal_dispatch},
        save_path=save_path,
        plot=True,
    )
    print(f"[done] checkpoint + plots in {save_path}/")
    print(f"       see {save_path}/training_compare.png "
          "(steps/success panels: agent should converge to the Optimal line)")


if __name__ == "__main__":
    main()
