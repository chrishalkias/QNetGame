"""Gap-to-optimal across a (p_gen, p_swap) grid, for the heatmap figure.

Reads exact DP T_opt (from optimal_baseline's json) and MC-evaluates two agents
at each grid point on the identical config:
  - purify-enabled agent  (PURIFY allowed)        -> gap_purify_pct
  - swap-only agent        (PURIFY masked at eval) -> gap_swaponly_pct
where gap% = 100 * (T_agent - T_opt) / T_opt. gap < 0 means the agent beats the
swap-only optimum (only possible for the purify agent, by freeing memory).

  PYTHONPATH=. python experiments/heatmap/eval_heatmap_gap.py \
      --ckpt_purify checkpoints/local/heat_purify/policy.pth \
      --ckpt_swaponly checkpoints/local/heat_swaponly/policy.pth
"""
from __future__ import annotations
import argparse, json, math, os

from experiments.heatmap import optimal_baseline as ob
from rl_stack.env_wrapper import PURIFY


def load_topt(json_path, N):
    rows = json.load(open(json_path))
    return {(round(r["p_gen"], 2), round(r["p_swap"], 2)): float(r["T_opt"])
            for r in rows if int(r["N"]) == N}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topt_json", default="results/heatmaps/heatmap_Topt_N4_9x9.json")
    ap.add_argument("--ckpt_purify", required=True)
    ap.add_argument("--ckpt_swaponly", required=True)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--n_ch", type=int, default=2)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mc_eps", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=0, help="this process's index")
    ap.add_argument("--nchunks", type=int, default=1, help="total processes")
    ap.add_argument("--out", default="results/heatmaps/heatmap_gap_N4_9x9.json")
    return ap.parse_args()


def main():
    args = parse_args()
    topt = load_topt(args.topt_json, args.N)
    pts = [p for i, p in enumerate(sorted(topt.keys()))
           if i % args.nchunks == args.chunk]
    print(f"{len(pts)} grid points (N={args.N}, n_ch={args.n_ch}, "
          f"cutoff={args.cutoff}, H={args.horizon}, mc_eps={args.mc_eps})")

    fn_pur = ob.make_agent_fn(args.ckpt_purify, hidden=args.hidden)
    fn_swo = ob.make_agent_fn(args.ckpt_swaponly, hidden=args.hidden,
                              disable_actions=(PURIFY,))

    rows = []
    for (pg, ps) in pts:
        To = topt[(pg, ps)]
        ta_p, sd_p = ob.mc_eval(fn_pur, args.N, args.n_ch, pg, ps,
                                args.cutoff, args.horizon, args.mc_eps)
        ta_s, sd_s = ob.mc_eval(fn_swo, args.N, args.n_ch, pg, ps,
                                args.cutoff, args.horizon, args.mc_eps)
        gap_p = 100.0 * (ta_p - To) / To
        gap_s = 100.0 * (ta_s - To) / To
        rows.append(dict(N=args.N, p_gen=pg, p_swap=ps, T_opt=To,
                         T_purify=ta_p, T_swaponly=ta_s,
                         gap_purify_pct=gap_p, gap_swaponly_pct=gap_s,
                         se_purify=sd_p / math.sqrt(args.mc_eps),
                         se_swaponly=sd_s / math.sqrt(args.mc_eps)))
        print(f"pg={pg:.2f} ps={ps:.2f} | T_opt={To:6.3f} "
              f"T_pur={ta_p:6.3f}({gap_p:+5.1f}%)  "
              f"T_swo={ta_s:6.3f}({gap_s:+5.1f}%)", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)  # incremental save

    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
