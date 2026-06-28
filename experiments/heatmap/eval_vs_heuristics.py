"""Agent-vs-heuristic gap across the (p_gen, p_swap) grid, for the heuristic-baseline
heatmap (companion to the vs-optimum figure).

Panel A: swap-only agent vs swap-asap          gap = (T_agent_swo - T_swapasap)/T_swapasap
Panel B: purify agent    vs purify-then-swap    gap = (T_agent_pur - T_purifyswap)/T_purifyswap

The agents' delivery times are read from the vs-optimum eval json (same config, same
mc_eps/seed), so only the two heuristics are MC-evaluated here -- on identically seeded
networks (paired comparison). gap < 0 => the learned agent beats the heuristic.

  PYTHONPATH=. python experiments/heatmap/eval_vs_heuristics.py --chunk K --nchunks 6
"""
from __future__ import annotations
import argparse, json, math, os

from experiments.heatmap import optimal_baseline as ob
from rl_stack import strategies


def load_agentT(json_path):
    rows = json.load(open(json_path))
    return {(round(r["p_gen"], 2), round(r["p_swap"], 2)):
            (float(r["T_swaponly"]), float(r["T_purify"])) for r in rows}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent_json", default="results/heatmaps/heatmap_gap_N4_9x9.json")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--n_ch", type=int, default=2)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--mc_eps", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--out", default="results/heatmaps/heatmap_heur_N4_9x9.json")
    return ap.parse_args()


def main():
    args = parse_args()
    agentT = load_agentT(args.agent_json)
    pts = [p for i, p in enumerate(sorted(agentT.keys()))
           if i % args.nchunks == args.chunk]
    print(f"{len(pts)} grid points, mc_eps={args.mc_eps}")

    swapasap_fn = lambda env, obs: strategies.swap_asap(env)
    purifyswap_fn = lambda env, obs: strategies.purify_then_swap(env)

    rows = []
    for (pg, ps) in pts:
        Tswo_agent, Tpur_agent = agentT[(pg, ps)]
        Tsa, sd_sa = ob.mc_eval(swapasap_fn, args.N, args.n_ch, pg, ps,
                                args.cutoff, args.horizon, args.mc_eps)
        Tpsw, sd_ps = ob.mc_eval(purifyswap_fn, args.N, args.n_ch, pg, ps,
                                 args.cutoff, args.horizon, args.mc_eps)
        gap_a = 100.0 * (Tswo_agent - Tsa) / Tsa
        gap_b = 100.0 * (Tpur_agent - Tpsw) / Tpsw
        rows.append(dict(N=args.N, p_gen=pg, p_swap=ps,
                         T_swapasap=Tsa, T_purifyswap=Tpsw,
                         T_agent_swaponly=Tswo_agent, T_agent_purify=Tpur_agent,
                         gap_swaponly_pct=gap_a, gap_purify_pct=gap_b,
                         se_swapasap=sd_sa / math.sqrt(args.mc_eps),
                         se_purifyswap=sd_ps / math.sqrt(args.mc_eps)))
        print(f"pg={pg:.2f} ps={ps:.2f} | swo_agent={Tswo_agent:6.3f} vs "
              f"swapasap={Tsa:6.3f} ({gap_a:+5.1f}%) | pur_agent={Tpur_agent:6.3f} "
              f"vs purifyswap={Tpsw:6.3f} ({gap_b:+5.1f}%)", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
