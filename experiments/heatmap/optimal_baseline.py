"""
Exact optimal small-N baseline for the entanglement-distribution MDP.

Roadmap item #1 from docs/paper/peer-review.md: anchor the learned policy's
delivery time against the *optimal* policy on small chains, not only swap-asap.

Method
------
For a homogeneous chain with channel_loss = 0 the only stochastic events in a
QRNEnv step are (i) one Bernoulli(p_swap) per swap and (ii) one Bernoulli(p_gen)
per elementary-generation attempt, plus the uniform shuffle of the auto-entangle
order. Crucially, *delivery time* depends only on which links exist and their
integer ages (links expire at the age cutoff; swap/gen success probabilities are
constant; Werner values affect fidelity but never connectivity or expiry).

Therefore the reachable state space is finite, with a state given canonically by
each qubit's (status, partner, age). We obtain the exact transition kernel by
driving the *real* QRNEnv with an intercepting RNG that enumerates every Bernoulli
branch and every auto-entangle ordering with its exact probability, then solve the
finite-horizon DP

    U_k(s) = 0                                   if s is delivered or k = 0
           = min_a [ 1 + sum_s' P(s'|s,a) * (0 if s' delivered else U_{k-1}(s')) ]

so that U_H(s0) is the minimum expected end-to-end delivery time under a budget of
H steps with failures censored at H -- exactly the metric used in batch_validate.

The optimal policy extracted from the DP is validated by Monte-Carlo rollout in the
unmodified env, and the trained agent and swap-asap are evaluated at the identical
config for a fair gap-to-optimal.

Usage:
    PYTHONPATH=. python experiments/heatmap/optimal_baseline.py
"""
from __future__ import annotations
import argparse, copy, math, itertools, json, os, pickle, sys, time
import numpy as np

from simulator import RepeaterNetwork
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP
from rl_stack import strategies

HUGE = 10**9  # env max_steps during DP: never self-terminate by budget


# ───────────────────────── intercepting RNG ─────────────────────────
class NeedDecision(Exception):
    def __init__(self, p: float):
        self.p = p


class PlannedRNG:
    """Replays a forced sequence of Bernoulli outcomes; signals a branch
    when the sequence is exhausted. Applies a fixed auto-entangle ordering."""
    def __init__(self, plan, perm):
        self.plan = plan
        self.i = 0
        self.perm = perm
        self.ctx = {"p": 1.0}      # success prob of the *current* draw

    def random(self, *a, **k):
        if self.i < len(self.plan):
            v = self.plan[self.i]
            self.i += 1
            return v
        raise NeedDecision(self.ctx["p"])

    def shuffle(self, x):
        if self.perm is not None and len(x) == len(self.perm):
            x[:] = [x[j] for j in self.perm]

    # never expected on the FARTHEST swap policy, but be safe
    def integers(self, *a, **k):
        return 0


_PATCHED = False

def _patch_class_once():
    """Wrap RepeaterNetwork.entangle/.swap at the class level (instances use
    __slots__, so per-instance patching is impossible). The wrappers tag the
    upcoming Bernoulli draw with its success probability when the network's RNG
    is a PlannedRNG, and are inert for ordinary numpy generators."""
    global _PATCHED
    if _PATCHED:
        return
    real_ent = RepeaterNetwork.entangle
    real_swap = RepeaterNetwork.swap

    def ent(self, r1, r2):
        if isinstance(self.rng, PlannedRNG):
            self.rng.ctx["p"] = self._gen_prob(r1, r2)   # = p_gen, channel_loss=0
        return real_ent(self, r1, r2)

    def swp(self, r):
        if isinstance(self.rng, PlannedRNG):
            self.rng.ctx["p"] = self.repeaters[r].p_swap
        return real_swap(self, r)

    RepeaterNetwork.entangle = ent
    RepeaterNetwork.swap = swp
    _PATCHED = True


def _install(env, plan, perm):
    """Point env's (and its network's) RNG at an intercepting PlannedRNG."""
    _patch_class_once()
    rng = PlannedRNG(plan, perm)
    env.rng = rng
    env.net.rng = rng
    return rng


# ───────────────────────── state abstraction ─────────────────────────
def state_key(env):
    """Canonical, delivery-relevant state: per repeater, the sorted tuple of
    (status, partner, age) over its qubits. Werner/steps are irrelevant."""
    out = []
    for rep in env.net.repeaters:
        qs = []
        for qi in range(rep.n_ch):
            if int(rep.status[qi]) == 1:      # QUBIT_OCCUPIED
                qs.append((1, int(rep.partner_repeater[qi]), int(rep.age[qi])))
            else:
                qs.append((0, -1, 0))
        out.append(tuple(sorted(qs)))
    return tuple(out)


def is_delivered(env):
    connected, _ = env._check_e2e()
    return bool(connected)


# ───────────────────────── transition enumeration ─────────────────────────
def enumerate_transition(base_env, action):
    """Exact next-state distribution for (base_env, action).

    Returns {next_key: [prob, delivered, repr_env]} summed over all Bernoulli
    branches and all auto-entangle orderings."""
    npairs = int(np.count_nonzero(np.triu(base_env.net.adj, k=1)))
    w_perm = 1.0 / math.factorial(npairs)
    res = {}
    act = np.asarray(action, dtype=int)

    def record(env2, w):
        k = state_key(env2)
        if k in res:
            res[k][0] += w
        else:
            res[k] = [w, is_delivered(env2), env2]

    def dfs(plan, w, perm):
        env2 = copy.deepcopy(base_env)
        _install(env2, plan, perm)
        try:
            env2.step(act)
        except NeedDecision as nd:
            p = nd.p
            if p > 0.0:
                dfs(plan + [0.0], w * p, perm)        # u=0 -> success
            if p < 1.0:
                dfs(plan + [1.0], w * (1.0 - p), perm)  # u=1 -> failure
            return
        record(env2, w)

    for perm in itertools.permutations(range(npairs)):
        dfs([], w_perm, perm)
    return res


# ───────────────────────── DP solver ─────────────────────────
def joint_actions(N):
    interior = [i for i in range(N) if i not in (0, N - 1)]
    acts = []
    for combo in itertools.product([NOOP, SWAP], repeat=len(interior)):
        a = np.zeros(N, dtype=int)
        for node, c in zip(interior, combo):
            a[node] = c
        acts.append(a)
    return acts


def build_kernel(N, n_ch, p_gen, p_swap, cutoff):
    """Enumerate the exact MDP: reachable states, terminal flags, transition
    kernel for every joint action, a representative env per state, and the
    initial (post-reset) distribution D0."""
    base = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                  F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=HUGE,
                  topology="chain", rng=np.random.default_rng(0))
    empty = copy.deepcopy(base)
    for rep in empty.net.repeaters:
        rep.reset()
    empty.steps = 0
    empty.done = False

    acts = joint_actions(N)
    noop = np.zeros(N, dtype=int)
    D0_full = enumerate_transition(empty, noop)
    D0 = {k: (p, d) for k, (p, d, _) in D0_full.items()}

    state_env, delivered = {}, {}
    for k, (p, d, e) in D0_full.items():
        state_env.setdefault(k, e)
        delivered[k] = d
    trans_cache = {}
    frontier = [k for k, d in delivered.items() if not d]
    while frontier:
        k = frontier.pop()
        e = state_env[k]
        for ai, a in enumerate(acts):
            r = enumerate_transition(e, a)
            trans_cache[(k, ai)] = {k2: (p2, d2) for k2, (p2, d2, _) in r.items()}
            for k2, (p2, d2, e2) in r.items():
                if k2 not in state_env:
                    state_env[k2] = e2
                    delivered[k2] = d2
                    if not d2:
                        frontier.append(k2)
    return dict(acts=acts, keys=list(state_env), delivered=delivered,
                trans=trans_cache, D0=D0, state_env=state_env)


def _horizon_value(action_for, ker, H):
    """Finite-horizon backward induction. action_for(k) -> chosen action index,
    or None to minimise over all actions (optimal)."""
    keys, delivered, trans, acts = ker["keys"], ker["delivered"], ker["trans"], ker["acts"]
    U = {k: 0.0 for k in keys}
    for _ in range(H):
        Un = {}
        for k in keys:
            if delivered[k]:
                Un[k] = 0.0
                continue
            if action_for is None:
                best = math.inf
                for ai in range(len(acts)):
                    r = trans.get((k, ai))
                    if not r:
                        continue
                    e = 1.0 + sum(p2 * (0.0 if d2 else U[k2]) for k2, (p2, d2) in r.items())
                    best = min(best, e)
                Un[k] = best
            else:
                ai = action_for(k)
                r = trans.get((k, ai))
                Un[k] = 1.0 + sum(p2 * (0.0 if d2 else U[k2]) for k2, (p2, d2) in r.items()) if r else U[k]
        U = Un
    return sum(p * (0.0 if d else U[k]) for k, (p, d) in ker["D0"].items())


def _greedy_policy(ker, H):
    """Optimal greedy action per state (for MC validation in the real env)."""
    keys, delivered, trans, acts = ker["keys"], ker["delivered"], ker["trans"], ker["acts"]
    U = {k: 0.0 for k in keys}
    for _ in range(H):
        Un = {}
        for k in keys:
            if delivered[k]:
                Un[k] = 0.0
                continue
            best = math.inf
            for ai in range(len(acts)):
                r = trans.get((k, ai))
                if r:
                    best = min(best, 1.0 + sum(p2 * (0.0 if d2 else U[k2]) for k2, (p2, d2) in r.items()))
            Un[k] = best
        U = Un
    pol = {}
    for k in keys:
        if delivered[k]:
            continue
        best, ba = math.inf, 0
        for ai in range(len(acts)):
            r = trans.get((k, ai))
            if r:
                e = 1.0 + sum(p2 * (0.0 if d2 else U[k2]) for k2, (p2, d2) in r.items())
                if e < best:
                    best, ba = e, ai
        pol[k] = ba
    return pol


def _act_index(acts, a):
    for i, aa in enumerate(acts):
        if np.array_equal(aa, a):
            return i
    return 0  # all-NOOP fallback


def swapasap_action_for(ker):
    """swap-asap as a function of abstracted state (exact, no MC)."""
    def f(k):
        env = ker["state_env"][k]
        a = strategies.swap_asap(env)
        return _act_index(ker["acts"], a)
    return f


# ───────────────────────── Monte-Carlo evaluation ─────────────────────────
def mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes, seed=42,
            p_gen_std=0.0, p_swap_std=0.0):
    # p_gen_std/p_swap_std > 0 -> per-repeater inhomogeneity (fresh draw each
    # episode); =0 keeps the homogeneous RNG stream bit-for-bit.
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(n_episodes):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     p_gen_std=p_gen_std, p_swap_std=p_swap_std,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(rng.integers(2**32)))
        obs = env.reset()
        step = 0
        for step in range(H):
            a = policy_fn(env, obs)
            obs, _, done, info = env.step(a)
            if done:
                break
        times.append(step + 1 if (done and info.get("fidelity", 0.0) > 0) else H)
    return float(np.mean(times)), float(np.std(times))


def optimal_policy_fn(policy, acts):
    def fn(env, obs):
        k = state_key(env)
        ai = policy.get(k, 0)
        return acts[ai]
    return fn


def swap_asap_fn(env, obs):
    return strategies.swap_asap(env)


def make_agent_fn(ckpt, hidden=64, disable_actions=None):
    """Policy fn for a trained checkpoint. `disable_actions` masks the given
    action columns at inference (e.g. (PURIFY,) for a swap-only evaluation)."""
    import torch
    from rl_stack.agent import QRNAgent
    agent = QRNAgent(hidden=hidden)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(sd)
    agent.policy_net.eval()

    def fn(env, obs):
        mask = env.get_action_mask()
        if disable_actions:
            mask = mask.copy()
            for a in disable_actions:
                mask[:, a] = False
        return agent.select_actions(obs, mask, training=False)
    return fn


# ───────────────────────── policy persistence ─────────────────────────
def save_policy(policy_dir, N, n_ch, cutoff, horizon, pg, ps, pol, acts):
    """Persist the exact optimal greedy policy for one (N, p_gen, p_swap) point.

    The pickle stores everything a *consumer* needs to act with this policy:
    the state->action_index map, the joint-action table, and the config it was
    solved for. To use it later as a baseline, load it and map a live env via
    `optimal_baseline.state_key(env)` -> policy[key] -> acts[idx]; an unseen key
    (a state never reached during enumeration) falls back to all-NOOP index 0."""
    os.makedirs(policy_dir, exist_ok=True)
    fname = (f"optimal_policy_N{N}_ch{n_ch}_co{cutoff}_h{horizon}"
             f"_pg{pg:.2f}_ps{ps:.2f}.pkl")
    path = os.path.join(policy_dir, fname)
    payload = {
        "config": dict(N=N, n_ch=n_ch, cutoff=cutoff, horizon=horizon,
                       p_gen=pg, p_swap=ps),
        "acts": [a.tolist() for a in acts],   # joint actions, indexable by policy values
        "policy": {k: int(v) for k, v in pol.items()},  # state_key -> action index
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# ───────────────────────── driver ─────────────────────────
def _parse_points(spec):
    """'0.5:0.7,0.3:0.3' -> [(0.5,0.7),(0.3,0.3)]."""
    pts = []
    for chunk in spec.split(","):
        pg, ps = chunk.split(":")
        pts.append((float(pg), float(ps)))
    return pts


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exact optimal-policy MDP baseline for the QRN chain "
                    "(value iteration via finite-horizon DP).")
    p.add_argument("--n_list", type=str, default="3,4",
                   help="comma-separated chain sizes, e.g. '3,4,5'")
    p.add_argument("--n_ch", type=int, default=2,
                   help="qubits per repeater (exact DP is only tractable for 2)")
    p.add_argument("--cutoff", type=int, default=5, help="memory age cutoff")
    p.add_argument("--horizon", type=int, default=30, help="DP / rollout horizon H")
    p.add_argument("--points", type=str, default=None,
                   help="override (p_gen:p_swap) scan, e.g. '0.5:0.7,0.9:0.9'")
    p.add_argument("--mc_eps", type=int, default=20000,
                   help="Monte-Carlo episodes for the agent column")
    p.add_argument("--mc_eps_opt", type=int, default=4000,
                   help="Monte-Carlo episodes validating the DP optimum")
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/sota/policy.pth",
                   help="agent checkpoint for the gap-to-optimal column (optional)")
    p.add_argument("--out_json", type=str, default="results/optimal/optimal_baseline.json")
    p.add_argument("--policy_dir", type=str, default="results/optimal/optimal_policies")
    p.add_argument("--save_policy", action="store_true",
                   help="dump the exact optimal policy per (N, p_gen, p_swap)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    N_CH, CUTOFF, H = args.n_ch, args.cutoff, args.horizon
    N_LIST = [int(x) for x in args.n_list.split(",") if x.strip()]
    POINTS = (_parse_points(args.points) if args.points else
              [(0.3, 0.3), (0.5, 0.5), (0.5, 0.7), (0.7, 0.5),
               (0.3, 0.7), (0.7, 0.3), (0.9, 0.9)])
    MC_EPS = args.mc_eps

    try:
        agent_fn = make_agent_fn(args.ckpt)
        have_agent = True
    except Exception as e:
        print(f"[warn] agent unavailable ({e}); skipping agent column")
        agent_fn, have_agent = None, False

    rows = []
    print(f"config: n_ch={N_CH} cutoff={CUTOFF} horizon H={H} mc_eps={MC_EPS} "
          f"N_list={N_LIST} save_policy={args.save_policy}")
    print("T_opt, T_swap exact (DP); T_agent Monte-Carlo (+- SE)\n")
    hdr = (f"{'N':>2} {'p_gen':>5} {'p_swap':>6} {'states':>6} "
           f"{'T_opt':>6} {'T_swap':>6} {'T_agent':>8} "
           f"{'opt<swap%':>9} {'agent_gap%':>10} {'a_vs_swap%':>10}")
    print(hdr)
    print("-" * len(hdr))
    for N in N_LIST:
        for pg, ps in POINTS:
            t0 = time.time()
            ker = build_kernel(N, N_CH, pg, ps, CUTOFF)
            nstates = len(ker["keys"])
            T_opt = _horizon_value(None, ker, H)
            T_swap = _horizon_value(swapasap_action_for(ker), ker, H)
            # validate exact optimal against MC rollout of the greedy policy
            pol = _greedy_policy(ker, H)
            T_opt_mc, _ = mc_eval(optimal_policy_fn(pol, ker["acts"]),
                                  N, N_CH, pg, ps, CUTOFF, H, args.mc_eps_opt)
            policy_path = None
            if args.save_policy:
                policy_path = save_policy(args.policy_dir, N, N_CH, CUTOFF, H,
                                          pg, ps, pol, ker["acts"])
            if have_agent:
                ta, sa = mc_eval(agent_fn, N, N_CH, pg, ps, CUTOFF, H, MC_EPS)
                T_agent = ta
                se = sa / math.sqrt(MC_EPS)
            else:
                T_agent, se = float("nan"), float("nan")
            ovs = 100 * (T_swap - T_opt) / T_swap
            gap = 100 * (T_agent - T_opt) / T_opt if have_agent else float("nan")
            avs = 100 * (T_swap - T_agent) / T_swap if have_agent else float("nan")
            rows.append(dict(N=N, p_gen=pg, p_swap=ps, states=nstates,
                             T_opt=T_opt, T_opt_mc=T_opt_mc, T_swap=T_swap,
                             T_agent=T_agent, T_agent_se=se, opt_vs_swap=ovs,
                             agent_gap_pct=gap, agent_vs_swap=avs,
                             policy_path=policy_path, secs=time.time() - t0))
            print(f"{N:>2} {pg:>5} {ps:>6} {nstates:>6} "
                  f"{T_opt:>6.3f} {T_swap:>6.3f} {T_agent:>6.3f}±{se:.2f} "
                  f"{ovs:>9.1f} {gap:>10.1f} {avs:>10.1f}   "
                  f"[opt_mc={T_opt_mc:.3f} {time.time()-t0:.0f}s]", flush=True)
            # Incremental save: a walltime kill mid-sweep then preserves every
            # completed point (N=5 builds can take hours each). Policy pickles
            # are already written per-point above.
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(rows, f, indent=2)

    print(f"\nsaved -> {args.out_json}")
    if args.save_policy:
        print(f"policies -> {args.policy_dir}/ ({len(rows)} files)")
    maxdiff = max(abs(r["T_opt"] - r["T_opt_mc"]) for r in rows)
    print(f"max |T_opt(DP) - T_opt(MC)| = {maxdiff:.4f}  (validates exact kernel)")
    bad = [r for r in rows if r["opt_vs_swap"] < -0.5]
    print("optimal is a valid lower bound (opt<=swap) at all points: "
          f"{'YES' if not bad else 'NO -- ' + str(len(bad)) + ' violations'}")


if __name__ == "__main__":
    main()
