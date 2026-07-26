"""
--------------------------------------------------------------------------------
Shared real-rollout collector for the policy-interpretability plots.

Everything here is grounded in the agent's *actual* visited distribution: load a
checkpoint (node_dim auto-inferred from the weights), roll out the GREEDY policy
over the training distribution under the serialized sweep (one micro-decision
at env.active_node per env.step call), and record for every such decision the
8-feature observation, the chosen action, the conv3 embedding, the Q-value
margin, and the full per-step state (x, edge_index, mask) so features can be
permuted and re-fed for importance analysis.
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np
import torch
from rl_stack.model import load_qnet
from rl_stack.agent import _obs_to_data
from rl_stack.env_wrapper import QRNEnv

FEATURE_NAMES = ["occ", "can_swap", "can_purify", "p_gen", "p_swap",
                 "urgency", "relative_position"]
# column 7 (is_active) is the only sweep scaffolding left, not an
# interpretability feature: it is always 1.0 for a collected decision (X is
# recorded at env.active_node), so it carries no signal for permutation
# importance; FEATURE_NAMES stays the 7-column subset that the probes
# analyse, even though X itself (== obs["x"][active_node]) is the full
# 8-wide observation.
ACTION_NAMES = ["NOOP", "SWAP", "PURIFY"]
ACTION_COLORS = ["#d9d9d9", "#1f63d6", "#1ba31b"]   # noop (grey), swap (blue), purify (green)


def greedy(model, x, edge_index, mask, device):
    """Masked-greedy action per node + raw Q-values for one state."""
    with torch.no_grad():
        q = model(_obs_to_data({"x": x, "edge_index": edge_index}, device)).cpu().numpy()
    qm = q.copy()
    qm[~mask] = -1e9
    return qm.argmax(1), q


def collect(ckpt, *, episodes=200, sizes=range(4, 13), n_chs=(2, 3, 4),
            p_lo=0.4, p_hi=0.9, cut_lo=10, cut_hi=40, max_steps=200,
            device="cpu", seed=0, layer="head"):
    """Return dict with flat per-decision arrays (X, A, H, margin), the per-step
    states + idx map (for permutation), and the loaded model.

    `layer` picks the captured representation:
      "conv3" = graph-encoder output (2 layers before Q);
      "head"  = ReLU inside the Q head, i.e. the penultimate (decision) layer."""
    model = load_qnet(ckpt, device)
    model.eval()
    cap = {}
    if layer == "conv3":
        mod, post = model.conv3, lambda o: torch.relu(o).detach().cpu().numpy()
    elif layer == "head":
        mod, post = model.head[1], lambda o: o.detach().cpu().numpy()  # ReLU output
    else:
        raise ValueError(f"unknown layer {layer}")
    handle = mod.register_forward_hook(lambda m, i, o: cap.__setitem__("h", post(o)))
    rng = np.random.default_rng(seed)
    states, X, A, H, margin, idx = [], [], [], [], [], []
    try:
        for _ in range(episodes):
            env = QRNEnv(n_repeaters=int(rng.choice(list(sizes))),
                         n_ch=int(rng.choice(n_chs)),
                         p_gen=float(rng.uniform(p_lo, p_hi)),
                         p_swap=float(rng.uniform(p_lo, p_hi)),
                         cutoff=int(rng.integers(cut_lo, cut_hi + 1)),
                         p_gen_std=0.15, p_swap_std=0.15, F0=1.0,
                         channel_loss=0.0, max_steps=max_steps,
                         rng=np.random.default_rng(int(rng.integers(2**31))))
            obs = env.reset()
            while not env.done:
                r = env.active_node
                mask = env.get_action_mask()
                acts, q = greedy(model, obs["x"], obs["edge_index"], mask, device)
                hh = cap["h"]
                si = len(states)
                states.append({"x": obs["x"].copy(),
                               "edge_index": obs["edge_index"], "mask": mask.copy()})
                X.append(obs["x"][r]); A.append(int(acts[r])); H.append(hh[r])
                top2 = np.sort(q[r])[::-1]
                margin.append(float(top2[0] - top2[1]))
                idx.append((si, r))
                obs, _, done, _ = env.step(int(acts[r]))
                if done:
                    break
    finally:
        handle.remove()
    print(f"collected {len(X)} interior decisions over {episodes} episodes")
    return dict(model=model, device=device, states=states, idx=idx,
                X=np.asarray(X, np.float32), A=np.asarray(A, np.int64),
                H=np.asarray(H, np.float32), margin=np.asarray(margin, np.float32))


def greedy_actions_for_states(model, states, device):
    """Re-feed each stored state; return {state_idx: greedy actions (N,)}."""
    return {si: greedy(model, st["x"], st["edge_index"], st["mask"], device)[0]
            for si, st in enumerate(states)}
