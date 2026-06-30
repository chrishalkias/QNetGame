"""Shared real-rollout collector for the policy-interpretability plots.

Everything here is grounded in the agent's *actual* visited distribution: load a
checkpoint (node_dim auto-inferred from the weights), roll out the GREEDY policy
over the training distribution, and record for every interior-node decision the
9-feature observation, the chosen action, the conv3 embedding, the Q-value margin,
and the full per-step state (x, edge_index, mask) so features can be permuted and
re-fed for importance analysis.
"""
from __future__ import annotations
import numpy as np
import torch
from rl_stack.model import load_qnet
from rl_stack.agent import _obs_to_data
from rl_stack.env_wrapper import QRNEnv

FEATURE_NAMES = ["occ", "fidelity", "is_target", "avail", "can_swap",
                 "can_purify", "p_gen", "p_swap", "urgency"]
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
            device="cpu", seed=0):
    """Return dict with flat per-decision arrays (X, A, H, margin), the per-step
    states + idx map (for permutation), and the loaded model."""
    model = load_qnet(ckpt, device)
    model.eval()
    cap = {}
    handle = model.conv3.register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", torch.relu(o).detach().cpu().numpy()))
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
                         channel_loss=0.0, dt_seconds=0.0, max_steps=max_steps,
                         topology="chain",
                         rng=np.random.default_rng(int(rng.integers(2**31))))
            obs = env.reset()
            for _ in range(max_steps):
                mask = env.get_action_mask()
                acts, q = greedy(model, obs["x"], obs["edge_index"], mask, device)
                hh = cap["h"]
                si = len(states)
                states.append({"x": obs["x"].copy(),
                               "edge_index": obs["edge_index"], "mask": mask.copy()})
                for i in range(env.N):
                    if i in (env.source, env.dest):
                        continue
                    X.append(obs["x"][i]); A.append(int(acts[i])); H.append(hh[i])
                    top2 = np.sort(q[i])[::-1]
                    margin.append(float(top2[0] - top2[1]))
                    idx.append((si, i))
                obs, _, done, _ = env.step(acts)
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
