"""Offline policy distillation: teacher Q-values -> tiny student, masked MSE.

Roll the teacher greedily over the omni training distribution, freeze a dataset
of (state, teacher-Q) per step, then regress the student's Q onto the teacher's
Q over VALID actions only. Plain MSE (no softmax/temperature): the teacher's
Q-margins are ~0.005, so a KL objective would wash out to uniform; regression
fits those tiny margins directly. The greedy policy is recovered as argmax over
the student's Q at eval, exactly like the teacher.
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from rl_stack.agent import _obs_to_data
from rl_stack.env_wrapper import QRNEnv
from rl_stack.teacher_student.student_model import StudentQNetwork, STUDENT_FEAT_IDX


# ───────────────────────── data collection ─────────────────────────
def _masked_argmax(q, mask):
    qm = q.copy()
    qm[~mask] = -1e9
    return qm.argmax(1).astype(np.int32)


def _teacher_q(teacher, x, edge_index, device):
    with torch.no_grad():
        d = _obs_to_data({"x": x, "edge_index": edge_index}, device)
        return teacher(d).cpu().numpy()


def collect_teacher_dataset(teacher, *, episodes=400, seed=0, device="cpu",
                            sizes=range(4, 13), n_chs=(2, 3, 4),
                            p_lo=0.4, p_hi=0.9, cut_lo=10, cut_hi=50,
                            max_steps=200, dt_seconds=0.0):
    """Greedy teacher rollouts over the omni training distribution (mirrors
    experiments/policy_probes/_collect.py). Returns a list of per-step records
    {x:(N,9), edge_index:(2,E), mask:(N,3) bool, q:(N,3) teacher Q}."""
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(episodes):
        env = QRNEnv(n_repeaters=int(rng.choice(list(sizes))),
                     n_ch=int(rng.choice(n_chs)),
                     p_gen=float(rng.uniform(p_lo, p_hi)),
                     p_swap=float(rng.uniform(p_lo, p_hi)),
                     cutoff=int(rng.integers(cut_lo, cut_hi + 1)),
                     p_gen_std=0.15, p_swap_std=0.15, F0=1.0,
                     channel_loss=0.0, dt_seconds=dt_seconds, max_steps=max_steps,
                     topology="chain",
                     rng=np.random.default_rng(int(rng.integers(2**31))))
        obs = env.reset()
        for _ in range(max_steps):
            mask = env.get_action_mask()
            q = _teacher_q(teacher, obs["x"], obs["edge_index"], device)
            data.append({"x": obs["x"].copy(), "edge_index": obs["edge_index"].copy(),
                         "mask": mask.copy(), "q": q})
            obs, _, done, _ = env.step(_masked_argmax(q, mask))
            if done:
                break
    return data


# ───────────────────────── loss ─────────────────────────
def masked_mse_loss(q_student, q_teacher, mask):
    """MSE over valid (node, action) entries only. mask (bool/float) zeroes the
    forbidden actions so they contribute no gradient."""
    m = mask.float() if mask.dtype == torch.bool else mask
    sq = ((q_student - q_teacher) ** 2 * m).sum()
    return sq / m.sum().clamp(min=1.0)


# ───────────────────────── training ─────────────────────────
def _record_to_data(rec):
    return Data(
        x=torch.tensor(rec["x"][:, STUDENT_FEAT_IDX], dtype=torch.float32),
        edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
        num_nodes=rec["x"].shape[0],
        q=torch.tensor(rec["q"], dtype=torch.float32),
        mask=torch.tensor(rec["mask"], dtype=torch.float32),
    )


def _dataset_mse(student, items, batch, device):
    student.eval()
    sq = cnt = 0.0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = Batch.from_data_list(items[i:i + batch]).to(device)
            d2 = ((student(b) - b.q) ** 2 * b.mask)
            sq += float(d2.sum()); cnt += float(b.mask.sum())
    return sq / max(cnt, 1.0)


def distill_student(teacher, dataset, *, hidden=16, epochs=30, lr=1e-3,
                    batch=64, val_frac=0.2, seed=0, device="cpu", log=print):
    """Regress a StudentQNetwork onto the teacher's Q over the frozen dataset.
    Returns (student_with_best_weights_loaded, best_state, best_val, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    datas = [_record_to_data(r) for r in dataset]
    perm = rng.permutation(len(datas))
    n_val = max(1, int(len(datas) * val_frac))
    val = [datas[i] for i in perm[:n_val]]
    train = [datas[i] for i in perm[n_val:]]

    student = StudentQNetwork(node_dim=len(STUDENT_FEAT_IDX), hidden=hidden).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    best_val, best_state = float("inf"), None
    hist = {"train_mse": [], "val_mse": []}
    order = np.arange(len(train))
    for ep in range(epochs):
        student.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            items = [train[j] for j in order[i:i + batch]]
            b = Batch.from_data_list(items).to(device)
            loss = masked_mse_loss(student(b), b.q, b.mask)
            opt.zero_grad(); loss.backward(); opt.step()
        tr = _dataset_mse(student, train, batch, device)
        va = _dataset_mse(student, val, batch, device)
        hist["train_mse"].append(tr); hist["val_mse"].append(va)
        log(f"epoch {ep + 1}/{epochs}  train_mse {tr:.6f}  val_mse {va:.6f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in student.state_dict().items()}
    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    return student, best_state, best_val, hist


# ───────────────────────── eval policy fn ─────────────────────────
def student_policy_fn(student, device="cpu"):
    """fn(env, obs) -> masked-greedy actions; drops straight into mc_eval."""
    def fn(env, obs):
        x = torch.tensor(obs["x"][:, STUDENT_FEAT_IDX], dtype=torch.float32, device=device)
        ei = torch.tensor(obs["edge_index"], dtype=torch.long, device=device)
        with torch.no_grad():
            q = student(Data(x=x, edge_index=ei, num_nodes=x.shape[0])).cpu().numpy()
        q[~env.get_action_mask()] = -1e9
        return q.argmax(1).astype(np.int32)
    return fn
