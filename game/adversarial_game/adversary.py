from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import operator
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv

from quantum_repeater_sim.repeater import NO_PARTNER
from rl_stack.buffer import ReplayBuffer


NOOP = 0
DESTROY = 1
QUBIT_FEATURES = 5


class AdversaryFlavor(str, Enum):
    PHOTON_EATER = "photon_eater"
    GATE_DAEMON = "gate_daemon"
    COSMIC_RAY = "cosmic_ray"


@dataclass(frozen=True)
class SabotageTarget:
    node: int
    slot: int
    qubits: tuple[int, ...]


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _channel_count(n_ch) -> int:
    n_ch = _integer(n_ch, "n_ch")
    if n_ch < 2:
        raise ValueError("n_ch must be >= 2")
    return n_ch


def _nonnegative_integer(value, name: str) -> int:
    value = _integer(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _positive_integer(value, name: str) -> int:
    value = _integer(value, name)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _bounded_float(value, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number") from None
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def target_pairs(n_ch: int) -> tuple[tuple[int, int], ...]:
    n_ch = _channel_count(n_ch)
    return tuple(combinations(range(n_ch), 2))


def targets_per_node(flavor: AdversaryFlavor, n_ch: int) -> int:
    n_ch = _channel_count(n_ch)
    flavor = AdversaryFlavor(flavor)
    if flavor is AdversaryFlavor.PHOTON_EATER:
        return n_ch
    if flavor is AdversaryFlavor.GATE_DAEMON:
        return len(target_pairs(n_ch))
    raise NotImplementedError("CosmicRay targets are not implemented")


def decode_target(
    flavor: AdversaryFlavor,
    node: int,
    slot: int,
    n_ch: int,
) -> SabotageTarget:
    node = _nonnegative_integer(node, "node")
    slot = _integer(slot, "slot")
    n_ch = _channel_count(n_ch)
    n_targets = targets_per_node(flavor, n_ch)
    if slot < 0 or slot >= n_targets:
        raise IndexError(f"target slot {slot} out of range [0, {n_targets})")

    flavor = AdversaryFlavor(flavor)
    qubits = (slot,) if flavor is AdversaryFlavor.PHOTON_EATER else target_pairs(n_ch)[slot]
    return SabotageTarget(node=node, slot=slot, qubits=qubits)


def _fixed_width_states(env, n_ch: int):
    n_ch = _channel_count(n_ch)
    states = [env.backend.node_state(node) for node in range(env.N)]
    mismatched = [state.node_id for state in states if state.n_ch != n_ch]
    if mismatched:
        raise ValueError(
            f"configured n_ch={n_ch} does not match nodes {mismatched}"
        )
    return states


def build_adversary_observation(
    env,
    base_obs: Mapping[str, np.ndarray],
    n_ch: int,
) -> dict[str, np.ndarray]:
    states = _fixed_width_states(env, n_ch)
    base_x = np.asarray(base_obs["x"], dtype=np.float32)
    if base_x.shape != (env.N, 10):
        raise ValueError(f"base observation x must have shape ({env.N}, 10)")

    qubit_x = np.empty((env.N, n_ch * QUBIT_FEATURES), dtype=np.float32)
    partner_scale = max(env.N - 1, 1)
    for node, state in enumerate(states):
        partner = np.where(
            state.partner_node == NO_PARTNER,
            -1.0,
            state.partner_node.astype(np.float32) / partner_scale,
        )
        cutoff = max(env.backend.net.repeaters[node].cutoff, 1)
        age = np.minimum(state.age.astype(np.float32) / cutoff, 1.0)
        qubit_x[node] = np.column_stack(
            (
                state.occupied.astype(np.float32),
                state.locked.astype(np.float32),
                partner,
                state.fidelity.astype(np.float32),
                age,
            )
        ).reshape(-1)

    return {
        "x": np.concatenate((base_x, qubit_x), axis=1).astype(np.float32, copy=False),
        "edge_index": np.array(base_obs["edge_index"], dtype=np.int64, copy=True),
    }


def target_mask(
    env,
    flavor: AdversaryFlavor,
    n_ch: int,
) -> np.ndarray:
    n_targets = targets_per_node(flavor, n_ch)
    states = _fixed_width_states(env, n_ch)
    flavor = AdversaryFlavor(flavor)
    mask = np.zeros((env.N, n_targets), dtype=np.bool_)

    if flavor is AdversaryFlavor.PHOTON_EATER:
        for node, state in enumerate(states):
            mask[node] = (~state.occupied) & (~state.locked)
        return mask

    pairs = target_pairs(n_ch)
    for node, state in enumerate(states):
        for slot, (left, right) in enumerate(pairs):
            mask[node, slot] = (
                state.occupied[left]
                and state.occupied[right]
                and not state.locked[left]
                and not state.locked[right]
                and state.partner_node[left] != NO_PARTNER
                and state.partner_node[right] != NO_PARTNER
                and state.partner_node[left] != state.partner_node[right]
            )
    return mask


def obs_to_data(obs: Mapping[str, np.ndarray], device="cpu") -> Data:
    return Data(
        x=torch.tensor(obs["x"], dtype=torch.float32, device=device),
        edge_index=torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
    )


def greedy_action_tensor(
    q_values: torch.Tensor,
    destroy_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    k = _positive_integer(k, "k")
    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values must be a torch.Tensor")
    if q_values.ndim < 1 or q_values.shape[-1] != 2:
        raise ValueError("q_values shape must end with an action dimension of 2")
    if tuple(destroy_mask.shape) != tuple(q_values.shape[:-1]):
        raise ValueError("destroy_mask shape must match q_values.shape[:-1]")

    mask = torch.as_tensor(
        destroy_mask,
        dtype=torch.bool,
        device=q_values.device,
    )
    actions = torch.full(
        mask.shape,
        NOOP,
        dtype=torch.long,
        device=q_values.device,
    )
    advantages = q_values[..., DESTROY] - q_values[..., NOOP]
    eligible = (mask & (advantages > 0)).reshape(-1)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() == 0:
        return actions

    flat_advantages = advantages.reshape(-1)
    count = min(k, eligible_indices.numel())
    selected = eligible_indices[
        torch.topk(flat_advantages[eligible_indices], k=count).indices
    ]
    actions.reshape(-1)[selected] = DESTROY
    return actions


class AdversaryQNetwork(nn.Module):
    def __init__(self, node_dim: int, hidden: int, targets_per_node: int):
        super().__init__()
        self.targets_per_node = int(targets_per_node)
        self.conv1 = SAGEConv(node_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.targets_per_node * 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = F.relu(self.conv1(data.x, data.edge_index))
        return self.head(x).reshape(x.shape[0], self.targets_per_node, 2)


class AdversaryAgent:
    def __init__(
        self,
        flavor,
        n_ch=4,
        hidden=64,
        lr=3e-4,
        gamma=0.99,
        buffer_size=80_000,
        batch_size=64,
        tau=0.005,
        epsilon=1.0,
        k=1,
        rng=None,
        device=None,
    ):
        self.flavor = AdversaryFlavor(flavor)
        if self.flavor is AdversaryFlavor.COSMIC_RAY:
            raise NotImplementedError("CosmicRay adversary is not implemented")

        self.n_ch = _channel_count(n_ch)
        self.k = _positive_integer(k, "k")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        buffer_size = _positive_integer(buffer_size, "buffer_size")
        self.epsilon = _bounded_float(epsilon, "epsilon", 0.0, 1.0)
        self.gamma = _bounded_float(gamma, "gamma", 0.0, 1.0)
        self.tau = _bounded_float(tau, "tau", 0.0, 1.0)
        if self.tau == 0.0:
            raise ValueError("tau must be in (0, 1]")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_count = targets_per_node(self.flavor, self.n_ch)
        self.node_dim = 10 + self.n_ch * QUBIT_FEATURES

        self.policy_net = AdversaryQNetwork(
            self.node_dim,
            hidden,
            self.target_count,
        ).to(self.device)
        self.target_net = AdversaryQNetwork(
            self.node_dim,
            hidden,
            self.target_count,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(max_size=buffer_size)

    def observe(self, env, base_obs) -> dict[str, np.ndarray]:
        return build_adversary_observation(env, base_obs, self.n_ch)

    def get_target_mask(self, env) -> np.ndarray:
        return target_mask(env, self.flavor, self.n_ch)

    def decode(self, node, slot) -> SabotageTarget:
        return decode_target(self.flavor, node, slot, self.n_ch)

    def _random_actions(self, valid: np.ndarray) -> np.ndarray:
        valid = np.asarray(valid, dtype=np.bool_)
        actions = np.full(valid.shape, NOOP, dtype=np.int64)
        valid_slots = np.flatnonzero(valid.reshape(-1))
        max_count = min(self.k, valid_slots.size)
        count = int(self.rng.integers(0, max_count + 1))
        if count:
            selected = self.rng.choice(valid_slots, size=count, replace=False)
            actions.reshape(-1)[selected] = DESTROY
        return actions

    def select_actions(self, env, base_obs, training=True):
        valid = self.get_target_mask(env)
        if training and self.rng.random() < self.epsilon:
            actions = self._random_actions(valid)
        else:
            data = obs_to_data(self.observe(env, base_obs), self.device)
            mask = torch.tensor(valid, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                q_values = self.policy_net(data)
                actions = (
                    greedy_action_tensor(q_values, mask, self.k)
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )

        selected = [
            self.decode(node, slot)
            for node, slot in np.argwhere(actions == DESTROY)
        ]
        return actions, selected

    def train_step(self) -> Optional[float]:
        if self.memory.size() < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states = Batch.from_data_list(
            [obs_to_data(transition["s"], device="cpu") for transition in transitions]
        ).to(self.device)
        next_states = Batch.from_data_list(
            [obs_to_data(transition["s_"], device="cpu") for transition in transitions]
        ).to(self.device)

        actions = torch.cat(
            [
                torch.as_tensor(
                    transition["a"],
                    dtype=torch.long,
                    device=self.device,
                )
                for transition in transitions
            ],
            dim=0,
        )
        next_masks = torch.cat(
            [
                torch.as_tensor(
                    transition["m_"],
                    dtype=torch.bool,
                    device=self.device,
                )
                for transition in transitions
            ],
            dim=0,
        )

        rewards_per_graph = torch.tensor(
            [transition["r"] for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones_per_graph = torch.tensor(
            [float(transition["d"]) for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = rewards_per_graph[states.batch].unsqueeze(1).expand(
            -1, self.target_count
        )
        dones = dones_per_graph[states.batch].unsqueeze(1).expand(
            -1, self.target_count
        )

        q_values = self.policy_net(states)
        current_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_policy_q = self.policy_net(next_states)
            next_actions = []
            for graph_index in range(len(transitions)):
                start = int(next_states.ptr[graph_index])
                end = int(next_states.ptr[graph_index + 1])
                next_actions.append(
                    greedy_action_tensor(
                        next_policy_q[start:end],
                        next_masks[start:end],
                        self.k,
                    )
                )
            next_actions = torch.cat(next_actions, dim=0)
            next_target_q = self.target_net(next_states)
            next_q = next_target_q.gather(
                2,
                next_actions.unsqueeze(-1),
            ).squeeze(-1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            for policy_parameter, target_parameter in zip(
                self.policy_net.parameters(),
                self.target_net.parameters(),
            ):
                target_parameter.copy_(
                    self.tau * policy_parameter
                    + (1.0 - self.tau) * target_parameter
                )

        return loss.item()
