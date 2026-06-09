from __future__ import annotations

import copy
import inspect
import operator

import numpy as np

from quantum_repeater_sim.backends.legacy import LegacyBackend
from quantum_repeater_sim.repeater import QUBIT_FREE
from rl_stack.env_wrapper import QRNEnv

from .adversary import AdversaryFlavor, SabotageTarget, decode_target


class _NonzeroRandom:
    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def random(self, *args, **kwargs):
        value = self._rng.random(*args, **kwargs)
        replacement = np.nextafter(0.0, 1.0)
        if np.isscalar(value):
            return replacement if value == 0.0 else value
        return np.where(value == 0.0, replacement, value)


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


class AdversarialQRNEnv(QRNEnv):
    def __init__(self, flavor: AdversaryFlavor, *args, **kwargs):
        bound = inspect.signature(QRNEnv.__init__).bind_partial(
            None,
            *args,
            **kwargs,
        )
        backend = bound.arguments.get("backend", "legacy")
        if backend != "legacy":
            raise ValueError("AdversarialQRNEnv backend must be legacy")

        self.flavor = AdversaryFlavor(flavor)
        if self.flavor is AdversaryFlavor.COSMIC_RAY:
            raise NotImplementedError("CosmicRay adversary is not implemented")

        super().__init__(*args, **kwargs)
        assert isinstance(self.backend, LegacyBackend)
        self.active_targets: tuple[SabotageTarget, ...] = ()
        self._sabotage_triggered = False
        self._sabotage_result = None

    def _validate_targets(self, targets) -> tuple[SabotageTarget, ...]:
        try:
            targets = tuple(targets)
        except TypeError:
            raise TypeError("targets must be an iterable of SabotageTarget") from None

        validated: list[SabotageTarget] = []
        expected_arity = (
            1 if self.flavor is AdversaryFlavor.PHOTON_EATER else 2
        )
        arity_message = "exactly one" if expected_arity == 1 else "exactly two"

        for target in targets:
            if not isinstance(target, SabotageTarget):
                raise TypeError("every target must be a SabotageTarget")

            node = _integer(target.node, "target node")
            if node < 0 or node >= self.N:
                raise ValueError(f"target node {node} out of bounds")

            slot = _integer(target.slot, "target slot")
            if slot < 0:
                raise ValueError("target slot must be nonnegative")

            try:
                qubits = tuple(target.qubits)
            except TypeError:
                raise TypeError("target qubits must be an iterable") from None
            if len(qubits) != expected_arity:
                raise ValueError(
                    f"{self.flavor.value} target must contain {arity_message} qubit"
                    f"{'s' if expected_arity != 1 else ''}"
                )

            n_ch = self.backend.net.repeaters[node].n_ch
            normalized_qubits = []
            for qubit in qubits:
                qubit = _integer(qubit, "target qubit")
                if qubit < 0 or qubit >= n_ch:
                    raise ValueError(f"target qubit {qubit} out of bounds")
                normalized_qubits.append(qubit)

            canonical = decode_target(self.flavor, node, slot, n_ch)
            if tuple(normalized_qubits) != canonical.qubits:
                raise ValueError(
                    "target qubits do not match the canonical qubits for its slot"
                )

            if canonical in validated:
                raise ValueError("duplicate sabotage target")
            validated.append(canonical)

        if len(validated) > 1:
            raise ValueError("sabotage target budget K=1 exceeded")
        return tuple(validated)

    def _photon_eater_entangle(self, original_entangle, r1: int, r2: int):
        if self._sabotage_triggered:
            return original_entangle(r1, r2)
        if self.backend.net.adj[r1, r2] == 0:
            return original_entangle(r1, r2)

        selected = {}
        for node in (r1, r2):
            repeater = self.backend.net.repeaters[node]
            free = np.flatnonzero(
                (repeater.status == QUBIT_FREE) & (~repeater.locked)
            )
            if free.size == 0:
                return original_entangle(r1, r2)
            selected[node] = int(free[0])

        if any(
            target.node in selected
            and target.qubits[0] == selected[target.node]
            for target in self.active_targets
        ):
            self.backend.net.rng.random()
            self._sabotage_triggered = True
            self._sabotage_result = {
                "success": False,
                "fidelity": 0.0,
                "reason": "photon_eater",
            }
            return self._sabotage_result

        return original_entangle(r1, r2)

    def _gate_daemon_swap(self, original_swap, node: int):
        repeater = self.backend.net.repeaters[node]
        rng = self.backend.net.rng
        rng_state = copy.deepcopy(rng.bit_generator.state)
        try:
            pair = repeater.select_swap_pair(
                self.backend.net._positions,
                rng=rng,
            )
        finally:
            rng.bit_generator.state = rng_state

        if pair is None:
            return original_swap(node)

        selected = tuple(sorted(int(qubit) for qubit in pair))
        targeted = any(
            target.node == node and target.qubits == selected
            for target in self.active_targets
        )
        if not targeted:
            return original_swap(node)

        previous_p_swap = repeater.p_swap
        previous_rng = self.backend.net.rng
        self._sabotage_triggered = True
        try:
            repeater.p_swap = 0.0
            self.backend.net.rng = _NonzeroRandom(previous_rng)
            result = original_swap(node)
            self._sabotage_result = dict(result)
            return result
        finally:
            self.backend.net.rng = previous_rng
            repeater.p_swap = previous_p_swap

    def step_adversarial(self, defender_actions, targets=()):
        transition_targets = self._validate_targets(targets)
        self.active_targets = transition_targets
        self._sabotage_triggered = False
        self._sabotage_result = None
        had_instance_entangle = "entangle" in self.backend.__dict__
        prior_instance_entangle = self.backend.__dict__.get("entangle")
        original_entangle = self.backend.entangle
        had_instance_swap = "swap" in self.backend.__dict__
        prior_instance_swap = self.backend.__dict__.get("swap")
        original_swap = self.backend.swap

        try:
            if self.flavor is AdversaryFlavor.PHOTON_EATER:
                self.backend.entangle = lambda r1, r2: self._photon_eater_entangle(
                    original_entangle,
                    r1,
                    r2,
                )
            elif self.flavor is AdversaryFlavor.GATE_DAEMON:
                self.backend.swap = lambda node: self._gate_daemon_swap(
                    original_swap,
                    node,
                )

            observation, reward, done, info = super().step(defender_actions)
            info = dict(info)
            info["sabotage_targets"] = transition_targets
            info["sabotage_triggered"] = self._sabotage_triggered
            info["sabotage_result"] = self._sabotage_result
            return observation, reward, done, info
        finally:
            if had_instance_entangle:
                self.backend.__dict__["entangle"] = prior_instance_entangle
            else:
                self.backend.__dict__.pop("entangle", None)
            if had_instance_swap:
                self.backend.__dict__["swap"] = prior_instance_swap
            else:
                self.backend.__dict__.pop("swap", None)
            self.active_targets = ()
            self._sabotage_triggered = False
            self._sabotage_result = None
