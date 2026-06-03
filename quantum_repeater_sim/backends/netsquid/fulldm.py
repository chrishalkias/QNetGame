"""Full density-matrix NetSquid backend (M3): real qubits, real BSM swap,
depolarizing decoherence. Subclasses the analytic NetSquidBackend and overrides
only the physics hooks; all slot bookkeeping / timing / stale-guard is inherited.
Chain only, swap only (no purify). One instance per process (global sim state).
"""
from __future__ import annotations
import numpy as np
import netsquid as ns
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.ketstates as ks

from .backend import NetSquidBackend, NO_PARTNER


class FullDMBackend(NetSquidBackend):
    def __init__(self, *args, **kwargs):
        ns.set_qstate_formalism(ns.QFormalism.DM)
        super().__init__(*args, **kwargs)
        # per-slot real qubit store (None when free); parallels the M1 arrays
        self._qubits = np.empty((self._N, self._n_ch), dtype=object)
        self._qubits[:] = None

    # ---- physics hooks (real qubits) ----
    def _create_link(self, a, qa, b, qb, F, ec):
        # real Bell pair depolarized to target fidelity F
        q_a, q_b = qapi.create_qubits(2)
        qapi.operate(q_a, ns.H)
        qapi.operate([q_a, q_b], ns.CNOT)
        p = 4.0 * (1.0 - F) / 3.0          # one-sided depolarizing prob for target F
        if p > 0:
            qapi.depolarize(q_a, prob=p)
        self._qubits[a, qa] = q_a
        self._qubits[b, qb] = q_b
        # bookkeeping (p0 unused in full_dm; pass 0.0)
        self._set_link(a, qa, b, qb, 0.0, 0, ec)
        self._set_link(b, qb, a, qa, 0.0, 0, ec)

    def _read_fidelity(self, node, q):
        qubit = self._qubits[node, q]
        pn, pq = int(self._partner_node[node, q]), int(self._partner_qubit[node, q])
        if qubit is None or pn == NO_PARTNER:
            return 0.0           # free slot or transient locked singleton
        partner = self._qubits[pn, pq]
        if partner is None:
            return 0.0
        return float(qapi.fidelity([qubit, partner], ks.b00, squared=True))

    def _discard(self, node, q):
        self._qubits[node, q] = None

    def _decohere_tick(self):
        # Depolarize EVERY occupied qubit each tick with q = 1 - exp(-1/(2*cutoff)).
        # A two-ended pair then decays by (1-q)^2 = exp(-1/cutoff) per tick (matches
        # analytic), with NO dedup needed and no double-count for post-swap pairs
        # (whose two qubits are both occupied). A transient locked singleton gets
        # the half-rate (1-q); once it pairs into the e2e link both halves apply.
        for node in range(self._N):
            for q in np.flatnonzero(self._occupied[node]):
                q = int(q)
                qubit = self._qubits[node, q]
                if qubit is None:
                    continue
                cutoff = max(int(self._link_cutoff[node, q]), 1)
                q_tick = 1.0 - np.exp(-1.0 / (2.0 * cutoff))
                if q_tick > 0:
                    qapi.depolarize(qubit, prob=q_tick)

    def reset(self) -> None:
        super().reset()                    # clears arrays + ns.sim_reset via clock
        self._qubits[:] = None
