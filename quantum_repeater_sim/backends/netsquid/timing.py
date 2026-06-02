"""pydynaa-driven simulation clock for the analytic NetSquid backend.

Replaces the bespoke ``pending_events`` queue: deferred operations are scheduled
as pydynaa events and fire (via per-event one-shot callbacks) when ``advance``
runs the engine forward by one tick. One env ``step`` == one ``advance`` ==
``ns.sim_run(duration=TICK_NS)``.
"""
from __future__ import annotations
import math
from typing import Callable

import netsquid as ns
import pydynaa

# One env step advances the pydynaa clock by this many ns. Arbitrary unit — only
# the NUMBER of ticks matters (delays are expressed in ticks). Kept at 1.0 so
# tick count == sim_time / TICK_NS.
TICK_NS: float = 1.0

# pydynaa processes an event landing *exactly* on a tick boundary only once the
# engine advances strictly past it (an event at t=2.0 is not run by
# sim_run(duration=1.0) that ends at t=2.0, but by the following segment). To
# make an event due after N ticks fire on the N-th ``advance``, we nudge its
# scheduled time just inside the N-th tick window (N*TICK_NS - _TICK_EPS_NS).
_TICK_EPS_NS: float = 1e-6


class _Scheduler(pydynaa.Entity):
    """pydynaa entity that fires an independent one-shot callback per event.

    SINGLE-USE PER ENGINE EPOCH. The ``_n_pending`` count is only correct
    because an instance is DISCARDED on ``ns.sim_reset()`` and a fresh
    ``_Scheduler`` is created in its place (see ``SimClock.reset``). After a
    reset the engine drops all previously-scheduled events, so the orphaned
    events belonging to a stale scheduler will NEVER fire — their ``_on_fire``
    handlers never run, and the decrements that would balance the increments
    done in ``schedule`` never happen. Reusing a scheduler across a
    ``sim_reset`` would therefore leave ``_n_pending`` permanently inflated.
    Never reuse an instance across engine epochs; always create a new one.
    """

    EV = pydynaa.EventType("RESOLVE", "a deferred operation is due")

    def __init__(self):
        # pydynaa.Entity is a C-extension base that needs no super().__init__():
        # instances expose _schedule_after/_wait_once without init-time setup.
        # Do not "fix" this by adding super().__init__().
        self._n_pending = 0

    def schedule(self, delay_ns: float, callback: Callable[[], None]) -> None:
        # Floor a non-positive delay to a tiny POSITIVE value: a 0-tick event
        # computes a slightly-negative scheduled time (0*TICK_NS - _TICK_EPS_NS),
        # so it is clamped here to fire on the very next advance(). Any value far
        # below TICK_NS works.
        ev = self._schedule_after(max(delay_ns, 1e-9), self.EV)
        self._n_pending += 1

        def _on_fire(_ev, cb=callback):
            self._n_pending -= 1
            cb()

        handler = pydynaa.EventHandler(_on_fire)
        self._wait_once(handler, entity=self, event_type=self.EV, event_id=ev.id)

    @property
    def n_pending(self) -> int:
        return self._n_pending


class SimClock:
    """Tick-based wrapper over NetSquid's global pydynaa engine.

    NOTE: NetSquid simulation state is process-global, so only one SimClock may
    be actively driving the engine at a time. ``reset`` calls ``ns.sim_reset``.
    """

    def __init__(self, c_fiber: float = 200_000.0, dt_seconds: float = 0.0):
        self.c_fiber = float(c_fiber)
        self.dt_seconds = float(dt_seconds)
        self._sched = _Scheduler()
        self.tick = 0

    def reset(self) -> None:
        ns.sim_reset()
        self._sched = _Scheduler()
        self.tick = 0

    def delay_ticks(self, d_km: float) -> int:
        """Classical-comm delay in ticks (legacy formula; 0 when dt_seconds==0)."""
        if d_km <= 0.0 or self.dt_seconds <= 0.0:
            return 0
        return int(math.ceil(d_km / (self.c_fiber * self.dt_seconds)))

    def schedule(self, delay_ticks: int, callback: Callable[[], None]) -> None:
        """Fire ``callback`` after ``delay_ticks`` advances (0 => next advance)."""
        self._sched.schedule(delay_ticks * TICK_NS - _TICK_EPS_NS, callback)

    def advance(self) -> None:
        """Advance one tick; fires any callbacks now due."""
        self.tick += 1
        ns.sim_run(duration=TICK_NS)

    @property
    def n_pending(self) -> int:
        return self._sched.n_pending
