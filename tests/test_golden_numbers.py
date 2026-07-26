"""Numerical regression gate for behaviour-neutral refactors.

Most of the 2026-07-26 cleanup CLAIMS to change nothing observable (deletions,
renames, file moves). A green unit-test suite does not prove that: no unit test
pins a delivery time. This file does.

The cells below are evaluated with the two HEURISTICS only. They are pure
functions of the engine plus the RNG path, so any drift means the refactor
changed physics or seeding, which is exactly what must not happen. No agent
checkpoint is involved, so parking or retraining an agent never turns this red.

Equality is EXACT, not approximate. A refactor that shifts T by 1e-12 has
changed an RNG path or a float operation order, and that is a finding, not a
rounding artifact.

Regenerate ONLY when a change to physics is intended and reviewed:
    PYTHONPATH=src:. python tests/test_golden_numbers.py --regenerate
"""
from __future__ import annotations
import json
import os

import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_numbers.json")

# (N, n_ch, p_gen, p_swap, cutoff, horizon, episodes)
CELLS = [
    (6,  2, 0.6, 0.8,  7, 120, 200),   # tight memory: where the gate bites hardest
    (6,  2, 0.6, 0.8, 20, 120, 200),   # roomy memory
    (10, 2, 0.6, 0.8, 10, 200, 150),   # the +21.7% cell from the A2b measurement
    (10, 3, 0.6, 0.8, 20, 200, 150),   # wider memory, longer chain
    (12, 2, 0.4, 0.8, 15, 300, 100),   # the paper's headline rate regime
    (4,  1, 0.9, 0.9, 30, 100, 200),   # n_ch=1: purify can never fire
]


def _key(cell):
    return "N{}_nch{}_pg{}_ps{}_cut{}_H{}_eps{}".format(*cell)


def _measure():
    from experiments.mc_eval import mc_eval
    # Task A7 merges strategies.py and winnability.py into rl_stack.policies;
    # this import becomes `from rl_stack import policies` at that point.
    from rl_stack import strategies
    fns = {"swap_asap":   lambda env, obs: strategies.swap_asap(env),
           "purify_swap": lambda env, obs: strategies.purify_then_swap(env)}
    out = {}
    for cell in CELLS:
        N, n_ch, pg, ps, cut, H, eps = cell
        out[_key(cell)] = {
            name: mc_eval(fn, N, n_ch, pg, ps, cut, H, eps)[0]
            for name, fn in fns.items()
        }
    return out


@pytest.mark.slow
def test_golden_delivery_times():
    assert os.path.exists(GOLDEN), (
        "golden_numbers.json missing; generate it with "
        "`PYTHONPATH=src:. python tests/test_golden_numbers.py --regenerate`")
    expected = json.load(open(GOLDEN))
    actual = _measure()
    assert set(actual) == set(expected), "cell set changed; regenerate deliberately"
    bad = []
    for k in sorted(expected):
        for pol in sorted(expected[k]):
            e, a = expected[k][pol], actual[k][pol]
            if e != a:
                bad.append(f"  {k} {pol}: golden={e!r} actual={a!r} (delta {a - e:+.6g})")
    assert not bad, (
        "delivery times moved; the refactor was NOT behaviour-neutral:\n"
        + "\n".join(bad))


if __name__ == "__main__":
    import sys
    if "--regenerate" in sys.argv:
        json.dump(_measure(), open(GOLDEN, "w"), indent=2)
        print(f"wrote {GOLDEN}")
    else:
        print(__doc__)
