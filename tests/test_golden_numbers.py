"""Numerical regression gate for behaviour-neutral refactors.

Most of the 2026-07-26 cleanup CLAIMS to change nothing observable (deletions,
renames, file moves). A green unit-test suite does not prove that: no unit test
pins a delivery time, a reward, or an observation column. This file does.

Two tests share one JSON file.

`test_golden_delivery_times` runs the two HEURISTICS over a grid of cells and
pins EVERY scalar statistic mc_eval reports (T, T_std, conn_rate, mean_F_conn),
not just T. They are pure functions of the engine plus the RNG path, so drift
means the refactor changed physics or seeding, which is exactly what must not
happen. mean_F_conn carries disproportionate weight: delivery in this project is
TOPOLOGICAL (a link exists), so the whole fidelity pipeline
(werner_to_fidelity / fidelity_to_werner, the depolarizing floor, the BBPSSW
F_new formula) reaches T only through two discrete branches, and a refactor that
damaged fidelity without flipping those branches would otherwise pass green.
mean_F_conn is also the only pinned statistic that is a genuine continuous
float, so it is what restores sensitivity to float-level drift.

`test_golden_rollout_signature` runs seeded swap_asap rollouts DIRECTLY against
QRNEnv, bypassing mc_eval, and pins the summed per-step reward plus a SHA-256
over the raw float32 / int64 bytes of every observation. mc_eval discards the
reward and the heuristics never read obs, so without this second test the PBRS
potential, STEP_COST, SUCCESS_REWARD, gamma_eff, observation features
0/3/4/5/6/7 and edge_index are all free to move under a green gate.

Equality is EXACT, not approximate. What that actually buys, precisely: the
`times` list holds integer tick counts, so T is exactly k/n_episodes and its
smallest possible non-zero move is 1/n_episodes (0.005 to 0.01 for these cells).
This gate therefore fires when at least one seeded episode takes a DIFFERENT
NUMBER OF TICKS, or when any other pinned statistic moves. Continuous drift is
caught by mean_F_conn and by the rollout rewards / observation digest, not by T.
It is not, and cannot be, a 1e-12 detector on T alone.

What this gate does NOT cover:
  - mc_eval hardcodes F0=1.0 and channel_loss=0.0, so the distance-dependent
    generation branch and the depolarizing-loss floor are never exercised. A
    refactor touching `channel_loss` or `F0` handling can be green here.
  - No agent checkpoint is involved, by design, so nothing about the learned
    policy is pinned. Parking, retraining or deleting an agent never turns this
    red, and a policy-network change never turns it red either.
  - Only the chain topology, only swap_asap and purify_then_swap, and only the
    cells listed below. Green means "these numbers did not move", never
    "nothing changed".

Regenerate ONLY when a change to physics is intended and reviewed. A reason is
mandatory and lands in the file's `_meta` block:

    PYTHONPATH=src:. python tests/test_golden_numbers.py \
        --regenerate --reason "why this was allowed to move"
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_numbers.json")

# Top-level JSON keys that are NOT cells. Both are excluded from comparison by
# test_golden_delivery_times; the rollout block has its own test.
META_KEY = "_meta"
ROLLOUT_KEY = "_rollout_signature"

# (N, n_ch, p_gen, p_swap, cutoff, horizon, episodes, p_gen_std, p_swap_std)
CELLS = (
    (6,  2, 0.6, 0.8,  7, 120, 200, 0.0,  0.0),   # tight memory: gate bites hardest
    (6,  2, 0.6, 0.8, 20, 120, 200, 0.0,  0.0),   # roomy memory
    (10, 2, 0.6, 0.8, 10, 200, 150, 0.0,  0.0),   # the +21.7% cell from A2b
    (10, 3, 0.6, 0.8, 20, 200, 150, 0.0,  0.0),   # wider memory, longer chain
    (12, 2, 0.4, 0.8, 15, 300, 100, 0.0,  0.0),   # the paper's headline rate regime
    (4,  1, 0.9, 0.9, 30, 100, 200, 0.0,  0.0),   # n_ch=1: purify can never fire
    (8,  2, 0.6, 0.8, 20, 200, 150, 0.15, 0.15),  # per-repeater inhomogeneity
)

# (N, n_ch, p_gen, p_swap, cutoff, horizon, episodes, seed) for the signature test.
# Deliberately tiny: this is a fingerprint, not a statistic.
ROLLOUTS = (
    (5, 2, 0.6, 0.8, 15, 60, 4, 20260726),
    (8, 2, 0.5, 0.7, 20, 80, 3, 13579),
    (4, 1, 0.9, 0.9, 30, 40, 3, 2468),
)


def _key(cell):
    return ("N{}_nch{}_pg{}_ps{}_cut{}_H{}_eps{}"
            "_pgstd{}_psstd{}").format(*cell)


def _rollout_key(cfg):
    return "N{}_nch{}_pg{}_ps{}_cut{}_H{}_eps{}_seed{}".format(*cfg)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _measure_cells():
    from experiments.mc_eval import mc_eval
    from rl_stack import policies
    fns = {"swap_asap":   lambda env, obs: policies.swap_asap(env),
           "purify_swap": lambda env, obs: policies.purify_then_swap(env)}
    out = {}
    for cell in CELLS:
        N, n_ch, pg, ps, cut, H, eps, pg_std, ps_std = cell
        out[_key(cell)] = {
            name: mc_eval(fn, N, n_ch, pg, ps, cut, H, eps,
                          p_gen_std=pg_std, p_swap_std=ps_std,
                          return_stats=True)
            for name, fn in fns.items()
        }
    return out


def _feed(digest, obs):
    """Fold one observation into the running hash, byte-exactly.

    Raw float32 / int64 bytes, never repr: the digest must be an exact
    fingerprint of the arrays, not of their formatting.
    """
    digest.update(np.ascontiguousarray(obs["x"], dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(obs["edge_index"], dtype=np.int64).tobytes())


def _measure_rollouts():
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack import policies
    out = {}
    for cfg in ROLLOUTS:
        N, n_ch, pg, ps, cut, H, eps, seed = cfg
        rng = np.random.default_rng(seed)
        digest = hashlib.sha256()
        reward_sums = []
        for _ in range(eps):
            env = QRNEnv(N, n_ch=n_ch, p_gen=pg, p_swap=ps, cutoff=cut,
                         F0=1.0, channel_loss=0.0, max_steps=H,
                         rng=np.random.default_rng(rng.integers(2**32)))
            obs = env.reset()
            _feed(digest, obs)
            total = 0.0
            while not env.done and env.steps < H:
                obs, reward, done, _ = env.step(policies.swap_asap(env))
                total += float(reward)
                _feed(digest, obs)
                if done:
                    break
            reward_sums.append(total)
        out[_rollout_key(cfg)] = {"reward_sums": reward_sums,
                                  "obs_sha256": digest.hexdigest()}
    return out


# ---------------------------------------------------------------------------
# file io
# ---------------------------------------------------------------------------

def _load():
    with open(GOLDEN) as fh:
        return json.load(fh)


def _cell_keys(payload):
    """Comparable cell keys: everything that is not a reserved `_` block."""
    return {k for k in payload if not k.startswith("_")}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

_MISSING = (
    "golden_numbers.json missing; generate it with `PYTHONPATH=src:. "
    'python tests/test_golden_numbers.py --regenerate --reason "..."`')


@pytest.mark.slow
def test_golden_delivery_times():
    assert os.path.exists(GOLDEN), _MISSING
    stored = _load()
    expected = {k: stored[k] for k in _cell_keys(stored)}
    actual = _measure_cells()

    only_golden = sorted(set(expected) - set(actual))
    only_actual = sorted(set(actual) - set(expected))
    assert not (only_golden or only_actual), (
        "the pinned cell set and the measured cell set differ:\n"
        + "".join(f"  only in golden_numbers.json: {k}\n" for k in only_golden)
        + "".join(f"  only in CELLS:               {k}\n" for k in only_actual))

    bad = []
    for cell in sorted(expected):
        e_pols, a_pols = set(expected[cell]), set(actual[cell])
        if e_pols != a_pols:
            bad.append(f"  {cell}: policy set differs, golden={sorted(e_pols)} "
                       f"actual={sorted(a_pols)}")
            continue
        for pol in sorted(e_pols):
            e_stats, a_stats = set(expected[cell][pol]), set(actual[cell][pol])
            if e_stats != a_stats:
                bad.append(f"  {cell} {pol}: statistic set differs, "
                           f"golden={sorted(e_stats)} actual={sorted(a_stats)}")
                continue
            for stat in sorted(e_stats):
                e, a = expected[cell][pol][stat], actual[cell][pol][stat]
                if e != a:
                    bad.append(f"  {cell} {pol} {stat}: "
                               f"golden={e!r} actual={a!r}")
    assert not bad, (
        "pinned statistics moved; the refactor was NOT behaviour-neutral:\n"
        + "\n".join(bad))


@pytest.mark.slow
def test_golden_rollout_signature():
    """Pin the reward stream and the observation layout, which mc_eval cannot see.

    A red result here means the REWARD FUNCTION, the PBRS POTENTIAL, or the
    OBSERVATION LAYOUT changed: STEP_COST / SUCCESS_REWARD / FAILED_ACTION, the
    per-transition gamma_eff, `potential.path_progress`, the eight node feature
    columns (order, definition or normalization), or edge_index.

    For a task that CLAIMS behaviour-neutrality, that is a FINDING, not a reason
    to regenerate. Regenerate only once the change is intended and reviewed.
    """
    assert os.path.exists(GOLDEN), _MISSING
    stored = _load()
    assert ROLLOUT_KEY in stored, (
        f"{ROLLOUT_KEY} block missing from golden_numbers.json; " + _MISSING)
    expected = stored[ROLLOUT_KEY]
    actual = _measure_rollouts()

    only_golden = sorted(set(expected) - set(actual))
    only_actual = sorted(set(actual) - set(expected))
    assert not (only_golden or only_actual), (
        "the pinned rollout set and the measured rollout set differ:\n"
        + "".join(f"  only in golden_numbers.json: {k}\n" for k in only_golden)
        + "".join(f"  only in ROLLOUTS:            {k}\n" for k in only_actual))

    bad = []
    for cfg in sorted(expected):
        e_fields, a_fields = set(expected[cfg]), set(actual[cfg])
        if e_fields != a_fields:
            bad.append(f"  {cfg}: field set differs, golden={sorted(e_fields)} "
                       f"actual={sorted(a_fields)}")
            continue
        for field in sorted(e_fields):
            e, a = expected[cfg][field], actual[cfg][field]
            if e != a:
                bad.append(f"  {cfg} {field}: golden={e!r} actual={a!r}")
    assert not bad, (
        "the reward stream or the observation stream moved; the reward "
        "function, the PBRS potential or the observation layout changed:\n"
        + "\n".join(bad))


# ---------------------------------------------------------------------------
# regeneration (provenance + friction, never silent)
# ---------------------------------------------------------------------------

def _flatten(payload):
    """{'block|entry|field': scalar} over everything the tests compare."""
    flat = {}
    for key, value in payload.items():
        if key == META_KEY:
            continue
        if key == ROLLOUT_KEY:
            for cfg, sig in value.items():
                flat[f"{ROLLOUT_KEY}|{cfg}|obs_sha256"] = sig["obs_sha256"]
                for i, r in enumerate(sig["reward_sums"]):
                    flat[f"{ROLLOUT_KEY}|{cfg}|reward_sums[{i}]"] = r
            continue
        for pol, stats in value.items():
            if not isinstance(stats, dict):
                # older single-statistic schema: cell -> policy -> T
                flat[f"{key}|{pol}|T"] = stats
                continue
            for stat, val in stats.items():
                flat[f"{key}|{pol}|{stat}"] = val
    return flat


def _print_delta_table(old, new):
    """Old-to-new delta table, printed BEFORE anything is written."""
    if old is None:
        print("no existing golden_numbers.json; this is a first generation.")
        return
    a, b = _flatten(old), _flatten(new)
    removed, added = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    moved = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]

    print(f"\ndelta vs the file on disk: {len(moved)} moved, "
          f"{len(added)} added, {len(removed)} removed "
          f"(of {len(a)} pinned values)")
    print("-" * 78)
    for k in removed:
        print(f"  REMOVED  {k}: {a[k]!r}")
    for k in added:
        print(f"  ADDED    {k}: {b[k]!r}")
    for k in moved:
        old_v, new_v = a[k], b[k]
        try:
            delta = f"  (delta {float(new_v) - float(old_v):+.6g})"
        except (TypeError, ValueError):
            delta = ""
        print(f"  MOVED    {k}: {old_v!r} -> {new_v!r}{delta}")
    if not (removed or added or moved):
        print("  (no pinned value changed)")
    print("-" * 78)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _meta(reason):
    return {
        "reason": reason,
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "note": ("Diagnostic provenance only; excluded from every comparison. "
                 "A red gate on a different numpy or platform than the one "
                 "recorded here may be a Generator-stream or float difference "
                 "rather than a defect in the refactor."),
    }


def _regenerate(reason):
    old = _load() if os.path.exists(GOLDEN) else None
    payload = {META_KEY: _meta(reason)}
    payload.update(_measure_cells())
    payload[ROLLOUT_KEY] = _measure_rollouts()
    _print_delta_table(old, payload)
    with open(GOLDEN, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {GOLDEN}\nreason: {reason}")


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        print(__doc__)
        sys.exit(0)
    reason = ""
    if "--reason" in sys.argv:
        i = sys.argv.index("--reason")
        if i + 1 < len(sys.argv):
            reason = sys.argv[i + 1].strip()
    if not reason:
        sys.exit(
            "REFUSED: --regenerate overwrites a numerical gate that certifies\n"
            "refactors as behaviour-neutral. Regenerating hides exactly the\n"
            "evidence the gate exists to produce, so it needs a written reason\n"
            "that lands in the file's _meta block and shows up in review:\n\n"
            '    PYTHONPATH=src:. python tests/test_golden_numbers.py \\\n'
            '        --regenerate --reason "why these numbers were allowed to move"\n')
    _regenerate(reason)
