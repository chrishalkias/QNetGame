import numpy as np
import pytest


def test_normalize_n_ch_int_unchanged():
    from rl_stack.agent import QRNAgent
    assert QRNAgent._normalize_n_ch(4) == [4]
    assert QRNAgent._normalize_n_ch(2) == [2]


def test_normalize_n_ch_list_pool():
    from rl_stack.agent import QRNAgent
    assert QRNAgent._normalize_n_ch([2, 3]) == [2, 3]
    assert QRNAgent._normalize_n_ch((2, 3, 4)) == [2, 3, 4]


def test_normalize_n_ch_rejects_bad_input():
    from rl_stack.agent import QRNAgent
    with pytest.raises(ValueError):
        QRNAgent._normalize_n_ch([])        # empty
    assert QRNAgent._normalize_n_ch([1, 2]) == [1, 2]  # n_ch=1 valid (per-side)
    with pytest.raises(ValueError):
        QRNAgent._normalize_n_ch([0, 2])    # n_ch < 1
    with pytest.raises(ValueError):
        QRNAgent._normalize_n_ch([2, 2.5])  # non-int


def test_sample_rate_scalar_is_constant_and_draws_no_rng():
    from rl_stack.agent import QRNAgent
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state
    assert QRNAgent._sample_rate(rng, 0.7) == 0.7
    # scalar must not consume RNG (keeps existing runs reproducible)
    assert rng.bit_generator.state == state_before


def test_sample_rate_range_samples_in_bounds():
    from rl_stack.agent import QRNAgent
    rng = np.random.default_rng(0)
    vals = [QRNAgent._sample_rate(rng, (0.3, 0.9)) for _ in range(500)]
    assert all(0.3 <= v <= 0.9 for v in vals)
    assert min(vals) < 0.45 and max(vals) > 0.75   # actually varies across range


def test_sample_rate_set_is_discrete_choice():
    from rl_stack.agent import QRNAgent
    rng = np.random.default_rng(0)
    grid = {0.3, 0.5, 0.7, 0.9}
    vals = {QRNAgent._sample_rate(rng, grid) for _ in range(200)}
    assert vals <= grid          # only ever lands on grid points
    assert len(vals) >= 3        # genuinely varies across the grid


def _n_exposure(n_range, episodes, n_target, **kw):
    """Expected fraction of episodes the curriculum samples n_target."""
    from rl_stack.agent import QRNAgent
    exp = 0.0
    for ep in range(episodes):
        pool = QRNAgent._curriculum_pool(ep, episodes, n_range, **kw)
        if n_target in pool:
            exp += 1.0 / len(pool)
    return exp / episodes


def test_curriculum_pool_disabled_is_full_range():
    from rl_stack.agent import QRNAgent
    assert sorted(QRNAgent._curriculum_pool(0, 100, [3, 4, 5],
                                            curriculum=False)) == [3, 4, 5]


def test_curriculum_pool_starts_small_ends_full():
    from rl_stack.agent import QRNAgent
    assert QRNAgent._curriculum_pool(0, 100, [3, 4, 5], curriculum=True) == [3]
    assert sorted(QRNAgent._curriculum_pool(99, 100, [3, 4, 5],
                                            curriculum=True)) == [3, 4, 5]


def test_curriculum_largest_n_not_starved():
    # The largest size must get a real share of training, not be crammed into
    # the final fraction (the old schedule gave n_max only ~10%).
    # old schedule gave n_max ~0.10 ([3,4]) and ~0.05 ([4,5,6,7]); the fix lifts
    # both well clear of starvation.
    assert _n_exposure([3, 4], 2000, 4, curriculum=True) > 0.25
    assert _n_exposure([4, 5, 6, 7], 2000, 7, curriculum=True) > 0.12


def _tiny_train(tmp_path, name, save_best, episodes=120):
    import numpy as np
    from rl_stack import QRNAgent
    sd = str(tmp_path / name)
    agent = QRNAgent(rng=np.random.default_rng(0))
    agent.train(episodes=episodes, max_steps=6, n_range=[3], n_ch=2,
                p_gen=0.9, p_swap=0.9, cutoff=5, F0=0.95, channel_loss=0.0,
                curriculum=False,
                topology="chain", save_path=sd,
                save_best=save_best, best_window=20, plot=False)
    return sd


def test_train_save_best_writes_best_and_final(tmp_path):
    import os
    # Needs enough episodes to reach the settled late window (curriculum off ->
    # eps-floor gate at 0.9*episodes; best_window=20 must fit past it).
    sd = _tiny_train(tmp_path, "best", save_best=True, episodes=300)
    assert os.path.isfile(os.path.join(sd, "policy.pth"))        # best
    assert os.path.isfile(os.path.join(sd, "policy_final.pth"))  # final, for ref


def test_train_save_best_false_only_writes_policy(tmp_path):
    import os
    sd = _tiny_train(tmp_path, "nobest", save_best=False, episodes=40)
    assert os.path.isfile(os.path.join(sd, "policy.pth"))
    assert not os.path.isfile(os.path.join(sd, "policy_final.pth"))


def _es_train(tmp_path, name, eval_fn, episodes, patience):
    import numpy as np
    from rl_stack import QRNAgent
    sd = str(tmp_path / name)
    agent = QRNAgent(rng=np.random.default_rng(0))
    metrics = agent.train(
        episodes=episodes, max_steps=6, n_range=[3], n_ch=2, p_gen=0.9, p_swap=0.9,
        cutoff=5, F0=0.95, channel_loss=0.0,        curriculum=False, topology="chain", save_path=sd,
        eval_fn=eval_fn, eval_every=10, eval_patience=patience, eval_mode='min',
        plot=False)
    return sd, metrics


def test_compare_logs_paired_baseline_returns(tmp_path):
    import numpy as np
    from rl_stack import QRNAgent
    agent = QRNAgent(rng=np.random.default_rng(0))
    m = agent.train(episodes=12, max_steps=6, n_range=[4], n_ch=2,
                    p_gen=1.0, p_swap=1.0, cutoff=8, F0=1.0, channel_loss=0.0,
                    curriculum=False,
                    topology="chain", save_path=None,
                    save_best=False, plot=False, compare=True)
    for k in ("cmp_agent", "cmp_swap", "cmp_rand",
              "cmp_agent_steps", "cmp_swap_steps", "cmp_rand_steps",
              "cmp_agent_succ", "cmp_swap_succ", "cmp_rand_succ"):
        assert len(m[k]) == 12, k
    assert all(isinstance(v, float) for v in m["cmp_swap"])
    assert all(isinstance(v, int) for v in m["cmp_swap_steps"])
    assert set(m["cmp_agent_succ"]) <= {0.0, 1.0}


def test_compare_extra_logs_named_baseline(tmp_path):
    import numpy as np
    from rl_stack import QRNAgent
    from rl_stack.env_wrapper import NOOP
    agent = QRNAgent(rng=np.random.default_rng(0))
    # a trivial extra baseline: always NOOP for env.active_node
    noop_fn = lambda env, obs: NOOP
    m = agent.train(episodes=10, max_steps=6, n_range=[4], n_ch=2,
                    p_gen=1.0, p_swap=1.0, cutoff=8, F0=1.0, channel_loss=0.0,
                    curriculum=False,
                    topology="chain", save_path=None,
                    save_best=False, plot=False, compare=True,
                    compare_extra={"optimal": noop_fn})
    for k in ("cmp_optimal", "cmp_optimal_steps", "cmp_optimal_succ"):
        assert len(m[k]) == 10, k


def test_no_compare_leaves_cmp_metrics_empty(tmp_path):
    import numpy as np
    from rl_stack import QRNAgent
    agent = QRNAgent(rng=np.random.default_rng(0))
    m = agent.train(episodes=8, max_steps=6, n_range=[4], n_ch=2,
                    p_gen=1.0, p_swap=1.0, cutoff=8, F0=1.0, channel_loss=0.0,
                    curriculum=False,
                    topology="chain", save_path=None,
                    save_best=False, plot=False)
    assert m["cmp_agent"] == [] and m["cmp_swap"] == [] and m["cmp_rand"] == []
    assert m["cmp_agent_steps"] == [] and m["cmp_agent_succ"] == []


def test_early_stopping_stops_on_no_improvement(tmp_path):
    calls = {"n": 0}
    def bad_eval(agent):
        calls["n"] += 1
        return 5.0  # constant -> only the first probe "improves"
    sd, metrics = _es_train(tmp_path, "es", bad_eval, episodes=1000, patience=3)
    assert len(metrics["reward"]) < 1000      # stopped early
    assert calls["n"] <= 5                     # ~4 probes (set best, 3 stale) then stop
    assert len(metrics["eval"]) >= 1


def test_early_stopping_improving_runs_full_and_saves_best(tmp_path):
    import os
    seq = {"i": 0}
    def improving_eval(agent):
        seq["i"] += 1
        return 10.0 - seq["i"]  # strictly decreasing -> always improves
    sd, metrics = _es_train(tmp_path, "imp", improving_eval, episodes=60, patience=3)
    assert len(metrics["reward"]) == 60        # never early-stops
    assert os.path.isfile(os.path.join(sd, "policy.pth"))         # best (by eval)
    assert os.path.isfile(os.path.join(sd, "policy_final.pth"))   # final


def test_disable_actions_trains(tmp_path):
    import os
    import numpy as np
    from rl_stack import QRNAgent
    from rl_stack.env_wrapper import PURIFY
    sd = str(tmp_path / "da")
    agent = QRNAgent(rng=np.random.default_rng(0))
    agent.train(episodes=40, max_steps=6, n_range=[3], n_ch=2, p_gen=0.9, p_swap=0.9,
                cutoff=5, F0=0.95, channel_loss=0.0,                curriculum=False, topology="chain", save_path=sd,
                disable_actions=(PURIFY,), save_best=False, plot=False)
    assert os.path.isfile(os.path.join(sd, "policy.pth"))


def test_train_records_ticks_not_env_steps_on_mid_sweep_delivery(monkeypatch):
    """A mid-sweep delivery closes the episode with info["ticks"] = env.steps
    + 1 (the tick boundary hasn't been crossed yet). Force this on the very
    first micro-step of the one episode trained and confirm train() logs the
    larger info["ticks"] value, not the smaller (here: unchanged) env.steps
    -- guards the fix in agent.py that reads info["ticks"] instead of
    env.steps at the three delivery-time recording sites."""
    import numpy as np
    from rl_stack import QRNAgent
    from rl_stack.env_wrapper import QRNEnv

    orig_step = QRNEnv.step
    call_count = {"n": 0}

    def forced_step(self, action):
        # Capture steps BEFORE calling through: for N=3 there is only one
        # interior node, so its micro-step is ALSO the tick boundary and
        # orig_step may increment self.steps internally as part of resolving
        # it -- reading self.steps after the call would then race against
        # that increment (flaky). The pre-call value is always 0 here, which
        # is what "env.steps has NOT advanced yet" means.
        steps_before = self.steps
        obs, reward, done, info = orig_step(self, action)
        if call_count["n"] == 0:
            call_count["n"] += 1
            info = dict(info)
            info["ticks"] = steps_before + 1   # env.steps had NOT advanced yet
            info["terminated"] = True
            info["fidelity"] = 0.9
            return obs, reward, True, info
        return obs, reward, done, info

    monkeypatch.setattr(QRNEnv, "step", forced_step)

    agent = QRNAgent(rng=np.random.default_rng(0))
    metrics = agent.train(episodes=1, max_steps=6, n_range=[3], n_ch=2,
                           p_gen=0.9, p_swap=0.9, cutoff=5, F0=0.95,
                           channel_loss=0.0, curriculum=False, topology="chain",
                           save_path=None, save_best=False, plot=False)

    assert metrics["steps"][0] == 1   # info["ticks"]: correct mid-sweep count
    # env.steps itself never advanced past the tick boundary this episode,
    # so recording it directly would have (wrongly) logged 0.


def _repro_agent(seed):
    """A fully seed-determined agent: torch net init + agent RNG + replay
    sampler all pinned to `seed`."""
    import numpy as np
    import torch
    from rl_stack import QRNAgent
    torch.manual_seed(seed)
    return QRNAgent(rng=np.random.default_rng(seed), seed=seed)


def _repro_run(seed):
    """A tiny, fully seeded train() run (env physics seeded via env_seed).
    Returns the per-episode (reward, steps) metrics."""
    agent = _repro_agent(seed)
    m = agent.train(episodes=12, max_steps=6, n_range=[4], n_ch=2,
                    p_gen=0.7, p_swap=0.7, cutoff=5, F0=0.95, channel_loss=0.0,
                    curriculum=False, topology="chain",
                    save_path=None, save_best=False, plot=False, env_seed=seed)
    return m["reward"], m["steps"]


def test_same_seed_bit_reproduces_training_metrics():
    # Two runs with the SAME master seed must produce identical per-episode
    # reward + step sequences (env physics is now seeded via env_seed).
    r1, s1 = _repro_run(123)
    r2, s2 = _repro_run(123)
    assert r1 == r2, "same-seed reward sequences must be identical"
    assert s1 == s2, "same-seed step sequences must be identical"


def test_different_seed_changes_training_metrics():
    # A different master seed must change the trajectory (else nothing is
    # actually seeded and reproducibility would be vacuous).
    r1, _ = _repro_run(123)
    r2, _ = _repro_run(456)
    assert r1 != r2, "different seeds must yield different reward sequences"


def test_run_manifest_written_with_seed_and_commit(tmp_path):
    import argparse
    import json
    import os
    from experiments.training.train import write_run_manifest
    args = argparse.Namespace(
        run_id="mani", seed=7, lr=5e-4, hidden=64, episodes=10,
        n_lo=4, n_hi=12, p_gen=[0.4, 0.9], p_swap=[0.4, 0.9])
    sd = str(tmp_path / "mani")
    os.makedirs(sd, exist_ok=True)
    write_run_manifest(sd, args)
    path = os.path.join(sd, "run_config.json")
    assert os.path.isfile(path)
    with open(path) as f:
        man = json.load(f)
    assert man["args"]["seed"] == 7
    assert "git_commit" in man and man["git_commit"]        # commit or "unknown"
    assert "git_dirty" in man and "timestamp" in man
    assert set(man["versions"]) == {"numpy", "torch", "torch_geometric"}



def _probe_args(**kw):
    import argparse
    base = dict(p_gen=[1.0], p_swap=[1.0], n_lo=3, n_hi=4, n_ch=[2],
                cutoff=50, cutoff_lo=None, cutoff_hi=None,
                p_gen_std=0.0, p_swap_std=0.0, F0=1.0, channel_loss=0.0,
                max_steps=40, topology="chain", seed=0)
    base.update(kw)
    return argparse.Namespace(**base)


class TestCalibratedProbeCells:
    def test_pilot_rate_easy_cell_near_one(self):
        from experiments.training.train import _pilot_delivery_rate
        cell = {"n_repeaters": 3, "n_ch": 2, "p_gen": 1.0, "p_swap": 1.0,
                "cutoff": 50}
        assert _pilot_delivery_rate(cell, _probe_args(), n_episodes=10) > 0.9

    def test_pilot_rate_impossible_cell_zero(self):
        from experiments.training.train import _pilot_delivery_rate
        cell = {"n_repeaters": 8, "n_ch": 2, "p_gen": 0.3, "p_swap": 0.3,
                "cutoff": 3}
        assert _pilot_delivery_rate(cell, _probe_args(max_steps=30),
                                    n_episodes=6) == 0.0

    def test_calibration_returns_two_complete_cells(self):
        from experiments.training.train import make_calibrated_cells
        args = _probe_args(p_gen=[0.4, 0.9], p_swap=[0.4, 0.9],
                           n_lo=4, n_hi=6, cutoff_lo=10, cutoff_hi=20)
        cells = make_calibrated_cells(args, n_episodes=5)
        assert len(cells) == 2
        for c in cells:
            assert set(c) == {"n_repeaters", "n_ch", "p_gen", "p_swap",
                              "cutoff"}

    def test_calibration_deterministic_under_seed(self):
        from experiments.training.train import make_calibrated_cells
        args = _probe_args(p_gen=[0.4, 0.9], p_swap=[0.4, 0.9],
                           n_lo=4, n_hi=6, cutoff_lo=10, cutoff_hi=20)
        assert (make_calibrated_cells(args, n_episodes=5)
                == make_calibrated_cells(args, n_episodes=5))

    def test_dedup_degenerate_ranges_no_duplicate_cell(self, capsys):
        # Fully degenerate ranges (p_gen/p_swap single-valued, n_lo==n_hi,
        # cutoff scalar) collapse every candidate to ONE distinct parameter
        # tuple. Must not silently repeat it to pad out to n_cells=2.
        from experiments.training.train import make_calibrated_cells
        args = _probe_args(p_gen=[1.0], p_swap=[1.0], n_lo=4, n_hi=4,
                           cutoff=50, cutoff_lo=None, cutoff_hi=None)
        cells = make_calibrated_cells(args, n_episodes=5)
        assert len(cells) == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "1 distinct candidate" in captured.out

    def test_calibration_fallback_when_all_candidates_easy(self, capsys):
        # Every candidate is trivially easy (p_gen=p_swap=1.0, generous
        # cutoff, ample max_steps) -> pilot rate ~1.0 for all, none land in
        # the [0.30, 0.70] band -> closest-rate fallback must fire.
        from experiments.training.train import make_calibrated_cells
        args = _probe_args(p_gen=[1.0], p_swap=[1.0], n_lo=3, n_hi=6,
                           cutoff_lo=50, cutoff_hi=100, max_steps=60)
        cells = make_calibrated_cells(args, n_episodes=5)
        assert len(cells) == 2
        captured = capsys.readouterr()
        assert "falling back to closest rates" in captured.out

        cells_again = make_calibrated_cells(args, n_episodes=5)
        assert cells == cells_again
