import math
import numpy as np
import pytest


def test_phase1_config_invariants():
    from game.phases import PHASE1, PhaseConfig
    assert isinstance(PHASE1, PhaseConfig)
    assert PHASE1.name == "phase1"
    assert PHASE1.topology == "chain"
    assert tuple(PHASE1.n_range) == (4, 5)
    assert tuple(PHASE1.n_ch) == (2, 3)
    # all n_ch values are ints >= 2 (exact-optimal comparison needs n_ch=2 present)
    assert all(isinstance(c, int) and c >= 2 for c in PHASE1.n_ch)
    assert 2 in PHASE1.n_ch
    # cutoff must match the optimal-policy pickle naming the comparator loads
    assert PHASE1.cutoff == 5
    assert PHASE1.p_gen == 0.9 and PHASE1.p_swap == 0.9
    assert PHASE1.backend == "legacy"
    assert PHASE1.dt_seconds == 0.0


def test_phaseconfig_is_frozen():
    from game.phases import PHASE1
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        PHASE1.episodes = 1  # type: ignore[misc]


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
    with pytest.raises(ValueError):
        QRNAgent._normalize_n_ch([1, 2])    # n_ch < 2
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


def _tiny_train(tmp_path, name, save_best, episodes=120):
    import numpy as np
    from rl_stack import QRNAgent
    sd = str(tmp_path / name)
    agent = QRNAgent(rng=np.random.default_rng(0))
    agent.train(episodes=episodes, max_steps=6, n_range=[3], n_ch=2,
                p_gen=0.9, p_swap=0.9, cutoff=5, F0=0.95, channel_loss=0.0,
                dt_seconds=0.0, heterogeneous=False, curriculum=False,
                topology="chain", backend="legacy", save_path=sd,
                save_best=save_best, best_window=20, plot=False)
    return sd


def test_train_save_best_writes_best_and_final(tmp_path):
    import os
    sd = _tiny_train(tmp_path, "best", save_best=True)
    assert os.path.isfile(os.path.join(sd, "policy.pth"))        # best
    assert os.path.isfile(os.path.join(sd, "policy_final.pth"))  # final, for ref


def test_train_save_best_false_only_writes_policy(tmp_path):
    import os
    sd = _tiny_train(tmp_path, "nobest", save_best=False, episodes=40)
    assert os.path.isfile(os.path.join(sd, "policy.pth"))
    assert not os.path.isfile(os.path.join(sd, "policy_final.pth"))


def test_run_phase_trains_and_saves(tmp_path):
    import dataclasses
    import numpy as np
    from rl_stack import QRNAgent
    from game.phases import PHASE1
    from game.runner import run_phase

    tiny = dataclasses.replace(PHASE1, episodes=12, max_steps=8)
    agent = QRNAgent(rng=np.random.default_rng(0))
    save_dir = tmp_path / "phase1"
    metrics = run_phase(agent, tiny, str(save_dir), plot=False)

    assert (save_dir / "policy.pth").is_file()
    assert set(metrics.keys()) >= {"reward", "loss", "steps", "success"}
    assert len(metrics["reward"]) == 12


def test_gaps_computes_percentages():
    from game.report import gaps
    row = gaps(N=4, in_distribution=True, T_opt_swaponly=10.0, T_swap=12.0,
               T_agent=11.0, T_agent_swaponly=10.5)
    assert row["N"] == 4
    assert row["in_distribution"] is True
    assert row["T_opt_swaponly"] == 10.0
    assert row["gap_full_pct"] == pytest.approx(10.0)            # (11-10)/10
    assert row["scheduling_gap_pct"] == pytest.approx(5.0)       # (10.5-10)/10
    assert row["agent_vs_swap_pct"] == pytest.approx(100 * (12 - 11) / 12)


def test_gaps_handles_missing_optimal():
    from game.report import gaps
    row = gaps(N=5, in_distribution=False, T_opt_swaponly=None, T_swap=20.0,
               T_agent=19.0, T_agent_swaponly=19.5)
    assert math.isnan(row["gap_full_pct"])
    assert math.isnan(row["scheduling_gap_pct"])
    assert row["agent_vs_swap_pct"] == pytest.approx(5.0)


def test_gaps_without_swaponly_agent():
    from game.report import gaps
    row = gaps(N=4, in_distribution=True, T_opt_swaponly=10.0, T_swap=12.0,
               T_agent=11.0)  # T_agent_swaponly defaults to None
    assert row["T_agent_swaponly"] is None
    assert math.isnan(row["scheduling_gap_pct"])
    assert row["gap_full_pct"] == pytest.approx(10.0)


def test_format_report_has_columns():
    from game.report import format_report
    report = {
        "config": {"n_ch": 2, "cutoff": 5, "p_gen": 0.9, "p_swap": 0.9, "horizon": 30},
        "rows": [
            {"N": 3, "in_distribution": False, "T_opt_swaponly": 7.1, "T_swap": 7.1,
             "T_agent": 7.3, "T_agent_swaponly": 7.15, "gap_full_pct": 2.8,
             "scheduling_gap_pct": 0.7, "agent_vs_swap_pct": -2.8},
            {"N": 4, "in_distribution": True, "T_opt_swaponly": 19.9, "T_swap": 20.9,
             "T_agent": 20.1, "T_agent_swaponly": 20.5, "gap_full_pct": 1.0,
             "scheduling_gap_pct": 3.0, "agent_vs_swap_pct": 3.8},
        ],
    }
    text = format_report(report)
    assert "T_opt" in text and "gap_full" in text and "sched_gap" in text
    assert "ag_vs_swap" in text
    assert "N=3" in text or " 3 " in text


def _write_synthetic_pickle(policy_dir, N, n_ch, cutoff, horizon, pg, ps):
    """A trivial all-NOOP optimal policy: single action [0]*N, empty policy map
    (every lookup falls back to index 0 = all-NOOP)."""
    import os, pickle
    os.makedirs(policy_dir, exist_ok=True)
    fname = (f"optimal_policy_N{N}_ch{n_ch}_co{cutoff}_h{horizon}"
             f"_pg{pg:.2f}_ps{ps:.2f}.pkl")
    payload = {
        "config": dict(N=N, n_ch=n_ch, cutoff=cutoff, horizon=horizon,
                       p_gen=pg, p_swap=ps),
        "acts": [[0] * N],
        "policy": {},
    }
    with open(os.path.join(policy_dir, fname), "wb") as f:
        pickle.dump(payload, f)


def test_load_optimal_pickle_match_and_mismatch(tmp_path):
    from game.compare_optimal import load_optimal_pickle
    _write_synthetic_pickle(str(tmp_path), 4, 2, 5, 30, 0.9, 0.9)
    # exact match loads
    payload = load_optimal_pickle(str(tmp_path), N=4, n_ch=2, cutoff=5,
                                  horizon=30, p_gen=0.9, p_swap=0.9)
    assert payload is not None and payload["config"]["N"] == 4
    # absent file -> None
    assert load_optimal_pickle(str(tmp_path), N=3, n_ch=2, cutoff=5,
                               horizon=30, p_gen=0.9, p_swap=0.9) is None


def test_compare_to_optimal_with_injected_agent(tmp_path):
    # Use swap_asap as a stand-in "agent_fn" so the test needs no torch checkpoint.
    import sys, os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)  # diagnostics/unittests -> repo root
    sys.path.insert(0, os.path.join(repo_root, "train-test"))
    import optimal_baseline as ob
    from game.phases import PHASE1
    from game.compare_optimal import compare_to_optimal

    _write_synthetic_pickle(str(tmp_path), 3, 2, 5, 30, 0.9, 0.9)
    _write_synthetic_pickle(str(tmp_path), 4, 2, 5, 30, 0.9, 0.9)

    report = compare_to_optimal(
        ckpt=None, cfg=PHASE1, policy_dir=str(tmp_path),
        mc_eps=200, horizon=30, compare_N=(3, 4),
        agent_fn=ob.swap_asap_fn, agent_fn_swaponly=ob.swap_asap_fn,
    )
    assert report["config"]["n_ch"] == 2
    assert len(report["rows"]) == 2
    for r in report["rows"]:
        assert "T_agent" in r and "T_swap" in r and "T_opt_swaponly" in r
        assert "T_agent_swaponly" in r
        assert math.isfinite(r["T_agent"])


def test_run_phase1_main_end_to_end(tmp_path):
    """Tiny end-to-end: train a few episodes, run comparison against synthetic
    pickles, write checkpoint + optimal_comparison.json."""
    import os, json
    save_dir = tmp_path / "phase1"
    policy_dir = tmp_path / "policies"
    _write_synthetic_pickle(str(policy_dir), 3, 2, 5, 30, 0.9, 0.9)
    _write_synthetic_pickle(str(policy_dir), 4, 2, 5, 30, 0.9, 0.9)

    from game.run_phase1 import main
    main([
        "--episodes", "12",
        "--max_steps", "8",
        "--save_dir", str(save_dir),
        "--policy_dir", str(policy_dir),
        "--mc_eps", "100",
        "--seed", "0",
    ])

    assert (save_dir / "policy.pth").is_file()
    out = save_dir / "optimal_comparison.json"
    assert out.is_file()
    report = json.loads(out.read_text())
    assert report["config"]["n_ch"] == 2
    assert len(report["rows"]) == 2


def test_run_phase1_skip_compare(tmp_path):
    from game.run_phase1 import main
    save_dir = tmp_path / "p1"
    main(["--episodes", "8", "--max_steps", "6",
          "--save_dir", str(save_dir), "--skip_compare"])
    assert (save_dir / "policy.pth").is_file()
    assert not (save_dir / "optimal_comparison.json").exists()


def test_load_optimal_pickle_config_mismatch_raises(tmp_path):
    import os, pickle, pytest
    from game.compare_optimal import load_optimal_pickle
    # Write a pickle whose filename says N=4 but whose stored config says N=99.
    fname = "optimal_policy_N4_ch2_co5_h30_pg0.90_ps0.90.pkl"
    payload = {"config": dict(N=99, n_ch=2, cutoff=5, horizon=30, p_gen=0.9, p_swap=0.9),
               "acts": [[0, 0, 0, 0]], "policy": {}}
    with open(os.path.join(str(tmp_path), fname), "wb") as f:
        pickle.dump(payload, f)
    with pytest.raises(ValueError):
        load_optimal_pickle(str(tmp_path), N=4, n_ch=2, cutoff=5,
                            horizon=30, p_gen=0.9, p_swap=0.9)


def test_compare_to_optimal_degrades_without_pickle(tmp_path):
    import sys, os, math
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    sys.path.append(os.path.join(repo_root, "train-test"))
    import optimal_baseline as ob
    from game.phases import PHASE1
    from game.compare_optimal import compare_to_optimal
    # Empty policy_dir -> no pickles -> swap-asap-only rows with NaN optimal gap.
    report = compare_to_optimal(
        ckpt=None, cfg=PHASE1, policy_dir=str(tmp_path),
        mc_eps=100, horizon=30, compare_N=(3,),
        agent_fn=ob.swap_asap_fn, agent_fn_swaponly=ob.swap_asap_fn,
    )
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["T_opt_swaponly"] is None
    assert math.isnan(row["gap_full_pct"])
    assert math.isfinite(row["agent_vs_swap_pct"])
