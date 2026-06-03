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
    row = gaps(N=4, in_distribution=True, T_opt=10.0, T_swap=12.0, T_agent=11.0)
    assert row["N"] == 4
    assert row["in_distribution"] is True
    assert row["gap_to_optimal_pct"] == pytest.approx(10.0)   # (11-10)/10
    assert row["agent_vs_swap_pct"] == pytest.approx(100 * (12 - 11) / 12)


def test_gaps_handles_missing_optimal():
    from game.report import gaps
    row = gaps(N=5, in_distribution=False, T_opt=None, T_swap=20.0, T_agent=19.0)
    assert math.isnan(row["gap_to_optimal_pct"])
    assert row["agent_vs_swap_pct"] == pytest.approx(5.0)


def test_format_report_has_columns():
    from game.report import format_report
    report = {
        "config": {"n_ch": 2, "cutoff": 5, "p_gen": 0.9, "p_swap": 0.9, "horizon": 30},
        "rows": [
            {"N": 3, "in_distribution": False, "T_opt": 7.1, "T_swap": 7.1,
             "T_agent": 7.3, "gap_to_optimal_pct": 2.8, "agent_vs_swap_pct": -2.8},
            {"N": 4, "in_distribution": True, "T_opt": 19.9, "T_swap": 20.9,
             "T_agent": 20.1, "gap_to_optimal_pct": 1.0, "agent_vs_swap_pct": 3.8},
        ],
    }
    text = format_report(report)
    assert "T_opt" in text and "gap_to_optimal" in text and "agent_vs_swap" in text
    assert "N=3" in text or " 3 " in text
