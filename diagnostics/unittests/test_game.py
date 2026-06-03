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
