"""One evaluator: every comparisons entry point must route through mc_eval, so
a figure cannot silently disagree with another figure about what T means.

Note on provenance: the plan this test was written from (B1, Step 1) named the
target functions `_common.eval_stats` and a `mc_eval(..., return_stats=True)`
that also carried `se`/`seF_conn`. The consolidation that actually shipped
(commit ee9374f) took a slightly different shape: `_common.eval_stats` /
`eval_T` / `eval_T_and_F` were deleted outright rather than kept as a thin
wrapper, and the richer stats projection lives as a standalone function,
`mc_eval.mc_eval_stats`, called directly by every comparisons script. This
test is adapted to that real API (see experiments/mc_eval.py's module
docstring and experiments/comparisons/_common.py's), but keeps the same
intent: pin that the two remaining eval entry points -- `mc_eval` with
`return_stats=True` and `mc_eval_stats` -- share one rollout and cannot drift
apart, and that no CLI has to thread a redundant `hidden` argument."""
import pytest

from experiments.mc_eval import mc_eval, mc_eval_stats
from experiments.comparisons import _common as C
from rl_stack import policies


@pytest.fixture
def swap_fn():
    return lambda env, obs: policies.swap_asap(env)


CELL = dict(N=6, n_ch=2, p_gen=0.6, p_swap=0.8, cutoff=20, H=100, mc_eps=40)


def test_mc_eval_stats_matches_mc_eval_return_stats_exactly(swap_fn):
    s = mc_eval_stats(swap_fn, CELL["N"], CELL["n_ch"], CELL["p_gen"],
                       CELL["p_swap"], CELL["cutoff"], CELL["H"], CELL["mc_eps"])
    m = mc_eval(swap_fn, CELL["N"], CELL["n_ch"], CELL["p_gen"], CELL["p_swap"],
                CELL["cutoff"], CELL["H"], CELL["mc_eps"], return_stats=True)
    assert s["T"] == m["T"]
    assert s["conn_rate"] == m["conn_rate"]
    assert s["mean_F_conn"] == m["mean_F_conn"]


def test_make_agent_fn_infers_hidden_from_the_checkpoint():
    """No --hidden anywhere: a checkpoint already encodes its own width."""
    import inspect
    from experiments.mc_eval import make_agent_fn
    assert "hidden" not in inspect.signature(make_agent_fn).parameters
    assert "hidden" not in inspect.signature(C.build_policies).parameters
