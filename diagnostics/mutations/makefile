# diagnostics/mutations/Makefile
# ─────────────────────────────────
# Run from the project root:
#   make -f diagnostics/mutations/Makefile mutate-all

PYTHON   ?= python
RUNNER   := $(PYTHON) -m diagnostics.mutations.run_mutations

.PHONY: help mutate-all mutate-rl-agent mutate-rl-model mutate-rl-buffer \
        mutate-rl-env mutate-sim-repeater mutate-sim-network \
        results html dry-run clean

help:  ## Show this help
	@grep -E '^[a-z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  %-22s %s\n", $$1, $$2}'

# ── Run targets ──────────────────────────────────────────────────
mutate-all:  ## Mutation-test ALL targets
	$(RUNNER)

mutate-rl-agent:  ## Mutate rl_stack/agent.py
	$(RUNNER) -t rl_agent

mutate-rl-model:  ## Mutate rl_stack/model.py
	$(RUNNER) -t rl_model

mutate-rl-buffer:  ## Mutate rl_stack/buffer.py
	$(RUNNER) -t rl_buffer

mutate-rl-env:  ## Mutate rl_stack/env_wrapper.py
	$(RUNNER) -t rl_env_wrapper

mutate-sim-repeater:  ## Mutate quantum_repeater_sim/repeater.py
	$(RUNNER) -t sim_repeater

mutate-sim-network:  ## Mutate quantum_repeater_sim/network.py
	$(RUNNER) -t sim_network

# ── Reporting ────────────────────────────────────────────────────
results:  ## Show surviving mutants from the last run
	$(RUNNER) --results

html:  ## Generate HTML mutation report
	$(RUNNER) --html

dry-run:  ## List mutations without running tests
	$(RUNNER) --dry-run

# ── Housekeeping ─────────────────────────────────────────────────
clean:  ## Remove mutmut caches
	rm -rf .mutmut-cache diagnostics/mutations/.mutmut-cache
	@echo "Cache cleared."