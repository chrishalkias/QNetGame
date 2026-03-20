# Mutation Testing Suite

This directory contains the **mutmut**-based mutation testing harness for the
Quantum Repeater Network Simulator project.  Mutation testing works by
injecting small faults ("mutants") into the production source code and
verifying that the existing unit tests catch them.  A surviving mutant
reveals a gap in test coverage or assertion strength.

## Targets

The runner automatically maps each production module to the test file(s)
that exercise it:

| Target name       | Source file                            | Test file(s)                                       |
|-------------------|----------------------------------------|----------------------------------------------------|
| `rl_agent`        | `rl_stack/agent.py`                    | `diagnostics/unittests/test_rl_stack.py`           |
| `rl_model`        | `rl_stack/model.py`                    | `diagnostics/unittests/test_rl_stack.py`           |
| `rl_buffer`       | `rl_stack/buffer.py`                   | `diagnostics/unittests/test_rl_stack.py`           |
| `rl_env_wrapper`  | `rl_stack/env_wrapper.py`              | both test files (`diagnostics/unittests/`)          |
| `sim_repeater`    | `quantum_repeater_sim/repeater.py`     | `diagnostics/unittests/test_simulator.py`          |
| `sim_network`     | `quantum_repeater_sim/network.py`      | `diagnostics/unittests/test_simulator.py`          |

## Quick start

```bash
# Install mutmut (if not already in requirements.txt)
pip install "mutmut>=2.0"

# Run all targets
python -m diagnostics.mutations.run_mutations

# Run a single target
python -m diagnostics.mutations.run_mutations --target rl_agent

# Dry-run (list mutations without executing tests)
python -m diagnostics.mutations.run_mutations --dry-run

# View surviving mutants from last run
python -m diagnostics.mutations.run_mutations --results

# Show the diff of a specific mutant
python -m diagnostics.mutations.run_mutations --show 42

# Generate an HTML report
python -m diagnostics.mutations.run_mutations --html
```

Or use the Makefile shortcuts from the project root:

```bash
make -f diagnostics/mutations/Makefile mutate-all
make -f diagnostics/mutations/Makefile mutate-sim-repeater
make -f diagnostics/mutations/Makefile results
make -f diagnostics/mutations/Makefile html
```

## Configuration

### mutmut_config.py

The `mutmut_config.py` file contains hooks that filter out mutations that
would produce noise — lines that are inherently untestable (logging,
docstrings, `__repr__`, abstract stubs).  The runner points mutmut at this
file via the `MUTMUT_CONFIG` environment variable.

Add `# pragma: no mutate` to any source line you want to exclude.

### setup.cfg / pyproject.toml

The `setup.cfg.fragment` file contains a `[mutmut]` section you can merge
into your project-root `setup.cfg` (or the equivalent `[tool.mutmut]` for
`pyproject.toml`).  This is only needed if you want to run `mutmut run`
directly without the wrapper script.

## Interpreting results

| Exit code | Meaning                            |
|-----------|------------------------------------|
| 0         | All mutants killed — tests are strong |
| 2         | Some mutants survived — investigate   |
| 1         | Error (crash, missing files, etc.)    |

For each surviving mutant, run `--show <id>` to see the exact code change
that your tests missed.  Then either strengthen the assertion or add a new
test case that would fail under that mutation.

## Files

```
diagnostics/mutations/
├── __init__.py
├── run_mutations.py      # main CLI entry point
├── mutmut_config.py      # pre/post mutation hooks
├── Makefile              # convenience targets
├── setup.cfg.fragment    # merge into project root setup.cfg
└── README.md             # this file
```