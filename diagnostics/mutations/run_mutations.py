#!/usr/bin/env python3
"""
Mutation testing runner for the diagnostics test suite.

Uses mutmut to apply source-code mutations and checks whether the existing
unit tests catch them.  A "surviving" mutant means there is a gap in test
coverage or assertion strength.

Targets
-------
Each entry in TARGETS maps a *source module* to the *test file(s)* that
exercise it.  The source paths are derived from the imports in
``test_rl_stack.py`` and ``test_simulator.py``.

Usage
-----
    # Run every mutation target
    python -m diagnostics.mutations.run_mutations

    # Run a single target by name
    python -m diagnostics.mutations.run_mutations --target rl_agent

    # Show surviving mutants from the last run
    python -m diagnostics.mutations.run_mutations --results

    # Show the diff for a specific mutant
    python -m diagnostics.mutations.run_mutations --show 42

    # Generate an HTML report (mutmut >= 2.0)
    python -m diagnostics.mutations.run_mutations --html

    # Dry-run: list what would be mutated without running tests
    python -m diagnostics.mutations.run_mutations --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Project root is two levels above diagnostics/mutations/ ──────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Point mutmut at our custom config hooks ──────────────────────────────
os.environ.setdefault(
    "MUTMUT_CONFIG",
    str(Path(__file__).resolve().parent / "mutmut_config.py"),
)


# ═══════════════════════════════════════════════════════════════════════════
# Target definitions
# ═══════════════════════════════════════════════════════════════════════════
#
# Source paths inferred from the test-file import statements:
#
#   test_rl_stack.py imports from:
#       rl_stack.env_wrapper   → rl_stack/env_wrapper.py
#       rl_stack.model         → rl_stack/model.py
#       rl_stack.buffer        → rl_stack/buffer.py
#       rl_stack.agent         → rl_stack/agent.py
#
#   test_simulator.py imports from:
#       quantum_repeater_sim.repeater → quantum_repeater_sim/repeater.py
#       quantum_repeater_sim.network  → quantum_repeater_sim/network.py
#       rl_stack.env_wrapper          → rl_stack/env_wrapper.py  (shared)
#
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MutationTarget:
    """A pair of (source_to_mutate, tests_to_run)."""

    name: str
    source: str                # relative to PROJECT_ROOT
    tests: str                 # relative to PROJECT_ROOT (file or dir)
    extra_args: list[str] = field(default_factory=list)


# ── Test files ───────────────────────────────────────────────────────────
_RL_TESTS  = "diagnostics/unittests/test_rl_stack.py"
_SIM_TESTS = "diagnostics/unittests/test_simulator.py"
_ALL_TESTS = "diagnostics/unittests/"

TARGETS: list[MutationTarget] = [
    # ── RL-stack sources (exercised by test_rl_stack.py) ─────────────
    MutationTarget(
        name="rl_agent",
        source="rl_stack/agent.py",
        tests=_RL_TESTS,
    ),
    MutationTarget(
        name="rl_model",
        source="rl_stack/model.py",
        tests=_RL_TESTS,
    ),
    MutationTarget(
        name="rl_buffer",
        source="rl_stack/buffer.py",
        tests=_RL_TESTS,
    ),
    MutationTarget(
        name="rl_env_wrapper",
        source="rl_stack/env_wrapper.py",
        tests=_ALL_TESTS,       # both test files exercise env_wrapper
    ),
    # ── Simulator sources (exercised by test_simulator.py) ───────────
    MutationTarget(
        name="sim_repeater",
        source="quantum_repeater_sim/repeater.py",
        tests=_SIM_TESTS,
    ),
    MutationTarget(
        name="sim_network",
        source="quantum_repeater_sim/network.py",
        tests=_SIM_TESTS,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run(
    cmd: list[str],
    *,
    check: bool = False,
    **kw,
) -> subprocess.CompletedProcess:
    """Run a command from PROJECT_ROOT, printing it first."""
    print(f"\n{'─'*72}")
    print(f"  ▶  {' '.join(cmd)}")
    print(f"{'─'*72}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=check, **kw)


def _ensure_mutmut() -> None:
    """Make sure mutmut is importable; bail with instructions if not."""
    try:
        import mutmut  # noqa: F401
    except ImportError:
        sys.exit(
            "mutmut is not installed.  Add the following to requirements.txt:\n\n"
            "    mutmut>=2.0\n\n"
            "Then run:  pip install -r requirements.txt"
        )


def _validate_target(target: MutationTarget) -> list[str]:
    """Return a list of error strings (empty means OK)."""
    errors: list[str] = []
    src = PROJECT_ROOT / target.source
    tst = PROJECT_ROOT / target.tests
    if not src.exists():
        errors.append(f"  source not found: {src}")
    if not tst.exists():
        errors.append(f"  tests  not found: {tst}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════
# Core mutation run
# ═══════════════════════════════════════════════════════════════════════════

def run_mutmut(target: MutationTarget, *, dry_run: bool = False) -> int:
    """Run mutmut against *target*.  Returns the process exit code."""
    errors = _validate_target(target)
    if errors:
        print(f"\n⚠  Skipping target '{target.name}' — path problems:")
        print("\n".join(errors))
        print(f"   Adjust TARGETS in {Path(__file__).name}\n")
        return 1

    # pytest runner invoked by mutmut for every mutant.
    #   -x         stop on first failure (we only need one kill)
    #   -q         quiet
    #   --tb=line  minimal traceback
    test_runner = (
        f"python -m pytest {target.tests} -x -q --tb=line --no-header"
    )

    cmd: list[str] = [
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate", target.source,
        "--tests-dir", str(Path(target.tests).parent),
        "--runner", test_runner,
    ]
    cmd += target.extra_args

    if dry_run:
        cmd.append("--dry-run")

    result = _run(cmd)
    return result.returncode


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def show_results() -> None:
    """Print survived mutants from the last run."""
    _run([sys.executable, "-m", "mutmut", "results"])


def show_html() -> None:
    """Generate an HTML report (written to html/ in project root)."""
    _run([sys.executable, "-m", "mutmut", "html"])
    print(f"\nReport → {PROJECT_ROOT / 'html'}/index.html\n")


def show_mutant(mutant_id: str) -> None:
    """Show the diff for a specific mutant id."""
    _run([sys.executable, "-m", "mutmut", "show", mutant_id])


def _print_summary_table(results: dict[str, dict]) -> None:
    """Pretty-print a per-target result table."""
    col_w = 22
    hdr = f"  {'Target':<{col_w}} {'Killed':>7} {'Survived':>9} {'Score':>7}  "
    bar = "═" * (len(hdr) - 4)
    dash = "─" * (len(hdr) - 4)

    print()
    print(f"  ╔{bar}╗")
    print(f"  ║{'  MUTATION TESTING SUMMARY'.center(len(hdr) - 4)}║")
    print(f"  ╠{bar}╣")
    print(f"  ║{hdr.strip().center(len(hdr) - 4)}║")
    print(f"  ╟{dash}╢")

    total_k, total_s = 0, 0
    for name, stats in results.items():
        k = stats.get("killed", 0)
        s = stats.get("survived", 0)
        err = stats.get("error", False)
        total_k += k
        total_s += s
        t = k + s
        score = "ERROR" if err else (f"{k / t * 100:.1f}%" if t else "n/a")
        print(f"  ║  {name:<{col_w}} {k:>7} {s:>9} {score:>7}  ║")

    total = total_k + total_s
    overall = f"{total_k / total * 100:.1f}%" if total else "n/a"
    print(f"  ╟{dash}╢")
    print(f"  ║  {'TOTAL':<{col_w}} {total_k:>7} {total_s:>9} {overall:>7}  ║")
    print(f"  ╚{bar}╝")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_mutations",
        description="Run mutmut mutation tests for the diagnostics suite.",
    )
    p.add_argument(
        "--target", "-t",
        choices=[t.name for t in TARGETS],
        default=None,
        help="Run mutations for a single target (default: all).",
    )
    p.add_argument(
        "--results", "-r",
        action="store_true",
        help="Show surviving mutants from the last run.",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="Generate an HTML mutation report.",
    )
    p.add_argument(
        "--show",
        metavar="MUTANT_ID",
        help="Show the diff for a specific mutant id.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List mutations without running tests.",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Silently skip targets whose source files don't exist yet.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    _ensure_mutmut()
    args = build_parser().parse_args(argv)

    # ── Reporting sub-commands ───────────────────────────────────────
    if args.results:
        show_results()
        return
    if args.html:
        show_html()
        return
    if args.show:
        show_mutant(args.show)
        return

    # ── Select targets ───────────────────────────────────────────────
    selected = TARGETS
    if args.target:
        selected = [t for t in TARGETS if t.name == args.target]

    if args.skip_missing:
        selected = [t for t in selected if not _validate_target(t)]

    if not selected:
        sys.exit("No valid targets found.  Check TARGETS in run_mutations.py.")

    # ── Run each target ──────────────────────────────────────────────
    run_results: dict[str, dict] = {}
    any_crashed = False

    for target in selected:
        print(f"\n{'═'*72}")
        print(f"  Mutation target : {target.name}")
        print(f"  Source          : {target.source}")
        print(f"  Tests           : {target.tests}")
        print(f"{'═'*72}")

        rc = run_mutmut(target, dry_run=args.dry_run)

        # mutmut exit codes: 0 = all killed, 2 = survived mutants, 1 = error
        if rc not in (0, 2):
            any_crashed = True
            run_results[target.name] = {"killed": 0, "survived": 0, "error": True}
        else:
            # Best-effort parse of `mutmut results` output.
            probe = subprocess.run(
                [sys.executable, "-m", "mutmut", "results"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            survived = probe.stdout.lower().count("survived")
            killed, total = 0, 0
            for line in probe.stdout.splitlines():
                low = line.lower()
                if "killed" in low and "out of" in low:
                    parts = low.split()
                    try:
                        killed = int(parts[parts.index("killed") + 1])
                        total  = int(parts[parts.index("of") + 1])
                    except (ValueError, IndexError):
                        pass
            run_results[target.name] = {
                "killed": killed,
                "survived": total - killed if total else survived,
            }

    # ── Summary table ────────────────────────────────────────────────
    if run_results:
        _print_summary_table(run_results)

    if any_crashed:
        sys.exit(1)


if __name__ == "__main__":
    main()