"""
mutmut_config.py  –  mutation filtering hooks
==============================================

mutmut discovers this file via the ``MUTMUT_CONFIG`` environment variable
set by ``run_mutations.py``.  The hooks below tell mutmut to skip mutations
that would produce noise (unkillable mutants that don't reflect real test
weaknesses).

Customisation
-------------
Add patterns to ``SKIP_LINE_PATTERNS`` for lines that should never be
mutated.  Use ``pre_mutation_ast`` for AST-level filtering.
"""

from __future__ import annotations


# ── Lines containing any of these substrings are never mutated ───────────
SKIP_LINE_PATTERNS: tuple[str, ...] = (
    # Explicit opt-out marker — add ``# pragma: no mutate`` to any line
    "pragma: no mutate",

    # Logging / warnings (side-effect-only, not covered by logic tests)
    "logger.",
    "logging.",
    "log.",
    "warnings.warn",

    # Coverage markers
    "# nocov",
    "# pragma: no cover",

    # Abstract / stub methods
    "raise NotImplementedError",
    "pass  # abstract",

    # Type-checking-only blocks
    "TYPE_CHECKING",

    # Common __repr__ / __str__ boilerplate
    "def __repr__",
    "def __str__",

    # Docstring-only lines (mutmut sometimes mutates string constants)
    '"""',
    "'''",
)


# ── Lines containing these patterns in the context of the quantum sim ────
# These are physics-documentation comments that mutmut may try to mutate
# as string constants.
PHYSICS_COMMENT_PATTERNS: tuple[str, ...] = (
    "Werner",
    "BBPSSW",
    "fidelity",
    "decoherence",
    "entangle",
)


def pre_mutation(context) -> None:  # noqa: ANN001
    """Called before each mutation is applied.

    Set ``context.skip = True`` to skip the mutant entirely.

    Available attributes on *context*:
        filename       – path of the file being mutated
        current_line   – source line that will be mutated
        mutation_id    – internal mutation counter
    """
    line = context.current_line.strip()

    # Skip if the line matches any known-untestable pattern.
    for pattern in SKIP_LINE_PATTERNS:
        if pattern in line:
            context.skip = True
            return

    # Skip string-constant mutations on pure comment/docstring lines.
    if line.startswith("#"):
        context.skip = True
        return


def pre_mutation_ast(context) -> None:  # noqa: ANN001
    """Called after the AST mutation is chosen but before compilation.

    ``context.node`` holds the AST node.  Currently unused — extend this
    for surgical, node-level filtering (e.g. skip mutations inside
    specific function bodies).
    """


def post_mutation(context) -> None:  # noqa: ANN001
    """Called after a mutation round completes (killed or survived).

    Can be used for custom metrics collection.  Currently a no-op.
    """