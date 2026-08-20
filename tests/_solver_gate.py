# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""One definition of "this test needs an SMT solver", for the whole suite.

WHY IT IS ONE. There are three routes to a solver here, not two: the z3 wheel,
the cvc5 wheel, and an EXTERNAL ``cvc5`` binary named by ``STELLING_CVC5`` or
found on ``PATH`` — the route ``_optional.cvc5_binary`` exists for and the one
``docs`` tell a user to take, since it is how you get a build the PyPI wheel
does not ship. A test that asks ``available("z3") or available("cvc5")``
therefore says "no solver" in an environment that has one, and skips a test
that would have run. That is a FALSE SKIP: the instrument reporting "not
applicable here" about a configuration in which it applies perfectly well.

**THERE IS NO COUNT IN THIS DOCSTRING, AND THAT IS THE CORRECTION.** How many
copies there were has been stated here three times and been wrong three times
— *seven*, then *eight*, then *nine* — because each correction was made by
counting by hand, which is the act that produced the wrong number in the first
place. (For the record, since the specific claims were wrong and not merely
stale: the first version named ``test_0_2_0_regression.py:73`` as reading only
the wheels when at ``f82b87b`` it already accepted the binary; the second
missed a copy in ``test_three_rows_acceptance.py``; the third still left
``test_membership_idiom_hint.py``, ``test_preconditions.py`` and
``test_doc_examples.py`` deciding for themselves.)

What is stated instead is a RULE — *no module under ``tests/`` spells the
either-solver question for itself* — with its exceptions named, and
``tests/test_solver_gate.py`` parses the tree and holds it. A backend-specific
question (``available("z3")``, or ``not (HAVE_Z3 and HAVE_CVC5)``) is a
different question with a different answer and is untouched by that rule.

WHAT THE REASON STRING IS FOR. ``"needs an SMT solver"`` is not free text. It
is a key into ``tests/test_skip_inventory.py``'s ``RULES``, which pairs it with
the condition that makes it legitimate and CONTRADICTS the skip in a session
where that condition is false. A reason spelled a little differently is an
undisclosed skip and fails the inventory — which is exactly how the jax+solvers
lane was hiding five of them (``"needs z3 wheel"``, ``"z3 not installed"``,
``"needs the cvc5 wheel"``); nobody saw, because no lane ran without the
wheels. One definition here is one place for that string to be right.

THE ROUTE IS DELIBERATELY WIDER THAN THE INVENTORY'S CONDITION, and the two do
not have to agree. ``RULES`` says the skip is legitimate when neither WHEEL is
installed; this says a test can RUN when either wheel *or* an external binary
is there. The gap is one-directional and safe: with a binary and no wheels
nothing skips, so there is no skip for the inventory to judge.

``ci.yml``'s ``test-jax-no-solvers`` JOB ASSERTS :data:`HAVE_SOLVER` DIRECTLY,
and that is not decoration either. That lane's guard step used to ask
``importlib.util.find_spec("z3"|"cvc5")``, which proves WHEEL absence and was
read as SOLVER absence — a proxy for this predicate that is false along
exactly the third route. Driven with a ``cvc5`` shim on PATH and neither wheel
installed: the old guard printed *"jax 0.11.1 — and no solver wheel"* and
exited 0, this module reported ``HAVE_SOLVER = True``, and
``tests/test_scatter_row_gates.py tests/test_inductive.py`` gave **2 failed,
21 passed**. An instrument that proves a proxy proves the proxy.
"""

from __future__ import annotations

import pytest

from stelling import _optional

#: Whether any SMT backend is reachable: either wheel, or an external ``cvc5``
#: binary named by ``STELLING_CVC5`` or found on ``PATH``.
#:
#: Wrapped, because this runs at import in every lane including ones with no
#: solver machinery at all, and a probe that raises would take whole modules
#: out of collection with an error rather than a skip.
try:
    HAVE_SOLVER = (
        _optional.available("z3")
        or _optional.available("cvc5")
        or _optional.cvc5_binary() is not None
    )
except Exception:  # pragma: no cover - environment probe only
    HAVE_SOLVER = False

#: The marker. Spelled once; see the module docstring for why the reason
#: string is load-bearing.
need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")
