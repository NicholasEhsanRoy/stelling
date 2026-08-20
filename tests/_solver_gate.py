# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""One definition of "this test needs an SMT solver", for the whole suite.

WHY IT IS ONE AND NOT NINE. This predicate was written out **nine** times in
``tests/``, and exactly **one** of them — ``test_contracts.py``'s
``_HAVE_SOLVER = available("z3") or available("cvc5")`` — read only the two
wheels, while the other eight also accept an EXTERNAL ``cvc5`` binary: the
``STELLING_CVC5`` / ``cvc5``-on-PATH route that ``_optional.cvc5_binary``
exists for and that ``docs`` tell a user to use to get a build the PyPI wheel
does not ship. In an environment with that binary and no wheels the narrow
spelling skips tests that would have passed, which is a false skip: the
instrument reports "not applicable here" about a configuration in which it
applies perfectly well.

THE COUNT AND THE NAMES ARE CORRECTED HERE, and the correction is the point
rather than a tidy-up. This file first said *"seven times … two of them
(``test_0_2_0_regression.py``, ``test_contracts.py``) read only the two
wheels"*. At ``f82b87b``, ``test_0_2_0_regression.py:73`` read
``available("cvc5") or cvc5_binary() is not None`` — it accepted the binary,
and naming it as narrow was simply wrong. Eight definitions were folded in
when this file was written and a NINTH survived under the same name in
``test_three_rows_acceptance.py`` (with its own reason string, ``"needs a
solver"``), so *"one definition for the whole suite"* was not true either. It
is now; the ninth was folded in and this sentence counts it.

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
