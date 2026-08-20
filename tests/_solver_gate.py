# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""One definition of "this test needs an SMT solver", for the whole suite.

WHY IT IS ONE AND NOT SEVEN. This predicate was written out seven times in
``tests/`` in four different spellings, and two of them
(``test_0_2_0_regression.py``, ``test_contracts.py``) read only the two wheels
while the other five also accept an EXTERNAL ``cvc5`` binary — the
``STELLING_CVC5`` / ``cvc5``-on-PATH route that ``_optional.cvc5_binary``
exists for and that ``docs`` tell a user to use to get a build the PyPI wheel
does not ship. In an environment with that binary and no wheels the two narrow
spellings skip tests that would have passed, which is a false skip: the
instrument reports "not applicable here" about a configuration in which it
applies perfectly well.

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
