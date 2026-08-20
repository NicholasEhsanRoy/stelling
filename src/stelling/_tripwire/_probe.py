# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The two programs the self-check traces. They live in their own file on purpose.

The tripwire strips its own wrapper's frames before attributing a narrowing,
so a probe written inside ``_adapter_jax.py`` would be attributed to nobody
and would land in the suppressed bucket — the self-check would then be
measuring a different code path from the one users get. Here, the probe is
just another module the way a user's test module is, so ``selfcheck`` drives
the whole pipeline: hook, range check, stack walk, attribution, source quote,
finding.

Names no jax. The array comes in as an argument.

**Do not reformat these two functions.** ``selfcheck`` asserts the attributed
line number is the one the constant is written on, which is how attribution is
checked rather than assumed, and :data:`OVER_LINE` / :data:`UNDER_LINE` are
derived from the function objects rather than typed so that moving them is
safe.
"""

from __future__ import annotations

#: The value written in :func:`over`, and the dtype the self-check narrows it
#: into. 256 is the smallest positive integer that ``int8`` cannot hold, so
#: the positive direction is one step over the edge and the negative
#: direction (:data:`UNDER`) is comfortably inside it.
OVER = 256
UNDER = 3
DTYPE = "int8"


def over(a):
    """Out of range for int8: the narrowing must be seen."""
    return a + 256


def under(a):
    """In range for int8: the narrowing happens and must NOT be reported."""
    return a + 3


#: Line numbers of the two constants, derived from the code objects so that
#: editing this file cannot silently make the attribution check vacuous.
OVER_LINE = over.__code__.co_firstlineno + 2
UNDER_LINE = under.__code__.co_firstlineno + 2


# ---------------------------------------------------------------------------
# The EAGER detector's control programs (Mode 2).
#
# Here for the same reason the two above are here: a probe written inside the
# adapter or the canary is a probe on a code path no user takes. These take
# ``jnp`` as an argument rather than importing it, exactly as the two above
# take an array, so this module still names no jax and the canary's jax-less
# test lane can hand in a stand-in.
# ---------------------------------------------------------------------------

#: The value :func:`construct_over` writes. 256 into ``int8`` for the same
#: reason :data:`OVER` is: one step over the edge.
EAGER_OVER = 256
EAGER_UNDER = 3
EAGER_DTYPE = "int8"


def construct_over(jnp):
    """Construct an int8 array from a value int8 cannot hold.

    With the eager detector armed this MUST raise
    :class:`stelling.EagerTruncationError`. Without it, it is ``0`` and jax
    says nothing at all, which is the defect the detector exists for.
    """
    return jnp.full((), 256, jnp.int8)


def construct_under(jnp):
    """The same construction with a value that FITS. It must NOT raise.

    The negative direction, and it is not decoration: a hook replaced by
    "raise on everything" passes the positive control and fails this one.
    """
    return jnp.full((), 3, jnp.int8)
