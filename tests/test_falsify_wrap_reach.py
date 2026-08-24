# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE PROBE DOES NOT CATCH THE WRAP DEFECT, and this file measures it.

``SOUNDNESS.md`` carries this repository's one live, known-open false
VERIFIED: ``jnp.full((), 256, jnp.int8)`` narrows to ``0``, so the fence's
``x + 256 <= 10.0`` — false at all eleven declared points AS WRITTEN —
comes back VERIFIED. The obvious question to ask of a falsification probe
is whether it catches that, and the answer is **no**. This file is the
answer, executed rather than argued, and it is written to stay red if the
answer ever changes in either direction.

**WHY THIS IS A STRUCTURAL RESULT AND NOT A SAMPLING FAILURE.** The
distinction matters, and the two are told apart by a number this file
asserts: the probe's SKIP RATE on the reproducer is **zero**. It does not
decline the query, it does not decline a declaration, and it executes the
whole of the declared set — all eleven ``int8`` points of ``(0, 10)``, and
more besides. It finds nothing because **there is nothing there to find**:
the executed program computes ``[0.0 … 10.0]`` and the bound is ``10.0``,
so the obligation is TRUE of the program at every point of its declared
set. Only the SOURCE TEXT says ``256``.

That is what :func:`test_the_executed_program_satisfies_the_predicate_everywhere`
pins, and it is the load-bearing assertion of the file: a probe that
missed this because it sampled badly would be a probe to improve, and a
probe that misses it because the program it executes genuinely satisfies
the predicate is a probe pointed at the wrong axis.

**THE AXIS.** The probe measures *analysis versus program*: it asks
whether the thing stelling judged is true of the thing that runs. The wrap
defect lives on a different axis — *source versus program* — because jax
destroys the constant at or before the trace, so the program stelling
judged and the program the probe executes are the SAME program, ``v + 0``,
and the analysis is faithful to it. No sample budget closes that, because
the defect is inside the machinery the probe and the analysis SHARE — the
tracer, and the declaration API that runs in front of it.

**AND THE RESULT IS STRONGER THAN "THE TRACER DESTROYS IT", WHICH IS
WHERE THE ARGUMENT USED TO STOP.**
:func:`test_no_executable_form_of_the_program_carries_the_written_constant`
drives the constant with no tracer in sight: ``jnp.full((), 256,
jnp.int8)`` is ``0`` EAGERLY, and eager ``jnp.array(5, jnp.int8) + 256``
is ``5``, while numpy raises ``OverflowError`` on ``np.int8(256)`` rather
than wrapping quietly. So for the ``jnp.full`` door the destruction
happens strictly BEFORE the trace, and **there is no executable form of
this program — traced or eager — in which 256 survives.** An execution
probe of any budget or design therefore cannot see it; only the source
text, or a hook at the moment of the narrowing, can. ("At or before the
trace" is the accurate phrasing for the pair of doors; "inside the tracer"
is not, and this file used to imply it.)

**THE INSTRUMENT THAT DOES COVER THAT AXIS ALREADY EXISTS** and is a
different shape: ``stelling._tripwire``, the trace-time integer-narrowing
gate armed by ``pytest -p stelling.overflow``, which watches the constant
being destroyed as it happens rather than looking for consequences
downstream of it.
:func:`test_the_instrument_that_DOES_reach_this_axis_still_does` drives it
on the same fence, so that the pair of facts — this probe cannot, that
gate can — is a measurement in one file rather than a claim in a report.

**BOTH DOORS, BOTH CELLS.** ``SOUNDNESS.md`` prices two doors (the
``jnp.full`` constant and the inline ``x + 256``) at two separate jax
sites, and ``jax_enable_x64`` is load-bearing across that page. All four
combinations are driven, because a probe that reached one door and not the
other would be a different and much more interesting result than the one
recorded here.

**WHY THIS FILE IS NOT AN XFAIL.** An xfail would say "we expect the probe
to catch this and it does not yet". That is the wrong sentence: the probe
is not deficient here, it is out of scope here, and the measurement is
that the wrap is invisible to execution. If some future change made the
probe fire on this fence, these assertions go red — and they SHOULD,
because it would mean either that jax stopped wrapping (rewrite
``SOUNDNESS.md``) or that the probe grew a source-level reader (a genuinely
different instrument that must be described as one).

The reproducer is read out of ``SOUNDNESS.md`` through
``tests/test_soundness_wrap_reproducer.py``'s own reader, deliberately: the
mandate for this work notes that the ``tests/property/`` oracle's outcome
is sensitive to which local modules a session imported, so the
DETERMINISTIC reproducer is what is driven here.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from stelling.falsify import VerifiedFalsified, probe  # noqa: E402
from stelling._jax_compat import trace_with_jaxpr  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402

# the fence reader, not a second copy of the fence
import test_soundness_wrap_reproducer as W  # noqa: E402


@pytest.fixture(params=[False, True], ids=["x64=0", "x64=1"])
def x64(request):
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", old)


def _facts():
    return W._fence_facts(W._fence())


def _fence_harness():
    return W._compile_fence(W._fence())["_harness"]


def _inline_harness(facts):
    lo, hi = facts["box"]
    written, bound = facts["written"], facts["bound"]

    @jax.jit
    def shift(v):
        return (v + written).astype(jnp.float32)

    def harness():
        return assert_(
            shift(any_array((), facts["decl_dtype"], (lo, hi))) <= bound
        )

    return harness


def _statuses(harness):
    return [o.status for o in propagate(trace(harness)).obligations]


def _probe(harness):
    """Run the probe as the pipeline runs it, and return its report.

    A firing would RAISE, so this converts that into a returned marker
    rather than letting it escape: the point of these tests is to record
    which of the two happened, not to propagate one of them.

    ONE TRACE FEEDS BOTH HALVES, exactly as the pipeline now does it:
    `trace_with_jaxpr` returns the transcription the analysis judges and
    jax's own object the probe executes, so the statuses and the probed
    program cannot be about two different programs.
    """
    query, closed = trace_with_jaxpr(harness)
    statuses = [o.status for o in propagate(query).obligations]
    try:
        return probe(closed, statuses=statuses), None
    except VerifiedFalsified as exc:
        return exc.report, exc


# ------------------------------------------------------------ the headline


@pytest.mark.parametrize("door", ["fence", "inline"])
def test_the_probe_does_NOT_catch_the_wrap(x64, door):
    """THE MEASUREMENT. Both doors, both cells: the probe finds nothing."""
    facts = _facts()
    harness = _fence_harness() if door == "fence" else _inline_harness(facts)

    # the verdict really is the wrong VERIFIED this is about; without this
    # the rest of the file would be measuring a probe against a query that
    # had stopped being the disclosed defect
    closed = trace(harness)
    verdict = make_verdict(
        closed,
        propagate(closed),
        stelling_version="(this tree)",
        jax_version=jax.__version__,
        precision_config=f"jax_enable_x64={x64}",
    )
    assert verdict.status == "VERIFIED", (
        f"the {door} door returns {verdict.status}, not the wrong VERIFIED "
        f"SOUNDNESS.md discloses; this file measures a probe against that "
        f"defect and there is nothing here to measure against"
    )

    report, fired = _probe(harness)
    assert fired is None, (
        f"THE FALSIFICATION PROBE NOW CATCHES THE WRAP DEFECT at the "
        f"{door} door. That is a real change and this file is the wrong "
        f"shape for it: either jax stopped destroying the constant (in "
        f"which case SOUNDNESS.md's entry must be rewritten) or the probe "
        f"grew a reader for something above the trace (in which case it is "
        f"a different instrument and its reach must be described as one). "
        f"It fired with: {fired}"
    )
    assert report.falsification is None


@pytest.mark.parametrize("door", ["fence", "inline"])
def test_the_probe_did_not_merely_DECLINE_the_wrap_query(x64, door):
    """THE CONTROL THAT MAKES THE HEADLINE MEAN ANYTHING.

    "The probe found nothing" and "the probe did not run" are the two
    readings, and only the skip rate tells them apart. Here it is zero:
    the query was not declined, no declaration was declined, and every
    point built was executed and admitted.
    """
    facts = _facts()
    harness = _fence_harness() if door == "fence" else _inline_harness(facts)
    report, fired = _probe(harness)
    assert fired is None
    assert report.declined is None, (
        f"the probe DECLINED the wrap query ({report.declined}); then this "
        f"file measures nothing about its reach, only about its coverage"
    )
    assert report.points_executed > 0
    assert report.points_admissible == report.points_built, (
        f"{report.points_built - report.points_admissible} of "
        f"{report.points_built} sampled points were unusable; the wrap "
        f"query has one bounded int8 declaration and no assume, so every "
        f"point should be admissible and the skip accounting has drifted"
    )
    assert report.skip_rate == 0.0, (
        f"skip rate {report.skip_rate} on the wrap reproducer; this file's "
        f"whole claim is that the probe RAN and found nothing, which a "
        f"non-zero skip rate would undermine"
    )


def test_the_executed_program_satisfies_the_predicate_everywhere(x64):
    """WHY there is nothing to find, in the program's own numbers.

    The declared set is the eleven ``int8`` points of ``(0, 10)``. This
    executes the fence's own jitted function at every one of them and
    shows the obligation TRUE at each — while the same arithmetic on the
    constant AS WRITTEN is false at each. Two columns, one loop: that
    difference IS the defect, and it is invisible to anything downstream
    of the trace.
    """
    facts = _facts()
    lo, hi = facts["box"]
    written, bound = facts["written"], facts["bound"]

    @jax.jit
    def shift(v):
        return (v + written).astype(jnp.float32)

    executed, as_written = [], []
    for p in range(int(lo), int(hi) + 1):
        executed.append(float(np.asarray(shift(jnp.array(p, facts["decl_dtype"])))))
        as_written.append(p + written)

    assert all(v <= bound for v in executed), (
        f"the executed program VIOLATES the bound somewhere in the declared "
        f"box: {executed} against {bound}. Then the probe's null result on "
        f"this query was a sampling failure after all, and this file's "
        f"framing — that there is nothing there to find — is wrong."
    )
    assert not any(v <= bound for v in as_written), (
        f"the predicate as WRITTEN is no longer false at every declared "
        f"point ({as_written} against {bound}); SOUNDNESS.md's entry says "
        f"it is, and must be re-driven"
    )
    assert executed != as_written


def _narrowings(harness):
    """Narrowings the trace-time gate counts for one harness, or ``None``.

    Driven through the pipeline's OWN gate helpers — the same ones
    ``preconditions._pipeline`` pushes and pops around its trace, with the
    same fresh-closure trick — so this measures the gate as the pipeline
    uses it and not a private arrangement of it.
    """
    from stelling import _tripwire
    from stelling._tripwire import _pop_gate, _push_gate

    status, _ = _tripwire.arm()
    if not status.armed:
        return None
    try:
        _push_gate()
        try:
            trace(lambda: harness())
        finally:
            return _pop_gate()
    finally:
        _tripwire.disarm()


def test_the_trace_gate_reaches_the_inline_door_where_the_probe_does_not(x64):
    """The complement, so the pair is MEASURED rather than asserted.

    On the inline door the two instruments are exactly complementary: the
    execution probe finds nothing (above) and the trace-time gate counts
    the narrowing. That pair is the argument that the two live on
    different axes, and it is worth an executed test rather than a
    sentence in a report.
    """
    harness = _inline_harness(_facts())
    n = _narrowings(harness)
    if n is None:
        pytest.skip("tripwire could not arm on this jax")
    assert n > 0, (
        "the trace-time narrowing gate saw NO narrowing on the inline "
        "door, which is the door it is documented to reach. This file's "
        "claim is that the wrap is reachable above the trace and not "
        "below it; with this red, the first half of that claim is "
        "unsupported."
    )
    report, fired = _probe(harness)
    assert fired is None and report.declined is None


def test_no_executable_form_of_the_program_carries_the_written_constant(
    x64,
):
    """THE STRONGEST FORM OF THE NULL RESULT, and it needs no probe at all.

    The probe finding nothing is one measurement; that no execution COULD
    find anything is a different and much stronger one, and it is settled
    without tracing anything:

    * ``jnp.full((), 256, jnp.int8)`` evaluates to ``0`` eagerly — the
      value is destroyed at construction, before any jaxpr exists;
    * eager ``jnp.array(5, jnp.int8) + 256`` is ``5`` — the addend is
      destroyed the same way;
    * numpy, asked the same thing, RAISES ``OverflowError`` rather than
      wrapping, which is what makes this a jax convention and not an
      inevitability of the format.

    So the object the probe would have to execute in order to see ``256``
    does not exist in either evaluation mode. That is why this file's
    result is structural: it is not that the sampler looked in the wrong
    place, it is that no sampler has anything to look at. Driven in both
    ``x64`` cells because the whole ``SOUNDNESS.md`` page is sensitive to
    that dial.
    """
    from stelling._tripwire.eager import expected_truncation

    # A REGION DECLARATION. The two assertions below ARE the null result:
    # they say that the object a probe would have to execute in order to see
    # 256 does not exist, and they say it by destroying 256 twice. With
    # `--stelling-eager-truncation=error` armed session-wide the first of
    # them raises -- correctly, because that detector is the answer to this
    # very door -- and the measurement still has to be made. `intentional_
    # wrap` cannot serve: writing 0 asserts nothing about what jax does with
    # a 256.
    with expected_truncation(
        "the null result IS the destruction of 256: these lines exist to "
        "show that no executable form of the program carries it"
    ):
        assert int(jnp.full((), 256, jnp.int8)) == 0, (
            "jnp no longer destroys the int8 literal eagerly; if it now "
            "raises or saturates, SOUNDNESS.md's open false VERIFIED has "
            "changed shape and this whole file needs rewriting rather than "
            "patching"
        )
        assert int(jnp.array(5, jnp.int8) + 256) == 5
    with pytest.raises(OverflowError):
        np.int8(256)


def test_NEITHER_instrument_reaches_the_jnp_full_door(x64):
    """A GAP THIS BATCH FOUND AND DID NOT CLOSE — recorded, not fixed.

    The execution probe misses the ``jnp.full`` door for the reason this
    whole file is about. The trace-time narrowing gate ALSO misses it, and
    that was not expected: measured here, it counts **zero** narrowings on
    the fence, and zero again on a variant that calls ``jnp.full`` INSIDE
    the traced harness rather than above it — so it is not simply that the
    fence builds its constant eagerly. The narrowing at that door happens
    somewhere the gate's fold-rule wrapper does not see.

    ``SOUNDNESS.md`` already prices the fence's door and the inline door
    as two doors needing two fixes. This adds the instrument column to
    that table: the inline door has a working instrument and the fence's
    door has none. It is asserted in the direction of the gap, so that the
    day something starts reaching this door, this test goes red and the
    disclosure gets rewritten rather than quietly aging.
    """
    facts = _facts()
    n_fence = _narrowings(_fence_harness())
    if n_fence is None:
        pytest.skip("tripwire could not arm on this jax")

    lo, hi = facts["box"]

    def inside():
        offset = jnp.full((), facts["written"], getattr(jnp, facts["dtype"]))

        @jax.jit
        def shift(v):
            return (v + offset).astype(jnp.float32)

        return assert_(
            shift(any_array((), facts["decl_dtype"], (lo, hi))) <= facts["bound"]
        )

    n_inside = _narrowings(inside)

    assert n_fence == 0 and n_inside == 0, (
        f"the trace-time gate now counts narrowings at the jnp.full door "
        f"(fence={n_fence}, constant-built-inside={n_inside}). That door "
        f"had NO instrument when this test was written, and gaining one is "
        f"a change worth recording: update SOUNDNESS.md's two-doors entry "
        f"with the instrument column rather than deleting this test."
    )
    report, fired = _probe(_fence_harness())
    assert fired is None, "and the execution probe does not reach it either"
