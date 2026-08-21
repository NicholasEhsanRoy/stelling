# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""WHICH constant-construction routes the armed trace gate actually watches.

The gate in :func:`stelling.preconditions.check` refuses to certify a trace
in which an integer constant was narrowed. What it has never said is which
ways of writing that constant it can SEE — and the answer is not "all of
them". ``jnp.full(shape, N, dt)`` narrows N in numpy before any jax primitive
runs, so the const-fold rule the tripwire hooks is handed a value that is
already in range, does not fire, and the gate certifies a program whose
written constant no longer exists.

THIS FILE IS THE INVENTORY, AND IT IS ASSERTED RATHER THAN ASSERTED-TO.
:data:`GATE_COVERAGE` declares a bucket per route; the test below MEASURES
every route through ``check()`` itself and compares. Prose alone is how this
gap survived: ``report.UNCOVERED`` has named ``jnp.full`` since the tripwire
shipped, and nothing anywhere would have gone red if a jax release had moved
a watched route into the silent set, or if a stelling change had quietly
stopped watching one. This is ``tests/test_doc_examples.py``'s
``EXPECTED_HASH_COVERAGE`` idiom, applied to the door list.

THE FOUR BUCKETS, which are four different facts and not four shades of one:

``watched``
    the constant is destroyed at TRACE time and the gate sees it: ``check()``
    returns UNKNOWN with ``trace unfaithful``. This is the covered case.

``unwatched``
    the constant is destroyed and the gate does NOT see it. ``check()``
    returns a verdict about a program the source does not describe. Every
    route in this bucket is a live hole, is named in ``report.UNCOVERED``,
    and is measured here so that it stays named.

``loud``
    jax itself raises ``OverflowError`` before anything is traced. Nothing
    silent happens, so there is nothing for a tripwire to catch.

``deferred``
    the written constant reaches the jaxpr INTACT — the narrowing is not a
    transcription loss — so the trace gate has nothing to see and correctly
    sees nothing. These are not holes, and *"the gate ignores it"* is only
    acceptable while something else does not, so every row DECLARES the
    mechanism that declines it in :data:`DEFERRED_CATCHER` and the test
    below requires that mechanism to appear in the verdict's own notes.
    TWO mechanisms, not one, and saying "the transfer declines them" of the
    bucket was true of five of its six rows.

WHY N=40000 AND int16. It is out of range for int16 and wraps to -25536, far
enough from the declared envelope that a verdict on the wrapped program and a
verdict on the written one cannot agree by accident.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import re  # noqa: E402

import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from stelling import _tripwire  # noqa: E402
from stelling._tripwire import eager as _eager, report  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

from _repo_files import read_text_files  # noqa: E402

#: Out of int16 range; wraps to -25536.
OVER = 40000
DTYPE = jnp.int16
ENVELOPE = (0, 100)
#: Every route's obligation is ``value < BOUND``. True of the wrapped program
#: (-25536 + [0, 100]) and false of the written one (40000 + [0, 100]), so a
#: verdict cannot be right about both.
BOUND = 200


@pytest.fixture(autouse=True)
def _armed():
    status, _ = _tripwire.arm()
    if not status.armed:
        pytest.skip(f"tripwire could not arm: {status.code}")
    yield
    _tripwire.disarm()


#: One jitted helper per inline setting, built once and SHARED, because the
#: door being measured is a warm trace cache and a helper rebuilt per call
#: would be cold every time — the shape that made the original measurement
#: read as "no problem here".
_HELPERS: dict = {}


def _helper(inline):
    h = _HELPERS.get(inline)
    if h is None:
        h = jax.jit(lambda z: z + OVER, inline=inline)
        _HELPERS[inline] = h
    return h


#: route -> the body that writes OVER into a traced program.
ROUTES = {
    "x + N": lambda x: x + OVER,
    "N + x": lambda x: OVER + x,
    "x - N": lambda x: x - OVER,
    "x * N": lambda x: x * OVER,
    "x < N": lambda x: (x < OVER).astype(DTYPE),
    "x & N": lambda x: x & OVER,
    "jnp.maximum(x, N)": lambda x: jnp.maximum(x, OVER),
    "jnp.array(N).astype(dt)": lambda x: x + jnp.array(OVER).astype(DTYPE),
    "jnp.astype(jnp.array(N), dt)": lambda x: x + jnp.astype(jnp.array(OVER), DTYPE),
    "x.at[0].set(N)": lambda x: x.at[0].set(OVER),
    "x.at[0].add(N)": lambda x: x.at[0].add(OVER),
    "lax.cond branch": lambda x: lax.cond(
        x[0] > 0, lambda z: z + OVER, lambda z: z, x
    ),
    "lax.scan body": lambda x: lax.scan(lambda c, y: (c, y + OVER), x[0], x)[1],
    "lax.while_loop body": lambda x: lax.while_loop(
        lambda z: z[0] < 0, lambda z: z + OVER, x
    ),
    "jax.vmap(z + N)": lambda x: jax.vmap(lambda z: z + OVER)(x),
    "@jax.jit helper": lambda x: _helper(False)(x),
    "@jax.jit(inline=True) helper": lambda x: _helper(True)(x),
    "jnp.full(shape, N, dt)": lambda x: x + jnp.full(x.shape, OVER, DTYPE),
    "jnp.full_like(x, N)": lambda x: x + jnp.full_like(x, OVER),
    "lax.full(shape, N, dt)": lambda x: x + lax.full(x.shape, OVER, DTYPE),
    "lax.full_like(x, N)": lambda x: x + lax.full_like(x, OVER),
    "lax.convert_element_type(N, dt)": lambda x: x
    + lax.convert_element_type(OVER, DTYPE),
    "np.asarray(N).astype(dt)": lambda x: x + np.array(OVER).astype(np.int16),
    "jnp.asarray(np.array(N), dt)": lambda x: x
    + jnp.asarray(np.array(OVER), dtype=DTYPE),
    "jnp.stack([x, jnp.full(N)])": lambda x: jnp.stack(
        [x, jnp.full(x.shape, OVER, DTYPE)]
    )[0],
    # B8c fixup. MEASURED AND UNROSTERED until now: both were driven closed
    # by the eager detector and named in `design/eager-truncation-detector.md`
    # as closed, and neither was a row of GATE_COVERAGE or EAGER_COVERAGE — so
    # nothing here would have reddened if a jax release moved either of them.
    # A route that is measured, disclosed as closed, and enrolled in no
    # inventory is a claim with no guard behind it.
    "lax.select(c, jnp.full(N), x)": lambda x: lax.select(
        x > 0, jnp.full(x.shape, OVER, DTYPE), x
    ),
    "jnp.take(x, i, fill_value=N)": lambda x: jnp.take(
        x, jnp.array([9, 0]), mode="fill", fill_value=OVER
    ),
    "jnp.array(N, dtype=dt)": lambda x: x + jnp.array(OVER, dtype=DTYPE),
    "jnp.asarray(N, dtype=dt)": lambda x: x + jnp.asarray(OVER, dtype=DTYPE),
    "jnp.int16(N)": lambda x: x + jnp.int16(OVER),
    "x // N": lambda x: x // OVER,
    "x % N": lambda x: x % OVER,
    "jnp.where(c, N, x)": lambda x: jnp.where(x > 0, OVER, x),
    "jnp.clip(x, 0, N)": lambda x: jnp.clip(x, 0, OVER),
    "jnp.pad(x, 1, constant_values=N)": lambda x: jnp.pad(
        x, 1, constant_values=OVER
    )[:2],
}

#: THE DECLARATION. Measured on jax 0.11.0 with x64 on, by the test below and
#: not typed from hope. Moving a route between buckets is a change to what
#: this tool can see and must be argued in this comment, not merely made
#: green: `unwatched` -> `watched` is a hole closing (say what closed it);
#: `watched` -> `unwatched` is a hole OPENING and needs a line in
#: `report.UNCOVERED` in the same commit, which
#: `test_every_unwatched_route_is_named_in_the_reports_coverage_claim`
#: enforces BY ITERATING THIS DICT. It used to walk a copy of this set typed
#: beside it, so an added `unwatched` row passed all seven tests undisclosed
#: (measured, twice); a pointer to an enforcement is worth what the
#: enforcement iterates.
GATE_COVERAGE = {
    "x + N": "watched",
    "N + x": "watched",
    "x - N": "watched",
    "x * N": "watched",
    "x < N": "watched",
    "x & N": "watched",
    "jnp.maximum(x, N)": "watched",
    "jnp.array(N).astype(dt)": "watched",
    "jnp.astype(jnp.array(N), dt)": "watched",
    "x.at[0].set(N)": "watched",
    "x.at[0].add(N)": "watched",
    "lax.cond branch": "watched",
    "lax.scan body": "watched",
    "lax.while_loop body": "watched",
    "jax.vmap(z + N)": "watched",
    # B15: both of these were `unwatched` on a WARM cache until the gate
    # started emptying jax's trace caches before the trace it watches. The
    # inline one is the reason that fix is an eviction and not a detector --
    # it leaves no nested jaxpr behind for a detector to see.
    "@jax.jit helper": "watched",
    "@jax.jit(inline=True) helper": "watched",
    # numpy narrows the value before any jax primitive runs, so the const-fold
    # rule is handed something already in range. Every route in this group is
    # one mechanism. `lax.full_like` was measured and added after the rest:
    # unwatched, VERIFIED, 0 fires, jaxpr `broadcast_in_dim -25536`. The prose
    # in `report.UNCOVERED` covered it under "anything else built on `full`"
    # the whole time, which is why the disclosure was adequate and the
    # inventory was not.
    "jnp.full(shape, N, dt)": "unwatched",
    "jnp.full_like(x, N)": "unwatched",
    "lax.full(shape, N, dt)": "unwatched",
    "lax.full_like(x, N)": "unwatched",
    "lax.convert_element_type(N, dt)": "unwatched",
    "np.asarray(N).astype(dt)": "unwatched",
    # B16. THE SECOND NAMED ROUTE INTO THE SAME RESIDUE, and it is here
    # because the residue is what the eager detector CANNOT close and a
    # residue with one named member reads like an edge case. numpy builds the
    # array at its own default width and narrows it before jax is reached, so
    # neither instrument sees a written constant: the const-fold rule is
    # handed -25536 and the eager detector is handed an operand whose value
    # was already destroyed. Measured on 0.11.0 and 0.10.2, with the eager
    # detector armed: 0 fires, VERIFIED, and the value still wraps.
    "jnp.asarray(np.array(N), dt)": "unwatched",
    "jnp.stack([x, jnp.full(N)])": "unwatched",
    # B8c fixup: `lax.select`-of-`full` narrows inside jax at the `full` line
    # like the six above it, so the const-fold rule is handed an in-range
    # value and the gate certifies a program the source does not describe.
    # It was DRIVEN CLOSED by the eager detector and disclosed as closed in
    # `design/eager-truncation-detector.md` while being a row of neither
    # inventory -- a claim with no guard behind it, which is why it is here.
    "lax.select(c, jnp.full(N), x)": "unwatched",
    # jnp.array and friends VALIDATE the Python int against the dtype and
    # raise. The contrast with jnp.full, three lines up, is jax's and not
    # stelling's, and it is the single most useful thing on this page for a
    # reader deciding how to write a constant.
    "jnp.array(N, dtype=dt)": "loud",
    "jnp.asarray(N, dtype=dt)": "loud",
    "jnp.int16(N)": "loud",
    # the written 40000 is in the jaxpr; the narrowing is a runtime convert
    "x // N": "deferred",
    "x % N": "deferred",
    "jnp.where(c, N, x)": "deferred",
    "jnp.clip(x, 0, N)": "deferred",
    "jnp.pad(x, 1, constant_values=N)": "deferred",
    # B8c fixup, and the bucket is NOT the one it was enrolled for.
    # `jnp.take`'s `fill_value` was disclosed beside `lax.select`-of-`full` as
    # a route the eager detector closes, which invited reading it as a hole
    # the detector plugs. Driven here in three spellings -- over the traced
    # `x`, over a `jnp.zeros` of `x`'s shape, and over a constant `jnp.zeros`
    # -- all three put the written 40000 into the jaxpr INTACT, so the gate
    # has nothing to see and correctly sees nothing. It is `deferred` and it
    # was never one of the gate's holes. WHAT DECLINES IT IS NOT THE
    # TRANSFER THAT DECLINES THE FIVE ABOVE: the constant arrives as
    # `gather`'s own `fill_value` parameter, so there is no
    # `convert_element_type` in this jaxpr to decline (measured), and the
    # note the verdict carries is the definite out-of-bounds index on that
    # `gather`. `DEFERRED_CATCHER` declares that per row and the test reads
    # the notes against it. See EAGER_COVERAGE for the other half.
    "jnp.take(x, i, fill_value=N)": "deferred",
}

#: WHAT DECLINES EACH `deferred` ROUTE, one row per row of the bucket, as a
#: fragment of the note the verdict actually carries.
#:
#: THE BUCKET USED TO ASSERT ITS CATCHER IN PROSE, and the prose was already
#: false. *"The propagation's `convert_element_type` transfer declines
#: them"* stood in **SIX FILES** -- the unit is FILES, and the count is the
#: one measured by
#: `test_every_page_that_says_what_declines_the_deferred_bucket_names_them_all`
#: below, re-run over `68b219d`: `SOUNDNESS.md`,
#: `docs/overflow-tripwire.md`, `design/eager-truncation-detector.md`, this
#: file, `src/stelling/_tripwire/report.py` and
#: `src/stelling/_tripwire/_adapter_jax.py` -- while `jnp.take`'s row has
#: no `convert_element_type` in its jaxpr at all. It survived because the
#: assertion beside the docstring was `status != VERIFIED`, which is
#: strictly weaker than the sentence: ANY refusal, for any reason, passes
#: it. A bucket whose reason for not being a hole is written in a comment
#: is a bucket with no reason at all, so the reason is declared here and
#: read out of the verdict.
#:
#: THE UNIT IS STATED BECAUSE THIS SENTENCE MISCOUNTED ITSELF. It read *"in
#: six places"* until 2026-08-21, meaning PASSAGES, and six was the earlier
#: enumeration of six mixed items -- two code comments, two prose pages and
#: two routed blocks -- restated unchanged after the sweep beside it had
#: already found more. `898158c` corrected the four files outside `src/`
#: and left the two inside it, one of them `report.EAGER_UNCOVERED`'s
#: fourth bullet, which prints to the user on every armed run. Both are
#: corrected now and the partition above is what will name the next one.
#:
#: TWO MECHANISMS, and the difference is not cosmetic. The five convert rows
#: narrow at run time over a VARIABLE and the interval transfer refuses the
#: lossy conversion. `jnp.take`'s `fill_value` reaches the jaxpr as
#: `gather`'s own parameter -- measured, there is no `convert_element_type`
#: equation in it -- so there is nothing for that transfer to decline, and
#: what refuses the route is the definite out-of-bounds index the spelling
#: asks for. That index is load-bearing rather than incidental: it is what
#: makes the fill reachable. Driven with an in-bounds index the fill is
#: never selected, the program executes `[100, 0]`, and `check()` returns
#: VERIFIED -- correctly, because nothing wrapped.
DEFERRED_CATCHER = {
    "x // N": "'convert_element_type' declined this form",
    "x % N": "'convert_element_type' declined this form",
    "jnp.where(c, N, x)": "'convert_element_type' declined this form",
    "jnp.clip(x, 0, N)": "'convert_element_type' declined this form",
    "jnp.pad(x, 1, constant_values=N)": (
        "'convert_element_type' declined this form"
    ),
    "jnp.take(x, i, fill_value=N)": "OUT-OF-BOUNDS INDEX (definite) in 'gather'",
}


def _harness(body):
    def h():
        x = any_array((2,), DTYPE, ENVELOPE)
        assert_(body(x) < BOUND)

    return h


#: GATE_COVERAGE is a claim about the DEFAULT path -- what `check()` does in
#: the environment every user gets -- so the routes below have to be allowed
#: to narrow even when the whole session was run with
#: `--stelling-eager-truncation=error`. Declaring it here rather than skipping
#: the file keeps the inventory measuring the same thing in both worlds, and
#: the permission is counted and named in the eager report like any other.
_DEFAULT_PATH = (
    "GATE_COVERAGE is a measurement of the DEFAULT path; these routes are "
    "driven precisely to see which of them narrow in silence"
)


def _bucket_once(body):
    try:
        with _eager.expected_truncation(_DEFAULT_PATH):
            verdict = check(_harness(body), vacuity_mode="inputs-only")
    except OverflowError:
        return "loud", None
    gated = (
        verdict.status == "UNKNOWN"
        and bool(verdict.notes)
        and "trace unfaithful" in verdict.notes[0]
    )
    if gated:
        return "watched", verdict
    # not gated: did the written constant survive into the traced program?
    from stelling._jax_compat import trace as _trace

    _tripwire.disarm()
    try:
        with _eager.expected_truncation(_DEFAULT_PATH):
            survives = str(OVER) in str(_trace(_harness(body)))
    finally:
        _tripwire.arm()
    return ("deferred" if survives else "unwatched"), verdict


def _measure(name):
    """The bucket this route is in, decided by driving it TWICE.

    Once is not a measurement here, and that is the whole of B15 in one line:
    the first ``check()`` of a route with a jitted helper traces that helper
    cold and sees the narrowing; every later one finds the cache warm and sees
    nothing. A single-call inventory would have recorded ``watched`` for a
    route that is watched exactly once per process, which is indistinguishable
    from covered right up until it matters.

    So the answer is the SECOND call's, and a route whose two calls disagree
    gets its own bucket rather than being averaged into silence.

    WHAT THE SECOND CALL IS, EXACTLY: a REGRESSION DETECTOR FOR THE EVICTION,
    and not an independent control on the bucket. On this tree ``check()``
    clears jax's trace caches itself, so both calls trace cold and therefore
    always agree — the ``unstable:`` bucket is unreachable while the eviction
    is in place, and a reading of "the two calls agreed, so the bucket is
    real" would be reading a tautology. What it does detect is the eviction
    going away: driven against ``a759809``'s ``src`` it reports
    ``'@jax.jit helper': declared 'watched', measured
    'unstable:watched->unwatched'`` for both jit routes. That is worth having;
    it is just not the thing a control would be.
    """
    body = ROUTES[name]
    first, _ = _bucket_once(body)
    second, verdict = _bucket_once(body)
    if first != second:
        return f"unstable:{first}->{second}", verdict
    return second, verdict


def test_the_declared_gate_coverage_is_the_measured_gate_coverage():
    """The inventory, held against what the routes actually do.

    An unwatched door that nobody enumerated is how B15 got here: the gate
    read its own silence as evidence of a clean trace, and no test anywhere
    asked which doors that silence covered.
    """
    measured = {name: _measure(name)[0] for name in ROUTES}
    assert measured == GATE_COVERAGE, (
        "the trace gate's coverage moved.\n"
        + "".join(
            f"  {k!r}: declared {GATE_COVERAGE.get(k)!r}, measured {v!r}\n"
            for k, v in sorted(measured.items())
            if GATE_COVERAGE.get(k) != v
        )
        + "Update GATE_COVERAGE *and* say in its comment why the new bucket "
        "is the right one. A route moving into `unwatched` is a hole opening "
        "and needs report.UNCOVERED updated in the same commit."
    )


def test_every_unwatched_route_really_certifies_a_destroyed_constant():
    """The `unwatched` bucket is only worth naming if it is really a hole.

    Both halves, because "the gate did not fire" is also what a route with
    nothing to catch produces: the verdict says VERIFIED, and the program jax
    actually executes violates the very obligation that VERIFIED discharged.
    """
    unwatched = [k for k, v in GATE_COVERAGE.items() if v == "unwatched"]
    assert unwatched, "the inventory declares no holes, so it proves nothing"
    x = jnp.array([0, 100], DTYPE)
    for name in unwatched:
        bucket, verdict = _measure(name)
        assert bucket == "unwatched", name
        assert verdict.status == "VERIFIED", (
            f"{name} is declared a silent hole but did not certify anything, "
            f"so the disclosure names the wrong consequence: {verdict.status}"
        )
        with _eager.expected_truncation(_DEFAULT_PATH):
            executed = np.asarray(jax.jit(ROUTES[name])(x))
        assert (executed < BOUND).all(), (
            f"{name}: the executed program does NOT satisfy the obligation "
            f"the wrapped one does, so this route is not the hole described"
        )
        assert int(executed.max()) < OVER, (
            f"{name}: {OVER} survived execution, so nothing was destroyed "
            "and this route does not belong in the unwatched bucket"
        )


def test_every_deferred_route_is_declined_by_the_mechanism_it_declares():
    """`deferred` says the gate ignores it. Something else must not — and
    WHICH something is declared per route and read out of the verdict.

    The written constant reaches the jaxpr, so there is no transcription loss
    for the trace gate to report — but the program still wraps at run time,
    and a bucket that meant "the gate ignores it and so does everyone else"
    would be a hole wearing a reassuring name.

    **THIS DOCSTRING USED TO BE THE ONLY PLACE THE CATCHER WAS NAMED, AND
    THE NAME WENT STALE UNDER IT.** It read *"Measured: the propagation's
    `convert_element_type` transfer declines the form"* while the assertion
    was `status != VERIFIED` — strictly weaker, and passed by ANY refusal
    for ANY reason. So when `jnp.take`'s `fill_value` joined the bucket with
    no `convert_element_type` in its jaxpr at all, the sentence became false
    of one of this test's own rows and the test stayed green. The name of
    this function was the same claim and went stale with it; it was
    `test_every_deferred_route_is_caught_by_the_transfer_instead`.

    The mechanism is DECLARED now, in :data:`DEFERRED_CATCHER`, and this
    holds the verdict's own notes to it. Driven: declaring
    `'convert_element_type' declined this form` for `jnp.take`'s row turns
    this red. `status != VERIFIED` stays as the weaker half, because "no
    note says what was declared" and "nothing declined this at all" are
    different failures and both should be legible.

    **IT IS `notes[0].startswith(...)` AND NOT `any(catcher in note)`, AND
    THAT WAS A REAL HOLE.** `in`-over-`any` is satisfied by any note in the
    verdict, including notes DOWNSTREAM of the decline and the ⊤-summary
    note that lists every primitive that fell. Driven, as shipped with
    `any(...)`, one mutation at a time: declaring `'div' declined this
    form` for `x // N`, `'add' declined this form` for `x % N`, `pad ×1`
    for `jnp.pad` and `⊤` — ONE CHARACTER, present in every note of every
    row — for all six were **`1 passed`** each. All four are `1 failed`
    against `notes[0].startswith(...)`, which is what "this is the
    mechanism that declines the route" actually says: the ROOT decline, not
    something the run mentions later.

    **A MEASURED LIMIT, because the note this reads is not always the one
    that makes the verdict non-VERIFIED.** `jnp.clip(x, 0, N)` declines
    TWICE — the conversion of the literal `0` at `lax_numpy.py:3408` and of
    `N` at `:3410` — and `notes[0]` is the first. Driven on jax 0.11.0,
    x64 on: `jnp.clip(x, 0, 100)`, where nothing wraps at all, produces the
    same two notes and the same UNKNOWN; and `jnp.clip(x, jnp.int16(0), N)`,
    which removes only the `0`-conversion, returns VERIFIED with the
    `N`-conversion note still attached. So this row's evidence is a real
    decline of a real conversion in the traced program, and it is not
    evidence about the written constant. `jnp.take` has the same shape at
    `fill_value=100`. Recorded rather than forced: what closes it is a
    note that names the VALUE, which is a change to the propagation's
    messages and not to this test.
    """
    deferred = [k for k, v in GATE_COVERAGE.items() if v == "deferred"]
    assert deferred, "no deferred routes, so this claim is vacuous"
    assert set(DEFERRED_CATCHER) == set(deferred), (
        "`DEFERRED_CATCHER` is a declaration per `deferred` row, not a list "
        "of the rows someone remembered. Declares a catcher and is not in "
        f"the bucket: {sorted(set(DEFERRED_CATCHER) - set(deferred))}; in "
        f"the bucket and declares none: "
        f"{sorted(set(deferred) - set(DEFERRED_CATCHER))}. A route admitted "
        "to `deferred` without naming what declines it is the bucket back "
        "to asserting a catcher it may not have."
    )
    for name in deferred:
        bucket, verdict = _measure(name)
        assert bucket == "deferred", name
        assert verdict.status != "VERIFIED", (
            f"{name} is declared covered by {DEFERRED_CATCHER[name]!r} "
            f"rather than by the gate, and NOTHING covered it: "
            f"{verdict.status}"
        )
        catcher = DEFERRED_CATCHER[name]
        assert verdict.notes and verdict.notes[0].startswith(catcher), (
            f"{name} declares that {catcher!r} is what declines it and the "
            f"verdict's FIRST note does not say so. The verdict is "
            f"{verdict.status}, so SOMETHING refused this route — but the "
            f"root decline is not the mechanism this bucket's account of it "
            f"names, which makes that account prose about a different "
            f"program. It is the first note and not any note on purpose: "
            f"`any(catcher in note …)` is passed by a downstream decline, "
            f"and by `⊤`, which stands in every note of every row. Notes: "
            f"{[note[:110] for note in verdict.notes]}"
        )


#: Routes whose key here and spelling in ``report.UNCOVERED`` differ. ONLY
#: spelling belongs in this table: the test below requires a line in
#: ``report.UNCOVERED`` for EVERY ``unwatched`` route in ``GATE_COVERAGE``,
#: and a route with no entry here must be named under its own key. Adding a
#: route to this table instead of to the report would be the same evasion in
#: a new place, so it is held to routes that exist and is read against
#: ``GATE_COVERAGE`` below.
UNCOVERED_SPELLING = {
    "jnp.stack([x, jnp.full(N)])": "jnp.stack([x, jnp.full(shape, N, dt)])",
    "jnp.asarray(np.array(N), dt)": "jnp.asarray(np.array(N), dtype=dt)",
    "lax.select(c, jnp.full(N), x)": "lax.select(p, jnp.full(shape, N, dt), x)",
}


def test_every_unwatched_route_is_named_in_the_reports_coverage_claim():
    """Prose and measurement cannot drift while this holds.

    The report is what a user reads to answer "what does it not see". A door
    measured here and unnamed there is a door the reader is not told about.

    THIS ITERATES ``GATE_COVERAGE``, NOT A LIST TYPED BESIDE IT. It used to
    walk a six-entry dict literal, so ``GATE_COVERAGE``'s comment — "a route
    moving into `unwatched` ... needs a line in `report.UNCOVERED` in the
    same commit, which the second test enforces" — described an enforcement
    nobody performed. Measured: adding ``lax.full_like(x, N)`` as a seventh
    ``unwatched`` row, named nowhere in ``report.UNCOVERED``, passed all
    seven tests in this file. A hardcoded copy of a set cannot police the
    set.
    """
    text = " ".join(report.UNCOVERED)
    unwatched = [k for k, v in GATE_COVERAGE.items() if v == "unwatched"]
    assert unwatched, "the inventory declares no holes, so this is vacuous"
    missing = [
        (name, UNCOVERED_SPELLING.get(name, name))
        for name in unwatched
        if UNCOVERED_SPELLING.get(name, name) not in text
    ]
    assert not missing, (
        "an `unwatched` route the report does not name — a hole the reader "
        "is not told about:\n"
        + "".join(f"  {n!r}: report.UNCOVERED has no {p!r}\n" for n, p in missing)
        + "Add the route to `report.UNCOVERED` in this commit. Add it to "
        "UNCOVERED_SPELLING only if the report already names it in different "
        "words."
    )
    stale = set(UNCOVERED_SPELLING) - {
        k for k, v in GATE_COVERAGE.items() if v == "unwatched"
    }
    assert not stale, (
        f"UNCOVERED_SPELLING carries routes that are no longer `unwatched` "
        f"({sorted(stale)}), so it is excusing something that is not there"
    )
    assert "GATE_COVERAGE" in text, (
        "the report does not point at this inventory, so a reader who wants "
        "the enumerated version has no way to find it"
    )


def test_the_warm_trace_cache_door_is_recorded_as_closed_for_the_GATE_only():
    """B15's door, and the three ways it is still open.

    `check()` evicts jax's trace caches, so a verdict's observation is
    complete — with respect to JAX's caches, on ONE thread, and no further.
    Three things sit outside that, each measured, and each of them is a place
    where "closed" would be the same over-claim in a new spelling:

    * the SESSION report has no such moment — it watches whatever the suite
      happens to trace, so a user's jitted function first traced in an
      earlier test is still never re-traced and never reported;
    * a value narrowed into a memo JAX DOES NOT OWN survives the eviction:
      `jax.extend.core.jaxpr_as_fun` over a saved jaxpr, a user
      `functools.lru_cache`, and `jax.closure_convert` (a public jax API)
      each return VERIFIED with 0 fires on a program whose 40000 is already
      -25536;
    * jax's cache is PROCESS-GLOBAL and the gate's counter is per-thread, so
      the eviction-to-trace window is not atomic: 0/400 wrong VERIFIED
      single-threaded against 247/400 with four competing threads.

    This asserts all four so that dropping any one of them goes red.
    """
    text = " ".join(report.UNCOVERED)
    assert "WARM TRACE CACHE" in text
    assert "inline=True" in text
    assert "`preconditions.check()`" in text and "jax.clear_caches()" in text
    assert "SINGLE-THREADED PROCESS" in text, (
        "the completeness claim is unqualified again; a process-global cache "
        "and a per-thread counter do not make one"
    )
    assert "session report" in text.lower()
    for phrase in (
        "jax.extend.core.jaxpr_as_fun(saved)",
        "functools.lru_cache",
        "jax.closure_convert",
    ):
        assert phrase in text, (
            f"{phrase} is a construct the eviction does not reach and it is "
            f"no longer disclosed"
        )
    assert "ANOTHER THREAD" in text and "247/400" in text, (
        "the thread-safety disclosure and its measurement are gone"
    )


def test_the_report_does_not_still_say_clear_caches_is_never_called():
    """A disclosure that B15 made false, held down where it was made false.

    The session report's arm-order line read *"`jax.clear_caches()` is NOT
    called -- that would change your suite's timing and behaviour to flatter
    a report"*. It is still true of the report and it stopped being true of
    the session the moment the gate started calling it, and a reader who
    budgets their suite's runtime off that sentence would be budgeting off a
    claim about a different program. The line now says which of the two it
    is talking about, and this asserts BOTH halves so that deleting either
    one goes red.
    """
    from stelling._tripwire import record, report

    text = " ".join(report.render(_tripwire.Status("armed"), record.Recorder()))
    assert "arm order" in text
    assert "THIS REPORT never calls `jax.clear_caches()`" in text, (
        "the report no longer says that IT does not clear caches"
    )
    assert "`preconditions.check()` DOES call it" in text, (
        "the report claims caches are never cleared, which the trace gate "
        "has made false for any session that calls check()"
    )
    assert "`contracts.check_contract()`" in text, (
        "check_contract() reaches the same `_pipeline` and evicts the same "
        "caches, so a disclosure that names only check() is narrower than "
        "the behaviour it describes"
    )


def test_the_inventory_discriminates():
    """Four buckets, each non-empty, and the predicate that sorts them is not
    a constant function."""
    buckets = set(GATE_COVERAGE.values())
    assert buckets == {"watched", "unwatched", "loud", "deferred"}
    for b in buckets:
        assert sum(1 for v in GATE_COVERAGE.values() if v == b) >= 3, b
    assert set(GATE_COVERAGE) == set(ROUTES), (
        "a route with no declared bucket, or a bucket with no route: the "
        "inventory and the probes must be the same set"
    )


# ===========================================================================
# B16: the same inventory, asked of the OPT-IN EAGER DETECTOR.
#
# A SECOND DECLARATION AND NOT A COLUMN ADDED TO THE FIRST, because the two
# answer different questions and only one of them is on by default.
# :data:`GATE_COVERAGE` is what ``check()`` does in the environment every user
# gets, and it must keep saying so; a route whose bucket moved because a
# non-default flag was passed would make the default inventory a claim about
# a program nobody runs. So ``GATE_COVERAGE`` is measured with the detector
# OFF, as before, and this is measured with it ON.
#
# WHAT IT BUYS, and it is the thing the `unwatched` bucket most needed: the
# bucket used to be a list of holes with a disclosure beside it. Now every row
# in it is either CLOSED by the opt-in detector or is one of the two named
# numpy routes, and the test below is what stops a third one being added to
# the residue quietly.
# ===========================================================================

#: What the eager detector does with each route, MEASURED by executing the
#: route with it armed. Three buckets and they are three different facts:
#:
#: ``raises``
#:     the detector refuses the construction: ``EagerTruncationError``, at the
#:     line that wrote the constant, before jax narrows anything.
#: ``loud``
#:     jax itself raises ``OverflowError``, exactly as it does with the
#:     detector off. These rows are the evidence that arming changes nothing
#:     about a route jax already refuses.
#: ``silent``
#:     nothing is raised. For a ``watched`` or ``deferred`` route that is
#:     correct and expected -- the constant is not destroyed at construction,
#:     so there is nothing here to see. For an ``unwatched`` one it is a
#:     residual hole, and :func:`test_the_unwatched_routes_the_eager_detector_
#:     cannot_close_are_the_two_named_numpy_ones` holds the residue to exactly
#:     the two routes ``report.EAGER_UNCOVERED`` names.
EAGER_COVERAGE = {
    # the six that narrow at `lax._convert_element_type` with the written
    # value still intact -- the whole point of the second instrument
    "jnp.full(shape, N, dt)": "raises",
    "jnp.full_like(x, N)": "raises",
    "lax.full(shape, N, dt)": "raises",
    "lax.full_like(x, N)": "raises",
    "lax.convert_element_type(N, dt)": "raises",
    "jnp.stack([x, jnp.full(N)])": "raises",
    # B8c fixup: driven FIRED, conv=1 trunc=1 with `jit` on and conv=2
    # trunc=1 with `JAX_DISABLE_JIT=1`, warm, on jax 0.11.0. A row now,
    # not prose.
    "lax.select(c, jnp.full(N), x)": "raises",
    # jax refuses these three itself, with or without the detector
    "jnp.array(N, dtype=dt)": "loud",
    "jnp.asarray(N, dtype=dt)": "loud",
    "jnp.int16(N)": "loud",
    # the inline door: the constant survives construction and dies in the
    # const-fold rule, which is the OTHER instrument's. Nothing for this one.
    "x + N": "silent",
    "N + x": "silent",
    "x - N": "silent",
    "x * N": "silent",
    "x < N": "silent",
    "x & N": "silent",
    "jnp.maximum(x, N)": "silent",
    "jnp.array(N).astype(dt)": "silent",
    "jnp.astype(jnp.array(N), dt)": "silent",
    "x.at[0].set(N)": "silent",
    "x.at[0].add(N)": "silent",
    "lax.cond branch": "silent",
    "lax.scan body": "silent",
    "lax.while_loop body": "silent",
    "jax.vmap(z + N)": "silent",
    "@jax.jit helper": "silent",
    "@jax.jit(inline=True) helper": "silent",
    # deferred: the written constant reaches the jaxpr and the narrowing is a
    # runtime convert over a VARIABLE, which is not a construction at all
    "x // N": "silent",
    "x % N": "silent",
    "jnp.where(c, N, x)": "silent",
    "jnp.clip(x, 0, N)": "silent",
    "jnp.pad(x, 1, constant_values=N)": "silent",
    # B8c fixup, and it is the ONE row that is `deferred` for the gate and
    # `raises` for the detector at the same time. Nothing is inconsistent
    # about that: under a TRACE the written 40000 reaches the jaxpr and the
    # narrowing is a run-time convert, so the gate has nothing to see; run
    # EAGERLY there is no trace to reach and the fill array is built at the
    # construction site, which is this instrument's. Measured `raises` here
    # and `deferred` in GATE_COVERAGE, and neither reading is the other's.
    "jnp.take(x, i, fill_value=N)": "raises",
    # THE RESIDUE. numpy finished before jax was reached.
    "np.asarray(N).astype(dt)": "silent",
    "jnp.asarray(np.array(N), dt)": "silent",
}


@pytest.fixture
def eager_armed():
    """Arm the eager detector for one test, and take it back out.

    ARMED PER-TEST AND NOT PER-MODULE, because it is a rule and not a report:
    while it is live, every construction in the process is subject to it,
    including the ones the fixtures above make. Its blast radius is bounded
    here to the tests that are measuring it.
    """
    from stelling import _tripwire as tw

    was_armed = _eager.is_armed()
    status = tw.arm_eager()
    if not status.armed:
        pytest.skip(f"the eager detector could not attach: {status.code}")
    yield status
    # RESTORE, do not disarm. A session run with
    # `--stelling-eager-truncation=error` armed this for the whole run, and a
    # fixture that took it out here would leave every later file unwatched
    # with nothing saying so.
    if not was_armed:
        tw.disarm_eager()


def _eager_bucket(name):
    """What executing this route does with the eager detector armed.

    EXECUTED CONCRETELY rather than through ``check()``, and the difference
    matters: the detector fires at CONSTRUCTION, which happens whether or not
    anything is tracing, so driving it through the harness would measure the
    harness. :func:`test_the_eager_detector_closes_the_hole_in_check_itself`
    is the one that asks the verdict path.
    """
    import stelling

    x = jnp.array([0, 100], DTYPE)
    try:
        ROUTES[name](x)
    except stelling.EagerTruncationError:
        return "raises"
    except OverflowError:
        return "loud"
    return "silent"


def test_the_declared_eager_coverage_is_the_measured_eager_coverage(eager_armed):
    """The second inventory, held against what the routes actually do.

    Same discipline as the first: moving a route between buckets is a change
    to what the tool can see and must be argued in ``EAGER_COVERAGE``'s
    comment, not merely made green. ``raises`` -> ``silent`` is a hole
    OPENING and is the failure this whole instrument is built to fail closed
    on -- though it should never get here, because the detector's own arm-time
    self-check drives every route it claims and refuses to attach when one
    goes blind.
    """
    measured = {name: _eager_bucket(name) for name in ROUTES}
    assert measured == EAGER_COVERAGE, (
        "the eager detector's coverage moved.\n"
        + "".join(
            f"  {k!r}: declared {EAGER_COVERAGE.get(k)!r}, measured {v!r}\n"
            for k, v in sorted(measured.items())
            if EAGER_COVERAGE.get(k) != v
        )
        + "Update EAGER_COVERAGE *and* say in its comment why the new bucket "
        "is the right one."
    )
    assert set(EAGER_COVERAGE) == set(ROUTES), (
        "a route with no declared eager bucket, or a bucket with no route"
    )


#: The eager detector's coverage of the `unwatched` bucket, as ONE pair of
#: numerals. `test_the_unwatched_routes_…` holds the measured buckets to
#: them and `test_the_documented_fraction_is_the_measured_one` holds every
#: file that states the fraction to them, so the sentence in the documents
#: and the dict in this file cannot drift apart in either direction.
_CLOSED = 7
_UNWATCHED = 9

#: The fraction as a document writes it: a numeral, `of the`, a numeral,
#: `unwatched`. Tolerant of markdown emphasis, of wrapping and of case,
#: because the six files that state it spell it six different ways.
#:
#: A QUOTED occurrence is skipped by `_live_fractions`, because a project
#: that records what it got wrong has to be able to write the wrong
#: sentence down. Quoting is the shape the retraction takes everywhere it
#: appears here, and it is checked as a shape rather than trusted: an
#: unquoted stale fraction fails wherever it stands.
_FRACTION_RE = re.compile(
    r"(?P<num>[A-Za-z]+|\d+)\s*\**\s*of\s+the\s*\**\s*"
    r"(?P<den>[A-Za-z]+|\d+)\**\s+`*unwatched`*",
    re.S,
)

#: THE CENSUS, the other shape the same numbers are written in:
#: `N routes -- 17 `watched`, 9 `unwatched`, 3 `loud`, 6 `deferred``. It is
#: a different sentence from the fraction and a check on one is not a check
#: on the other -- driven in the B8c fixup, enrolling `lax.select`-of-`full`
#: moved the fraction in every file that stated it AND left the census
#: stale in `SOUNDNESS.md`'s `SF-0.2.0-07` block, which the fraction
#: pattern does not match. Both shapes are read now.
_CENSUS_RE = re.compile(
    r"(?P<total>\d+)\s*\**\s*(?:constant-)?construction\s+routes|"
    r"(?P<total2>\d+)\s+routes\s*[,—-]",
    re.S,
)
_BUCKET_RE = re.compile(
    r"(?P<n>\d+)\s*\**\s*`*(?P<bucket>watched|unwatched|loud|deferred)`*",
    re.S,
)

_NUMERALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

#: Every file that states the CENSUS. A partition, like the fraction's.
_CENSUS_SITES = (
    "SOUNDNESS.md",
    "docs/overflow-tripwire.md",
    "design/eager-truncation-detector.md",
)

#: Every file that states the fraction. This is a partition and not a
#: reminder list: `test_the_documented_fraction_is_the_measured_one`
#: requires each of these to state it AND requires no other file in the
#: tree to state it, so adding the sentence somewhere new fails until it
#: is listed here, and deleting it from a listed file fails too.
#:
#: THIS FILE IS ON THE LIST. The message that used to name the sites named
#: six documents, included `SOUNDNESS.md` (which carried no such sentence
#: before the 0.2.0 routing put one there) and omitted this file, which
#: states the fraction in `test_the_unwatched_routes_…`'s own docstring —
#: so *"move all six"* would have left the instruction's own file stale.
#: `CHANGELOG.md` LEFT THIS LIST in the same commit that widened it, and
#: that is the routing working rather than a site going quiet: the Mode 2
#: detail moved into `SOUNDNESS.md` under §8.3, so the ledger states the
#: fraction and the changelog links to it. The partition below is what
#: made the move announce itself.
_FRACTION_SITES = (
    "SOUNDNESS.md",
    "docs/overflow-tripwire.md",
    "docs/quickstart.md",
    "design/eager-truncation-detector.md",
    "src/stelling/_tripwire/eager.py",
    "tests/test_tripwire_gate_coverage.py",
)

#: THE MECHANISM TOKENS, DERIVED FROM `DEFERRED_CATCHER` and not typed
#: beside it: the jax primitive named inside each declared catcher
#: (`'convert_element_type' declined this form` -> `convert_element_type`;
#: `OUT-OF-BOUNDS INDEX (definite) in 'gather'` -> `gather`). A third
#: mechanism joining the bucket adds its own token here on the day the row
#: is declared, and every page below has to name it in the same commit.
_MECHANISM_TOKENS = tuple(sorted({
    tok
    for catcher in DEFERRED_CATCHER.values()
    for tok in re.findall(r"'([a-z_]+)'", catcher)
}))

#: How far from the word `deferred` a mechanism may stand and still be part
#: of the same passage. A paragraph in this repository is 300-500
#: characters, and the FILE SET this window selects is identical at 320,
#: 400, 500 and 600 measured over the whole tree — so the number is a
#: paragraph rather than a threshold tuned until the answer came out right.
_MECHANISM_WINDOW = 400

#: Every file that tells a reader WHAT DECLINES a `deferred` route. A
#: partition, read exactly the way `_FRACTION_SITES` and `_CENSUS_SITES`
#: are — and it is here because the sweep-and-correct that should have been
#: this check MISSED TWO SITES, both of them in `src/` and one of them
#: `report.EAGER_UNCOVERED`'s fourth bullet, printed to the user on every
#: armed run. A correction applied by grepping is only as wide as the grep.
#:
#: WHAT THE PARTITION ENFORCES is that the bucket has MORE THAN ONE catcher
#: and a page that names one of them names all of them. Every page that
#: went false on 2026-08-21 went false the same way: it named
#: `convert_element_type` and nothing else, and then `jnp.take`'s
#: `fill_value` joined the bucket with no conversion in its jaxpr.
_DEFERRED_MECHANISM_SITES = (
    "SOUNDNESS.md",
    "design/eager-truncation-detector.md",
    "docs/overflow-tripwire.md",
    "src/stelling/_tripwire/_adapter_jax.py",
    "src/stelling/_tripwire/report.py",
    "tests/_soundness_routing_manifest.py",
    "tests/test_tripwire_gate_coverage.py",
)

#: A quoted run, which is how this repository writes a sentence it has
#: retracted -- in prose with `"..."`, and in
#: `tests/_soundness_routing_manifest.py` as a Python literal quoting a
#: source line the destination does NOT carry. Bounded so that two unrelated
#: quotes cannot swallow the text between them, and the single-quote form is
#: guarded by letter lookarounds so an apostrophe in `jax's` cannot open one.
_QUOTED = re.compile(
    r"\"[^\"\n]{0,400}\"|(?<![A-Za-z])'[^'\n]{0,400}'(?![A-Za-z])", re.S
)


def _live_fractions(text: str) -> list[tuple[str, str]]:
    """The fraction as this file ASSERTS it — quotations excluded.

    A retraction has to be able to quote the sentence it retracts:
    *"six of the SEVEN unwatched routes"* is written down in several
    places on purpose, and a check that could not tell it from a live
    claim would force the project to stop recording its own errors.
    """
    quoted = [m.span() for m in _QUOTED.finditer(text)]
    out = []
    for m in _FRACTION_RE.finditer(text):
        s, e = m.span()
        if any(qs <= s and e <= qe for qs, qe in quoted):
            continue
        out.append((m.group("num"), m.group("den")))
    return out


def found_censuses() -> dict[str, list[tuple[int, dict[str, int]]]]:
    """`{file: [(total, {bucket: n}), …]}` for every live census in the tree.

    A census is a sentence of the form `N construction routes — 17
    `watched`, 9 `unwatched`, 3 `loud`, 6 `deferred``, and it is a
    DIFFERENT sentence from the fraction: the fraction names two of the
    buckets and the census names the size of the dict and all four. A
    check on one is not a check on the other, which is exactly how
    `SOUNDNESS.md`'s `SF-0.2.0-07` block kept `33 … 8 unwatched` while
    every stated fraction moved to seven of nine.
    """
    out: dict[str, list[tuple[int, dict[str, int]]]] = {}
    for rel, text in read_text_files():
        quoted = [q.span() for q in _QUOTED.finditer(text)]
        for m in _CENSUS_RE.finditer(text):
            if any(qs <= m.start() and m.end() <= qe for qs, qe in quoted):
                continue
            total = int(m.group("total") or m.group("total2"))
            tail = text[m.end():m.end() + 260]
            buckets = {
                b.group("bucket"): int(b.group("n"))
                for b in _BUCKET_RE.finditer(tail)
            }
            if len(buckets) == 4:
                out.setdefault(rel, []).append((total, buckets))
    return out


def _check_the_census(found):
    measured = {"total": len(GATE_COVERAGE)}
    for bucket in ("watched", "unwatched", "loud", "deferred"):
        measured[bucket] = sum(1 for v in GATE_COVERAGE.values() if v == bucket)
    assert set(found) == set(_CENSUS_SITES), (
        f"the `GATE_COVERAGE` census is stated in {sorted(found)} and "
        f"`_CENSUS_SITES` lists {sorted(_CENSUS_SITES)}. States it and is "
        f"not listed: {sorted(set(found) - set(_CENSUS_SITES))}; listed and "
        f"no longer states it: {sorted(set(_CENSUS_SITES) - set(found))}."
    )
    wrong = [
        (rel, total, buckets)
        for rel, hits in sorted(found.items())
        for total, buckets in hits
        if total != measured["total"]
        or any(buckets[b] != measured[b] for b in buckets)
    ]
    assert not wrong, (
        f"`GATE_COVERAGE` holds {measured} and these censuses say "
        f"otherwise: {wrong}. The census and the fraction are two "
        f"sentences over one dict, and moving one without the other is how "
        f"`SF-0.2.0-07` kept a stale 33/8 through a commit that corrected "
        f"the fraction in six files."
    )


def test_the_documented_fraction_is_the_measured_one():
    """THE PROSE IS READ. That is the whole of this check, and it is new.

    *"Six of the SEVEN unwatched routes"* stood in six shipped files from
    ``fc98241`` until 2026-08-20 while ``GATE_COVERAGE`` held eight, and
    the reason it survived is not that a Python assertion was one-sided —
    ``len(closed) == 6`` and ``residue == {two}`` already entailed
    ``len(unwatched) == 8`` between them. It survived because **no test
    read the sentence.** Driven at ``de80ad8``: revert the fraction to
    *"seven"* in all six prose sites and 419 tests over every suite that
    opens any of those files still pass.

    So this reads them. Both halves of the fraction, in every file that
    states it, against the two numerals the buckets are held to — and in
    BOTH directions, because a list of sites is only as wide as its list:
    a file that states the fraction and is not listed fails here as loudly
    as a listed file that has stopped stating it.
    """
    unwatched = {k for k, v in GATE_COVERAGE.items() if v == "unwatched"}
    residue = {k for k in unwatched if EAGER_COVERAGE[k] == "silent"}
    closed = unwatched - residue
    assert (len(closed), len(unwatched)) == (_CLOSED, _UNWATCHED), (
        "the declared numerals do not match the buckets; "
        "test_the_unwatched_routes_… says the same thing with the detail"
    )

    found: dict[str, list[tuple[str, str]]] = {}
    for rel, text in read_text_files():
        hits = _live_fractions(text)
        if hits:
            found[rel] = hits

    assert set(found) == set(_FRACTION_SITES), (
        f"the eager detector's `unwatched` fraction is stated in "
        f"{sorted(found)} and `_FRACTION_SITES` lists "
        f"{sorted(_FRACTION_SITES)}. States it and is not listed: "
        f"{sorted(set(found) - set(_FRACTION_SITES))}; listed and no longer "
        f"states it: {sorted(set(_FRACTION_SITES) - set(found))}. A page "
        f"list is only as wide as its list, so this is a partition: a new "
        f"site has to be added here in the same commit, and a site that "
        f"drops the sentence fails rather than passing as compliance."
    )

    _check_the_census(found_censuses())

    wrong = []
    for rel, hits in sorted(found.items()):
        for num, den in hits:
            n = int(num) if num.isdigit() else _NUMERALS.get(num.lower())
            d = int(den) if den.isdigit() else _NUMERALS.get(den.lower())
            if (n, d) != (_CLOSED, _UNWATCHED):
                wrong.append((rel, num, den))
    assert not wrong, (
        f"the eager detector closes {_CLOSED} of the {_UNWATCHED} "
        f"`unwatched` routes and these sites say otherwise: {wrong}. This "
        f"is the sentence that read *\"six of the SEVEN\"* against a dict "
        f"holding eight, in six files at once, for as long as it took "
        f"someone to do the arithmetic in it."
    )


def found_deferred_mechanism_sites() -> dict[str, set[str]]:
    """`{file: {mechanism tokens named beside a `deferred`}}`, tree-wide.

    A file counts as describing the bucket's mechanism when one of
    `_MECHANISM_TOKENS` stands within `_MECHANISM_WINDOW` characters of the
    word `deferred`, and the value is the union of the tokens its passages
    name. Quotations are NOT excluded here, unlike `_live_fractions`: a
    file that quotes a mechanism at all is a file a reader can take a
    mechanism from, and the two files that quote one — this one, in the
    history it records, and `tests/_soundness_routing_manifest.py`, in the
    source line `SF-0.2.0-07` did not carry — both name the other
    mechanism as well, which is the whole requirement.
    """
    out: dict[str, set[str]] = {}
    for rel, text in read_text_files():
        named: set[str] = set()
        for m in re.finditer(r"deferred", text):
            window = text[
                max(0, m.start() - _MECHANISM_WINDOW):
                m.end() + _MECHANISM_WINDOW
            ]
            named |= {tok for tok in _MECHANISM_TOKENS if tok in window}
        if named:
            out[rel] = named
    return out


def test_every_page_that_says_what_declines_the_deferred_bucket_names_them_all():
    """The bucket has TWO catchers, and a page may not name one of them.

    This is `_FRACTION_SITES`' idiom pointed at the sentence that went
    false on 2026-08-21: *"the propagation's `convert_element_type`
    transfer declines them"*, written of a bucket one of whose rows has no
    `convert_element_type` in its jaxpr at all.

    **IT IS HERE BECAUSE THE CORRECTION WAS A SWEEP AND THE SWEEP MISSED
    TWO SITES.** Measured over the whole tree at `68b219d` — every file
    putting a mechanism token within `_MECHANISM_WINDOW` of the word
    `deferred` — SIX FILES described the bucket's mechanism and all six
    named `convert_element_type` alone: `SOUNDNESS.md`,
    `docs/overflow-tripwire.md`, `design/eager-truncation-detector.md`,
    `tests/test_tripwire_gate_coverage.py`,
    `src/stelling/_tripwire/report.py` and
    `src/stelling/_tripwire/_adapter_jax.py`. `898158c` corrected the four
    outside `src/`. The two inside it survived — one of them
    `report.EAGER_UNCOVERED`'s fourth bullet, which prints to the user on
    every armed run — and are corrected in the commit that adds this test.

    Both directions, because a list is only as wide as its list: a file
    that starts describing the bucket's mechanism and is not listed fails
    here, and a listed file that stops describing it fails too. The
    strength of the second assertion is the DECLARATION's: while
    `DEFERRED_CATCHER` names one distinct mechanism there is nothing for a
    page to leave out, and it says two today.
    """
    assert all(
        re.findall(r"'([a-z_]+)'", catcher)
        for catcher in DEFERRED_CATCHER.values()
    ), (
        f"a declared catcher names no jax primitive in quotes, so "
        f"`_MECHANISM_TOKENS` cannot be derived from it and this check "
        f"would silently narrow: {sorted(DEFERRED_CATCHER.values())}"
    )
    found = found_deferred_mechanism_sites()
    assert set(found) == set(_DEFERRED_MECHANISM_SITES), (
        f"the `deferred` bucket's mechanism is described in "
        f"{sorted(found)} and `_DEFERRED_MECHANISM_SITES` lists "
        f"{sorted(_DEFERRED_MECHANISM_SITES)}. Describes it and is not "
        f"listed: {sorted(set(found) - set(_DEFERRED_MECHANISM_SITES))}; "
        f"listed and no longer describes it: "
        f"{sorted(set(_DEFERRED_MECHANISM_SITES) - set(found))}."
    )
    partial = {
        rel: sorted(set(_MECHANISM_TOKENS) - named)
        for rel, named in sorted(found.items())
        if set(named) != set(_MECHANISM_TOKENS)
    }
    assert not partial, (
        f"`DEFERRED_CATCHER` declares {len(set(DEFERRED_CATCHER.values()))} "
        f"distinct mechanisms, named by {list(_MECHANISM_TOKENS)}, and "
        f"these files describe the bucket while naming only some of them: "
        f"{partial}. A page that credits one mechanism with the whole "
        f"bucket is the sentence this check exists to refuse — it was true "
        f"of five of the six rows and printed as though it were true of "
        f"all of them."
    )


def test_the_unwatched_routes_the_eager_detector_cannot_close_are_the_two_named_numpy_ones():
    """The residue is EXACTLY two routes, and both are disclosed.

    This is the assertion that keeps the ``unwatched`` bucket from quietly
    growing a third member. Seven of the nine ``unwatched`` routes are closed
    by an opt-in flag and two remain, and the two that remain are the ones
    numpy destroys before jax is reached. A route added to ``unwatched``
    that the detector does not close has to be argued into
    ``report.EAGER_UNCOVERED`` here, in the same commit.

    THE FRACTION IN THE SENTENCE ABOVE IS READ, by
    ``test_the_documented_fraction_is_the_measured_one``, in this file and
    in the six shipped documents that state it. It is written in the same
    form as theirs on purpose: the instruction *"move the bucket and move
    all six"* used to stand in a message whose own file was the seventh
    site and was not on its list.

    **WHY THE FRACTION DRIFTED, STATED CORRECTLY THIS TIME.** The first
    account of this said the sentence survived because the test *"asserted
    the NUMERATOR alone"* and that *"the denominator is asserted now"*.
    That is wrong about the mechanism, and the correction matters because
    it points at the guard that was actually missing. ``residue`` is a
    SUBSET of ``unwatched`` by construction, and the pre-existing
    ``residue == {two named routes}`` already pinned ``|residue| = 2``; so
    ``len(closed) == 6`` already entailed ``len(unwatched) == 8``, and
    there is no state of ``GATE_COVERAGE`` in which the old assertions are
    green and ``len(unwatched) == 8`` is red. Adding it detected nothing.

    What was missing is a check that reads the PROSE. Driven at
    ``de80ad8``: reverting the fraction to *"six of the seven"* in all six
    prose sites gave **419 passed** over every suite that reads any of
    those files. Nothing had ever read the sentence, and asserting the
    denominator in Python did not change that.
    ``test_the_documented_fraction_is_the_measured_one`` is the check that
    reads it, and the numerals below are what it compares against.

    Reads the DECLARATIONS rather than re-measuring, on purpose: the
    measurement is the test above, and a second copy of it here would be a
    second thing to keep in step. What this asserts is that the two
    declarations AGREE with each other and with the prose.
    """
    unwatched = {k for k, v in GATE_COVERAGE.items() if v == "unwatched"}
    residue = {k for k in unwatched if EAGER_COVERAGE[k] == "silent"}
    assert residue == {
        "np.asarray(N).astype(dt)",
        "jnp.asarray(np.array(N), dt)",
    }, (
        f"the residue the eager detector cannot close is {sorted(residue)}. "
        "Two routes are disclosed in `report.EAGER_UNCOVERED` and in "
        "`report.UNCOVERED`'s pre-narrowed bullet; anything else here is a "
        "hole the reader is not told about."
    )
    closed = unwatched - residue
    assert (len(closed), len(unwatched)) == (_CLOSED, _UNWATCHED), (
        f"the eager detector closes {len(closed)} of {len(unwatched)} "
        f"`unwatched` route(s) and this file declares {_CLOSED} of "
        f"{_UNWATCHED}: closed={sorted(closed)}, "
        f"unwatched={sorted(unwatched)}. Those two numerals are the "
        f"fraction stated in every file of `_FRACTION_SITES`, and "
        f"`test_the_documented_fraction_is_the_measured_one` holds each of "
        f"them to these. Move the bucket and move the prose in the same "
        f"commit — the list is in this file and it includes this file."
    )
    text = " ".join(report.EAGER_UNCOVERED)
    for phrase in (
        "np.asarray(N).astype(dt)",
        "jnp.asarray(np.array(N), dtype=dt)",
        "immutable type attribute",
    ):
        assert phrase in text, (
            f"`report.EAGER_UNCOVERED` no longer names {phrase!r}, so the "
            "residue is undisclosed"
        )


def test_the_eager_detector_closes_the_hole_in_check_itself(eager_armed):
    """The verdict path, not just the construction.

    The ``unwatched`` bucket's whole cost is that ``check()`` returns VERIFIED
    about a program whose constant no longer exists -- which
    :func:`test_every_unwatched_route_really_certifies_a_destroyed_constant`
    drives and asserts. With the detector armed, that VERIFIED cannot be
    reached: the harness cannot finish being traced, so there is no verdict to
    be wrong. A refusal, not a better answer, and that is the honest shape --
    the tool has nothing true to say about a program it cannot read.
    """
    import stelling

    closed = [
        k
        for k, v in GATE_COVERAGE.items()
        if v == "unwatched" and EAGER_COVERAGE[k] == "raises"
    ]
    assert closed, "no closed routes, so this is vacuous"
    for name in closed:
        with pytest.raises(stelling.EagerTruncationError) as caught:
            check(_harness(ROUTES[name]), vacuity_mode="inputs-only")
        assert caught.value.written == OVER
        assert caught.value.to_dtype == str(np.dtype(DTYPE))
        assert caught.value.became == int(np.asarray(OVER).astype(DTYPE))


def test_the_default_path_is_BYTE_IDENTICAL_without_the_flag():
    """Mode 2 is opt-in, and this is what "opt-in" is asserted to mean.

    Not "the flag defaults to off" -- that is a claim about a parser. This
    asserts the thing a user cares about: with nothing armed, the private jax
    attribute the detector patches is jax's own function, and every route
    produces exactly the value it produced before this feature existed.

    The value column is derived from the two's-complement arithmetic rather
    than typed, so this cannot go green by being updated to match a
    regression.
    """
    import contextlib

    from stelling._tripwire import _adapter_jax as adapter

    @contextlib.contextmanager
    def detached():
        """The hook GONE, not the truncation permitted, and put back after.

        A region declaration would be the wrong tool: what is being measured
        is the program with nothing patched, and `expected_truncation` leaves
        the wrapper installed.
        """
        was_armed = _eager.is_armed()
        if was_armed:
            _tripwire.disarm_eager()
        try:
            yield
        finally:
            if was_armed:
                _tripwire.arm_eager()

    with detached():
        assert adapter.eager_live_check() == "detached"
        _the_default_path_body()


def _the_default_path_body():
    x = jnp.array([0, 100], DTYPE)
    for name, bucket in EAGER_COVERAGE.items():
        if bucket == "loud":
            with pytest.raises(OverflowError):
                ROUTES[name](x)
            continue
        # every other route completes, and completes by DESTROYING the
        # constant -- which is the state of the world this whole file
        # documents and the reason the detector exists
        out = np.asarray(ROUTES[name](x)).ravel().tolist()
        assert OVER not in out, (
            f"{name}: {OVER} survived execution with nothing armed, so the "
            "default path is not the one every other test on this page "
            "describes"
        )
