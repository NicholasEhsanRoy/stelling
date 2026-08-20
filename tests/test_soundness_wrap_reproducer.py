# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""``SOUNDNESS.md``'s wrap reproducer is EXECUTED, and its wrap asserted.

The integer-literal wrap entry in ``SOUNDNESS.md`` is the one artefact in
this repository whose failure mode is a user's wrong VERIFIED. Until this
file, nothing ran it.

**What was and was not protected.** ``tests/test_doc_examples.py``
executes fenced blocks under ``README.md`` and ``docs/*.md``;
``SOUNDNESS.md`` is at the repository root, so its one ```python fence —
the four-line reproducer — was executed by nothing. Exactly one test read
``SOUNDNESS.md`` at all (``tests/test_release_doc_claims.py``, pinning a
single sentence about units, thousands of lines away). The defect's
EXISTENCE is guarded, by ``xfail(strict=True, raises=WrongVerifiedFromWrap)``
in ``tests/property/test_oracle.py`` — but that module gates on
``hypothesis``, a dev-group dependency absent from both shared jax venvs
and from the zero-dep CI job, so it is skipped at collection in every
environment this project routinely measures in. Nothing anywhere read the
entry's own reproducer.

**The direction that matters.** The entry is a disclosure of an OPEN
defect. The way it goes wrong is not that it breaks — it is that jax fixes
the wrap, the entry keeps saying stelling returns VERIFIED for a predicate
false at every declared point, and the page quietly becomes false while
the suite stays green. That is the failure this file converts into a red
test: if `jnp.full((), 256, jnp.int8)` ever raises, saturates, or
preserves its value, :func:`test_the_fence_still_wraps` goes red and the
entry gets rewritten instead of rotting.

**TWO DOORS, BECAUSE THE ENTRY NOW PRICES TWO AND THIS FILE WATCHED ONE.**
``SOUNDNESS.md`` records that the reproducer's constant, written through
``jnp.full``, dies at the ``type(operand) is int`` ``.astype`` in
``lax.py``, while the inline spelling ``x + 256`` dies somewhere else
entirely — in ``convert_element_type``'s CONSTANT-FOLDING rule — and that
neither of the two priced fixes reaches the other door. It calls those
"two prices for two doors". A jax that fixed only the folding site would
therefore leave this file's original five assertions **completely green**
while half of what the entry claims had become false. Measured, not
argued: with a range check installed on
``pe.const_fold_rules[convert_element_type_p]`` so that ``x + 256`` raises
and ``jnp.full((), 256, jnp.int8)`` still returns ``0``, this file ran
**11 passed, 0 failed** — the whole tripwire, blind. So
:func:`test_the_other_door_still_wraps` and
:func:`test_the_other_door_still_returns_a_wrong_VERIFIED` watch that door
too, and under the same simulation they are the tests that go red.

**It runs the fence, it does not restate it.** The reproducer is read out
of ``SOUNDNESS.md``, parsed, and executed — so a reproducer that stops
running, or that is edited into something this file no longer describes,
fails here. What is asserted about it is derived from the fence's own
literals (the written constant, the declared box, the comparison bound),
not typed in again.

**Measured red, not argued red.** A mutation battery replaced ``jnp.full``
with three versions of a jax that has fixed this — one that RAISES on the
out-of-range constant, one that SATURATES it to 127, one that PRESERVES it
by widening the dtype — and ran the two assertions above against each.
Six of six go red, each with a message naming which of the three happened;
unmutated, both pass. That is the property this file exists for, and it is
a measurement rather than a claim about what the assertions ought to do.

**And the two doors were measured against each other, which is what makes
them two.** Two simulated fixes, each installed alone, on jax 0.11.0 AND
jax 0.10.2, both ``x64`` cells:

* a range check on ``pe.const_fold_rules[convert_element_type_p]`` — jax
  fixed at the folding site only, so ``x + 256`` raises and
  ``jnp.full((), 256, jnp.int8)`` still returns ``0``: **4 failed, 13
  passed**, and the four are exactly this file's second-door cases;
* a range check on the ``type(operand) is int`` narrowing in
  ``lax._convert_element_type`` — jax fixed at the fence's door only, so
  ``jnp.full((), 256, jnp.int8)`` raises and ``x + 256`` still returns
  ``[0 0 0]``: **4 failed, 13 passed**, and the four are exactly the
  fence's cases.

Neither simulation is visible to the other door's tests. Before the
second door was added, the first of those two ran **11 passed, 0 failed** —
which is the reason it was added.

**Controls, because an assertion that cannot go red proves nothing.**
:func:`test_control_the_probe_reports_a_raise_where_jax_already_refuses`
drives the same probe through ``jnp.array``, which raises for this
constant today — that is exactly the shape a jax fix at ``jnp.full`` would
take, and the probe reports it rather than passing. The in-range control
shows the wrap check is not satisfied by every constant, and the verdict
control re-drives the entry's own ``5`` row, which is UNKNOWN: the
VERIFIED asserted here is not something this harness shape returns
regardless. The second door carries ONE of those two:
:func:`test_control_the_other_door_does_not_wrap_in_range`. It has NO
verdict control. What stands beside it inside the verdict test — the check
that the predicate really is false at every declared point before any
verdict is asked for — is a source-arithmetic precondition the fence's
verdict test also carries, so it is not the same control and does not do
the same work: a `make_verdict` that returned VERIFIED regardless reddens
the fence's control and nothing here. Measured, it is not vacuous — inline
``v + 5`` is UNKNOWN where ``v + 0`` and ``v + -1`` are VERIFIED — but that
is asserted by this sentence and executed by nothing.

**Both doors are derived from the fence, not typed in twice.** The second
door writes the SAME constant, at the same dtype, over the same declared
box, against the same bound — all four read out of ``SOUNDNESS.md``'s one
fence. Only the spelling differs, which is the whole point of it being a
second door.

Interval-only throughout: no solver is invoked, so this file says the
same thing in every environment that has jax.

WHAT THIS STILL DOES NOT WATCH, and it is the same shape as the gap the
second door was added to close, one spelling over. **A jax — or a NumPy —
that began range-checking NumPy SCALARS leaves every test here green.**
MEASURED by simulating exactly that at ``_convert_element_type`` for
``np.generic``, and again with the three construction doors included: 17
passed, 0 failed on both series, while the entry's ``LUT[7]`` constructions
go from VERIFIED to ``OverflowError`` at construction. That would silently
falsify the ``jnp.full`` + ``LUT[7]`` reproducer, the same VERIFIED through
``jnp.array``/``jnp.asarray``/``jnp.int8``, the 13-spelling table's "24
wrap silently", and the evidence under the withdrawn guard. The entry
already says what it does not know about a third NumPy; nothing here turns
red if that changes. Not a false claim today — an uncovered one.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SOUNDNESS = REPO / "SOUNDNESS.md"


# --------------------------------------------------------------- the fence


def _fence() -> str:
    """The reproducer, read out of ``SOUNDNESS.md``.

    The page holds exactly one ```python fence and this is it; that is
    asserted rather than assumed, because "the first fence" would silently
    follow a later one added above it.
    """
    lines = SOUNDNESS.read_text(encoding="utf-8").splitlines()
    opens = [i for i, ln in enumerate(lines) if ln.strip() == "```python"]
    assert len(opens) == 1, (
        f"SOUNDNESS.md holds {len(opens)} ```python fences, expected exactly "
        f"one (the wrap reproducer). If a second example landed, this file "
        f"has to say which fence it runs."
    )
    i = opens[0]
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip() == "```")
    return textwrap.dedent("\n".join(lines[i + 1 : j])) + "\n"


def _compile_fence(src: str):
    """Turn the fence into (eager statements, nullary harness).

    The fence is written as a user would write it: two statements that run
    eagerly — the constant and the jitted function — then a line that
    declares an input and states an obligation. ``trace`` wants a nullary
    harness returning its assert, so the statements from the first mention
    of ``any_array`` onward become one, with the trailing expression
    returned. Nothing is rewritten beyond that.
    """
    mod = ast.parse(src, filename="<SOUNDNESS.md fence>")
    split = next(
        (i for i, s in enumerate(mod.body) if "any_array" in ast.dump(s)), None
    )
    assert split is not None, (
        "the fence no longer declares an input with any_array; this file "
        "runs it as a stelling harness and cannot any more"
    )
    body = list(mod.body[split:])
    assert isinstance(body[-1], ast.Expr), (
        "the fence's last statement is no longer an expression, so there is "
        "no obligation for the harness to return"
    )
    body[-1] = ast.Return(value=body[-1].value)
    fn = ast.FunctionDef(
        name="_harness",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=body,
        decorator_list=[],
        type_params=[],
    )
    tree = ast.Module(body=list(mod.body[:split]) + [fn], type_ignores=[])
    ast.fix_missing_locations(tree)
    ns: dict = {
        "jax": jax,
        "jnp": jnp,
        "any_array": any_array,
        "assert_": assert_,
    }
    # A REGION DECLARATION, and this is the single site in the repository
    # where it is least optional. The fence IS the defect: its first line
    # writes 256 into an int8 and the whole page exists to record that jax
    # destroys it in silence. With `--stelling-eager-truncation=error` armed
    # that line raises -- correctly, because the detector is the fix for
    # exactly this -- and the reproducer must still RUN, because a disclosure
    # whose reproducer is only quoted is the state this file was written to
    # end. `stelling.intentional_wrap` cannot serve here at all: the fence is
    # read out of `SOUNDNESS.md` and executed verbatim, so there is no source
    # to edit, and editing it would be editing the defect out of the
    # disclosure.
    from stelling._tripwire.eager import expected_truncation

    with expected_truncation(
        "SOUNDNESS.md's wrap reproducer is executed verbatim; the truncation "
        "it performs is the defect this page discloses"
    ):
        exec(compile(tree, "<SOUNDNESS.md fence>", "exec"), ns)  # noqa: S102
    return ns


def _fence_facts(src: str) -> dict:
    """The numbers the fence itself writes: constant, dtype, box, bound."""
    mod = ast.parse(src, filename="<SOUNDNESS.md fence>")
    facts: dict = {}
    for node in ast.walk(mod):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name == "jnp.full" and len(node.args) >= 3:
            facts["written"] = ast.literal_eval(node.args[1])
            facts["dtype"] = ast.unparse(node.args[2]).rsplit(".", 1)[-1]
        elif name == "any_array" and len(node.args) >= 3:
            facts["decl_dtype"] = ast.literal_eval(node.args[1])
            facts["box"] = ast.literal_eval(node.args[2])
    for node in ast.walk(mod):
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.LtE):
            facts["bound"] = ast.literal_eval(node.comparators[0])
    missing = {"written", "dtype", "decl_dtype", "box", "bound"} - set(facts)
    assert not missing, (
        f"the fence no longer writes {sorted(missing)} where this file reads "
        f"them; the reproducer changed and this file must be updated with it"
    )
    return facts


# ------------------------------------------------------------- the probe


def _traced_constant(door, written, dtype):
    """What jax actually puts in the trace for ``written`` at ``door``.

    Returns the traced value, or the exception type if the door refuses.
    One function for the reproducer's door and for the controls, so that
    "wraps", "raises" and "keeps the value" are three answers from one
    instrument rather than three instruments.
    """
    from stelling._tripwire.eager import expected_truncation

    # The same argument as the fence's own execution: this instrument's whole
    # job is to report whether the door wraps, raises or keeps the value, so
    # the wrap has to be allowed to happen for it to be reported.
    with expected_truncation(
        "measuring whether a construction door wraps, raises or keeps the "
        "value -- the wrap is the measurement"
    ):
        try:
            return int(np.asarray(door((), written, dtype)))
        except OverflowError as exc:
            return type(exc)


def _inline_offset(written, dtype):
    """What `x + written` puts in the trace, at `dtype`. THE SECOND DOOR.

    The constant is closed over as a Python value, never passed as a jit
    argument — a jit argument is a tracer and nothing is folded, which is a
    different program from the one ``SOUNDNESS.md`` is about. ``x`` is
    zeros, so the value read back IS the narrowed constant.

    Returns the traced array, or the exception type if jax refuses.
    """

    def f(v):
        return v + written

    try:
        return jax.jit(f)(jnp.zeros((3,), dtype))
    except OverflowError as exc:
        return type(exc)


@pytest.fixture(params=[False, True], ids=["x64=0", "x64=1"])
def x64(request):
    """Both cells. `JAX_ENABLE_X64` is load-bearing everywhere else on this
    page, so the reproducer is not asserted in one of its two settings."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", old)


# -------------------------------------------------------------- the tests


def test_the_page_still_carries_the_reproducer_this_file_runs():
    src = _fence()
    for token in ("jnp.full", "jnp.int8", "any_array", "assert_", "jax.jit"):
        assert token in src, (
            f"SOUNDNESS.md's reproducer no longer contains {token!r}; this "
            f"file executes that fence and describes it in its docstring"
        )
    facts = _fence_facts(src)
    info = np.iinfo(getattr(np, facts["dtype"]))
    assert facts["written"] > info.max, (
        f"the fence's constant {facts['written']} is inside "
        f"{facts['dtype']}'s range [{info.min}, {info.max}], so there is no "
        f"out-of-range narrowing left for this file to be about"
    )


def test_the_fence_still_wraps(x64):
    """THE TRIPWIRE. If jax fixes this, the entry goes red here."""
    facts = _fence_facts(_fence())
    ns = _compile_fence(_fence())
    offset = ns["OFFSET"]
    bits = np.iinfo(getattr(np, facts["dtype"])).bits
    assert str(offset.dtype) == facts["dtype"], (
        f"jnp.full no longer produces {facts['dtype']} for the fence's "
        f"constant — it produced {offset.dtype}. jax has widened rather than "
        f"narrowed, which is a fix by another name: rewrite the entry."
    )
    assert int(offset) != facts["written"], (
        "jnp.full no longer narrows the fence's out-of-range constant. The "
        "SOUNDNESS.md entry this file guards describes an OPEN defect and is "
        "now describing something that does not happen: rewrite the entry."
    )
    assert int(offset) == facts["written"] % (2**bits), (
        f"the constant is still destroyed, but no longer by wrapping mod "
        f"2**{bits}: {facts['written']} became {int(offset)}. The entry's "
        f"framing is 'wraps mod 2**bits' and no longer describes this."
    )


def test_the_fence_still_returns_a_wrong_VERIFIED(x64):
    """The whole claim, end to end and interval-only: source-false at every
    declared point, VERIFIED anyway."""
    facts = _fence_facts(_fence())
    lo, hi = facts["box"]
    points = range(int(lo), int(hi) + 1)
    false_at = [p for p in points if not (p + facts["written"] <= facts["bound"])]
    assert list(points) and len(false_at) == len(list(points)), (
        "the fence's predicate is no longer false at every declared point in "
        "exact integer arithmetic, so a VERIFIED here would not be wrong"
    )

    ns = _compile_fence(_fence())
    closed = trace(ns["_harness"])
    p = propagate(closed)
    verdict = make_verdict(
        closed,
        p,
        stelling_version="(this tree)",
        jax_version=jax.__version__,
        precision_config=f"jax_enable_x64={x64}",
    )
    assert verdict.status == "VERIFIED", (
        f"the reproducer returns {verdict.status}, not VERIFIED. Either the "
        f"wrap is gone or the pipeline moved; SOUNDNESS.md's entry says "
        f"VERIFIED and must be re-driven either way."
    )


def test_the_other_door_still_wraps(x64):
    """THE TRIPWIRE, SECOND DOOR: `x + 256`, the inline spelling.

    ``SOUNDNESS.md`` prices this door separately from the fence's, at a
    separate site (the constant-folding rule), and says only the more
    expensive of its two candidate fixes reaches it. A jax that fixed the
    fence's door alone would leave this red; a jax that fixed this one
    alone would leave the fence's tests green. Neither may pass silently.
    """
    facts = _fence_facts(_fence())
    written = facts["written"]
    npdtype = getattr(np, facts["dtype"])
    bits = np.iinfo(npdtype).bits

    got = _inline_offset(written, getattr(jnp, facts["dtype"]))
    assert got is not OverflowError, (
        f"`x + {written}` at {facts['dtype']} now RAISES under jit. Something "
        f"has fixed the narrowing this door reaches — SOUNDNESS.md names the "
        f"constant-folding site as the one no cheap fix reaches, and prices "
        f"it at 1031 newly failing cases OF 16,798 — so the entry's 'two "
        f"prices for two doors' is no longer true and must be rewritten. Do "
        f"not assume the fix was at that site: this watches the BEHAVIOUR, "
        f"and any change on the inline path reddens it."
    )
    assert str(got.dtype) == facts["dtype"], (
        f"`x + {written}` no longer produces {facts['dtype']} — it produced "
        f"{got.dtype}. jax has widened rather than narrowed at this door, "
        f"which is a fix by another name: rewrite the entry."
    )
    value = int(np.asarray(got)[0])
    assert value != written, (
        f"`x + {written}` no longer destroys the constant at this door; it "
        f"came back as {value}. SOUNDNESS.md describes an OPEN defect at "
        f"two doors and this is one of them: rewrite the entry."
    )
    assert value == written % (2**bits), (
        f"the constant is still destroyed at this door, but no longer by "
        f"wrapping mod 2**{bits}: {written} became {value}. The entry's "
        f"framing is 'wraps mod 2**bits' and no longer describes this door."
    )


def test_the_other_door_still_returns_a_wrong_VERIFIED(x64):
    """The second door, end to end and interval-only. Same declared box,
    same bound, same constant as the fence — only the spelling differs."""
    facts = _fence_facts(_fence())
    lo, hi = facts["box"]
    written, bound = facts["written"], facts["bound"]
    points = list(range(int(lo), int(hi) + 1))
    false_at = [p for p in points if not (p + written <= bound)]
    assert points and len(false_at) == len(points), (
        "the inline spelling's predicate is no longer false at every "
        "declared point in exact integer arithmetic, so a VERIFIED here "
        "would not be wrong"
    )

    @jax.jit
    def shift(v):
        return (v + written).astype(jnp.float32)

    def harness():
        return assert_(
            shift(any_array((), facts["decl_dtype"], (lo, hi))) <= bound
        )

    try:
        closed = trace(harness)
    except OverflowError as exc:
        pytest.fail(
            f"tracing the inline spelling now RAISES ({exc}). jax has fixed "
            f"the narrowing at this door, so SOUNDNESS.md's claim that it "
            f"costs the same wrong VERIFIED as the fence's is no longer "
            f"true: rewrite the entry."
        )
    verdict = make_verdict(
        closed,
        propagate(closed),
        stelling_version="(this tree)",
        jax_version=jax.__version__,
        precision_config=f"jax_enable_x64={x64}",
    )
    assert verdict.status == "VERIFIED", (
        f"the inline spelling returns {verdict.status}, not VERIFIED. "
        f"SOUNDNESS.md says this door costs the same wrong VERIFIED as the "
        f"fence's and must be re-driven either way."
    )


def test_control_the_other_door_does_not_wrap_in_range(x64):
    """CONTROL for the second door: an in-range constant survives it, so
    `!= written` above is a real discrimination and not a tautology."""
    facts = _fence_facts(_fence())
    got = _inline_offset(5, getattr(jnp, facts["dtype"]))
    assert got is not OverflowError, "an in-range 5 was refused by `x + 5`"
    assert int(np.asarray(got)[0]) == 5, (
        f"an in-range 5 did not survive `x + 5`: {np.asarray(got).tolist()!r}"
    )


def test_control_the_probe_reports_a_raise_where_jax_already_refuses(x64):
    """CONTROL for the tripwire: the same probe, at a door where jax
    ALREADY does the thing a fix at `jnp.full` would do.

    `jnp.array(256, int8)` raises today. That is the shape the tripwire
    above is watching for, and the probe returns it as a refusal rather
    than as a wrap — so the tripwire's assertion is one this instrument can
    fail, not one it is built to pass.
    """
    facts = _fence_facts(_fence())

    def door(_shape, value, dtype):
        return jnp.array(value, dtype)

    got = _traced_constant(door, facts["written"], jnp.int8)
    assert got is OverflowError, (
        f"jnp.array no longer refuses the fence's constant (got {got!r}); "
        f"this control no longer demonstrates the refusing shape"
    )


def test_control_the_wrap_check_is_not_satisfied_by_every_constant(x64):
    """CONTROL: an in-range constant is not wrapped, so `!= written` in the
    tripwire is a real discrimination and not a tautology."""
    got = _traced_constant(jnp.full, 5, jnp.int8)
    assert got == 5, f"an in-range 5 did not survive jnp.full: {got!r}"


def test_control_the_VERIFIED_is_not_what_this_shape_always_returns(x64):
    """CONTROL: the entry's own `5` row. Same three lines, same box, only
    the constant changed — UNKNOWN, not VERIFIED. So the VERIFIED asserted
    above is a fact about the wrapped constant, not about the harness."""
    facts = _fence_facts(_fence())
    lo, hi = facts["box"]
    offset = jnp.full((), 5, jnp.int8)

    @jax.jit
    def shift(v):
        return (v + offset).astype(jnp.float32)

    def harness():
        return assert_(shift(any_array((), facts["decl_dtype"], (lo, hi))) <= facts["bound"])

    closed = trace(harness)
    verdict = make_verdict(
        closed,
        propagate(closed),
        stelling_version="(this tree)",
        jax_version=jax.__version__,
        precision_config=f"jax_enable_x64={x64}",
    )
    assert verdict.status == "UNKNOWN", (
        f"the in-range control returns {verdict.status}; SOUNDNESS.md's "
        f"control table records UNKNOWN for the `5` row, and the wrapped "
        f"row's VERIFIED means nothing if every row is VERIFIED"
    )
