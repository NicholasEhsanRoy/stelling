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

**Controls, because an assertion that cannot go red proves nothing.**
:func:`test_control_the_probe_reports_a_raise_where_jax_already_refuses`
drives the same probe through ``jnp.array``, which raises for this
constant today — that is exactly the shape a jax fix at ``jnp.full`` would
take, and the probe reports it rather than passing. The in-range control
shows the wrap check is not satisfied by every constant, and the verdict
control re-drives the entry's own ``5`` row, which is UNKNOWN: the
VERIFIED asserted here is not something this harness shape returns
regardless.

Interval-only throughout: no solver is invoked, so this file says the
same thing in every environment that has jax.
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
    try:
        return int(np.asarray(door((), written, dtype)))
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
