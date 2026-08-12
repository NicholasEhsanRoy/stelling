# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Transfer-decline notes carry provenance, and every claim is measured.

docs/proposed-decline-messages.md #2: a decline that reports a box must
say where the box came from — which equation produced it, and whether it
originated in the user's declaration or in upstream propagation. The
propagator's decline notes now name the declining SITE and, per operand,
the recorded provenance: a quoted literal, the declaring equation, the
producing equation with its propagated span, or — for stelling's own
artifact ⊤ — the upstream equation that minted it, with its cause. The
"it did not come from your declaration" sentence is emitted ONLY for a
box minted by ``top_out``, where it is literally true.

Message-content changes only, pinned here: every decline still declines,
statuses and coverage counters are unchanged on every exercised path, and
the verbatim-duplicate collapse keeps the counters' multiplicity.

Hand-built IR throughout — no jax needed.
"""

from __future__ import annotations

from math import inf

from stelling import ir
from stelling.propagate import interval_env, propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
I64 = ir.Aval(kind="ShapedArray", shape=(), dtype="int64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi, src=()):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(
            ("shape", out.aval.shape),
            ("dtype", out.aval.dtype),
            ("lo", lo),
            ("hi", hi),
        ),
        source_info=src,
    )


def eqn(prim, ins, out, params=(), src=()):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(ins),
        outvars=(out,),
        params=tuple(params),
        source_info=src,
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)
        )
    )


def _sqrt_query(lo, hi):
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi, src=("decl.py:3 (h)",)),
            eqn("sqrt", [x], s, src=("h.py:12 (h)",)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _decline_notes(p):
    return [n for n in p.notes if "declined this form" in n]


# -- declaration-origin provenance --------------------------------------------


def test_sqrt_decline_names_the_declared_input_not_an_upstream_top():
    q = _sqrt_query(-1.0, 4.0)
    p = propagate(q)
    (note,) = _decline_notes(p)
    # the site of the declining equation itself
    assert "'sqrt' declined this form at h.py:12 (h):" in note
    # the quoted lower bound is the OPERAND's measured minimum, negative
    box = interval_env(q)[0]
    assert min(box.los) == -1.0 < 0.0
    assert f"lower bound {min(box.los)} is negative" in note
    # provenance: the operand IS the declaration, named with its site and
    # its span — the span is the declared box, quoted from the live env
    assert (
        "operand 0 is the declared input itself (declared at decl.py:3 (h)), "
        f"spanning [{box.los[0]}, {box.his[0]}]" in note
    )
    # a declaration-derived box must never be blamed on upstream ⊤
    assert "did not come from your declaration" not in note
    # the trimmed reason no longer misattributes the box to "the declared
    # box" as a phrase (the interval layer cannot know provenance), and the
    # design-posture prose is gone
    assert "pow domain posture" not in note
    # message-content-only pins: same decline, same accounting
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1 and p.coverage.known == 3
    assert "sqrt" not in dict(p.transfers_used)


def test_sqrt_decline_downstream_of_top_names_the_upstream_origin():
    x, t, s, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0, src=("decl.py:3 (h)",)),
            eqn("sin", [x], t, src=("s.py:9 (f)",)),
            eqn("sqrt", [t], s, src=("h.py:12 (h)",)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    (note,) = _decline_notes(p)
    # the -inf the sqrt reason quotes is stelling's own artifact and the
    # note says so — origin equation, origin site, and cause all named
    assert "lower bound -inf is negative" in note
    assert (
        "operand 0 is stelling's own ⊤ from 'sin' at s.py:9 (f) "
        "(no interval transfer is registered for it) — it did not come "
        "from your declaration; resolve that upstream decline first, "
        "this one is downstream of it" in note
    )
    # the artifact must NOT read as the user's declaration
    assert "declared input itself" not in note
    # accounting: sin ⊤ + sqrt decline, both counted; status unchanged
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 2
    assert dict(p.coverage.unknown_primitives) == {"sin": 1, "sqrt": 1}


def test_intermediate_producer_is_named_with_its_span():
    # sqrt(neg(x)) over x in [1, 4]: the argument spans [-4, -1], produced
    # by 'neg' — neither a declaration nor an artifact ⊤, and the note
    # says exactly that much (producer + span), no more
    x, m, s, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 4.0, src=("decl.py:3 (h)",)),
            eqn("neg", [x], m, src=("n.py:7 (f)",)),
            eqn("sqrt", [m], s, src=("h.py:12 (h)",)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    (note,) = _decline_notes(p)
    box = interval_env(q)[1]
    assert (min(box.los), max(box.his)) == (-4.0, -1.0)
    assert (
        f"operand 0 was produced by 'neg' at n.py:7 (f), "
        f"spanning [{box.los[0]}, {box.his[0]}]" in note
    )
    assert "declared input itself" not in note
    assert "did not come from your declaration" not in note
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1


# -- literal provenance (the which-literal-produced-it complaint) -------------


def test_convert_decline_quotes_the_literal_operand_and_its_dtype():
    # Use 2**53+1: an int64 value NOT exactly representable in float64, so
    # the conversion declines (values <= 2**53 pass through as point intervals)
    big = 2**53 + 1
    c, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            eqn(
                "convert_element_type",
                [lit(big, I64)],
                c,
                params=(("new_dtype", "float64"),),
                src=("lib.py:88 (rhs)",),
            ),
            eqn("ge", [c, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    (note,) = _decline_notes(p)
    # the offending SITE and the offending VALUE: which literal produced
    # the int64 is now in the note, next to the cast it fed
    assert "'convert_element_type' declined this form at lib.py:88 (rhs):" in note
    assert "'int64' -> 'float64'" in note
    assert f"operand 0 is the literal {big} (int64)" in note
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1


# -- verbatim-duplicate collapse, accounting pinned ---------------------------


def test_identical_decline_notes_collapse_but_the_counters_do_not():
    x, sa, sb = var(0), var(1), var(2)
    pa, pb, oa, ob = var(3, BOOL), var(4, BOOL), var(5, BOOL), var(6, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 4.0, src=("decl.py:3 (h)",)),
            eqn("sqrt", [x], sa, src=("h.py:12 (h)",)),
            eqn("sqrt", [x], sb, src=("h.py:12 (h)",)),
            eqn("ge", [sa, lit(0.0)], pa),
            eqn("ge", [sb, lit(0.0)], pb),
            eqn("stelling_assert", [pa], oa),
            eqn("stelling_assert", [pb], ob),
        ],
        [oa, ob],
    )
    p = propagate(q)
    # the two declines are byte-identical (same site, same operand
    # provenance): ONE note carries them both...
    assert len(_decline_notes(p)) == 1
    # ...while the accounting keeps the multiplicity: two sqrt equations
    # fell to ⊤ and both obligations are unknown — dedup changed no count
    assert p.coverage.unknown == 2
    assert dict(p.coverage.unknown_primitives) == {"sqrt": 2}
    assert [o.status for o in p.obligations] == ["unknown", "unknown"]


def test_declines_at_distinct_sites_keep_distinct_notes():
    x, sa, sb = var(0), var(1), var(2)
    pa, pb, oa, ob = var(3, BOOL), var(4, BOOL), var(5, BOOL), var(6, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 4.0, src=("decl.py:3 (h)",)),
            eqn("sqrt", [x], sa, src=("h.py:12 (h)",)),
            eqn("sqrt", [x], sb, src=("h.py:99 (g)",)),
            eqn("ge", [sa, lit(0.0)], pa),
            eqn("ge", [sb, lit(0.0)], pb),
            eqn("stelling_assert", [pa], oa),
            eqn("stelling_assert", [pb], ob),
        ],
        [oa, ob],
    )
    p = propagate(q)
    notes = _decline_notes(p)
    # different sites are different addresses: both stay, each naming its own
    assert len(notes) == 2
    assert sum("at h.py:12 (h):" in n for n in notes) == 1
    assert sum("at h.py:99 (g):" in n for n in notes) == 1
    assert p.coverage.unknown == 2


# -- the ⊤ provenance is tracked, never guessed -------------------------------


def test_operand_with_infinite_declared_bound_is_not_called_an_artifact():
    # a DECLARED unbounded-below input reaching sqrt: the -inf is the
    # user's own declaration, and the note must name the declaration —
    # never the artifact-⊤ sentence (the exact misattribution #2 fixes,
    # in the opposite direction)
    q = _sqrt_query(-inf, 4.0)
    p = propagate(q)
    (note,) = _decline_notes(p)
    assert "lower bound -inf is negative" in note
    assert "operand 0 is the declared input itself (declared at decl.py:3 (h))" in note
    assert "did not come from your declaration" not in note
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
