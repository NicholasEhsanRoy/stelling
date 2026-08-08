"""The MECHC corpus: one source per case, run twice — once through the
stelling harness primitives, once as concrete numpy/jax over sampled
points (the oracle).

Scored PER OBLIGATION, never per query. Every case declares its inputs,
its assume thunks and its assert thunks separately, so the SAME case can
be traced with the assumes written BEFORE the asserts and with them
written AFTER — order is the whole subject.

`orders`:
  "both"  — emit the case in both trace orders (the order-dependence rows)
  "fixed" — the body pins its own interleaving (the two-obligation exhibit)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass
class Case:
    name: str
    # (shape, dtype, lo, hi) per declaration
    decls: tuple
    # each takes the declared values in order and returns a predicate
    assumes: tuple = ()
    asserts: tuple = ()
    nonvacuities: tuple = ()
    orders: str = "both"
    # for orders == "fixed": a list of ("assume"|"assert"|"nonvacuity", idx)
    script: tuple = ()
    # a case whose harness cannot be expressed by the script (lax.cond):
    # builder(order) -> nullary harness. `assumes`/`asserts` above are then
    # the ORACLE's implication form, not the trace form.
    builder: object = None
    note: str = ""


F = "float64"


def _c(*a, **k):
    return Case(*a, **k)


CASES = [
    # ---- dropped assume, region EMPTY: the wrong REFUTEDs -------------------
    _c("plain_empty", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: x > 5.0,),
       note="whole drop; region EMPTY"),
    _c("redundant_empty", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: (x >= -1.0) & jnp.all(x >= 2.0),),
       asserts=(lambda x: x > 5.0,),
       note="mixed drop beside a no-op narrowing; region EMPTY"),
    _c("jointly_empty", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: (x >= 0.5) & jnp.all(x <= 0.2),),
       asserts=(lambda x: x < -0.5,),
       note="neither conjunct alone false; region EMPTY"),
    # ---- dropped assume, region NON-empty: the real loss --------------------
    _c("mixed_nonempty", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: (x >= 0.0) & jnp.all(x >= 0.5),),
       asserts=(lambda x: x > 5.0,),
       note="region [0.5,1]^3 NON-empty; the refutation is GENUINE"),
    _c("restricting_relational", ((3,), F, 0.0, 10.0),
       assumes=(lambda a, b: (a >= 0.0) & (a <= b),),
       asserts=(lambda a, b: a > 50.0,),
       note="relational drop, region non-empty; genuine refutation"),
    _c("drop_nonempty_affine", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= -0.5),),
       asserts=(lambda x: x - x >= 0.5,),
       note="AFFINE-decided, region [-0.5,1]^3 NON-empty; genuine loss"),
    # ---- one-sided: discharges must survive ---------------------------------
    _c("mixed_verified", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: (x >= 0.0) & jnp.all(x >= 0.5),),
       asserts=(lambda x: x > -5.0,),
       note="discharged under a mixed drop"),
    _c("discharged_under_drop", ((3,), F, 0.0, 10.0),
       assumes=(lambda x: jnp.all(x >= 0.0),),
       asserts=(lambda x: jnp.sum(x) >= 0.0,),
       note="discharged under a whole drop"),
    _c("affine_dropped_verified", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: x - x >= -0.5,),
       note="AFFINE discharge under a dropped assume"),
    # ---- positive controls: these must KEEP refuting ------------------------
    _c("no_assume_interval", ((3,), F, -1.0, 1.0),
       asserts=(lambda x: x >= 5.0,),
       note="CONTROL (a): no assume, interval-decided"),
    _c("no_assume_affine", ((3,), F, -1.0, 1.0),
       asserts=(lambda x: x - x >= 0.5,),
       note="CONTROL (a): no assume, affine-decided"),
    _c("certified_input_assume", ((), F, 0.0, 1.0),
       assumes=(lambda x: x >= 0.9,),
       asserts=(lambda x: x <= 0.5,),
       note="CONTROL (b): certified narrowing of a declared (exact) box"),
    _c("f8_definitely_true_assume", ((), F, 0.0, 1.0),
       assumes=(lambda x: x + 0.0 <= 10.0,),
       asserts=(lambda x: x + 0.0 >= 5.0,),
       note="CONTROL (c): definitely-true assume, F8 channel"),
    _c("harmless_relational", ((3,), F, 0.0, 1.0),
       assumes=(lambda a, b: (a >= 0.0) & (a <= b),),
       asserts=(lambda a, b: a > 5.0,),
       note="CONTROL (c): the dropped conjunct is definitely TRUE"),
    # ---- the uncertified-narrowing flag, alone (H8 face 2) ------------------
    _c("uncertified_narrowing", ((), F, -1.0, 1.0),
       assumes=(lambda x: x * x <= -0.5,),
       asserts=(lambda x: x * x >= 0.0,),
       note="narrowing target is an over-approximated intermediate"),
    # ---- the two-obligation exhibit: THE defect, per obligation -------------
    _c("two_obligations_across_the_assume", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: x > 5.0, lambda x: x > 6.0),
       orders="fixed",
       script=(("assert", 0), ("assume", 0), ("assert", 1)),
       note="o0 traced BEFORE the assume, o1 AFTER; region EMPTY"),
    _c("two_obligations_certified", ((), F, 0.0, 1.0),
       assumes=(lambda x: x >= 0.9,),
       asserts=(lambda x: x <= 0.5, lambda x: x <= 0.95),
       orders="fixed",
       script=(("assert", 0), ("assume", 0), ("assert", 1)),
       note="CONTROL: same shape, CERTIFIED assume — both must stand"),
    _c("certified_assume_definite_violation", ((), F, 0.0, 1.0),
       assumes=(lambda x: x >= 0.9,),
       asserts=(lambda x: x >= 2.0,),
       note="CONTROL (b): certified assume + violation definite over the "
            "WHOLE declared box — must REFUTE in BOTH orders"),
    # ---- more dropped-assume shapes ----------------------------------------
    _c("drop_empty_sum", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: jnp.sum(x) >= 10.0,),
       note="whole drop, region EMPTY, reduce_sum obligation"),
    _c("drop_empty_scalar", ((), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: x > 5.0,),
       note="whole drop on a scalar declaration; region EMPTY"),
    _c("drop_nonempty_sum", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= -2.0),),
       asserts=(lambda x: jnp.sum(x) >= 10.0,),
       note="whole drop, region = the WHOLE box (non-empty); genuine loss"),
    # ---- the branch path, with the obligation traced BEFORE the assume ------
    _c("branch_assert_before_assume", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.where(x[0] > 0.0, jnp.all(x >= 2.0), True),),
       asserts=(lambda x: jnp.where(x[0] > 0.0, jnp.all(x > 5.0), True),
                lambda x: jnp.where(x[0] > 0.0, True, jnp.all(x > -5.0))),
       orders="fixed",
       builder="branch",
       note="lax.cond: the yes-branch obligation is traced BEFORE the "
            "branch's own assume; branch region EMPTY"),
    # ---- the nonvacuity FAILED face, same rule ------------------------------
    _c("nonvacuity_before_drop", ((3,), F, -1.0, 1.0),
       assumes=(lambda x: jnp.all(x >= 2.0),),
       asserts=(lambda x: x > -5.0,),
       nonvacuities=(lambda x: x > 5.0,),
       orders="fixed",
       script=(("nonvacuity", 0), ("assume", 0), ("assert", 0)),
       note="nonvacuity FAILED face traced BEFORE the dropped assume"),
]

# The two-declaration cases need their second declaration; `decls` above
# carries only the first for the one-input cases. Fill the rest here so the
# table stays readable.
_SECOND = {
    "restricting_relational": ((3,), F, 5.0, 6.0),
    "harmless_relational": ((3,), F, 5.0, 6.0),
}


def declarations(case):
    d = [case.decls]
    if case.name in _SECOND:
        d.append(_SECOND[case.name])
    return d


def build_harness(case, order):
    """A nullary stelling harness for this case in the given trace order."""
    from stelling.harness import any_array, assert_, assume, nonvacuity

    if case.builder == "branch":
        def hb():
            x = any_array((3,), F, (-1.0, 1.0))

            def yes(v):
                o = assert_(v > 5.0)      # traced BEFORE the branch's assume
                assume(jnp.all(v >= 2.0))
                return o

            def no(v):
                return assert_(v > -5.0)

            return (jax.lax.cond(x[0] > 0.0, yes, no, x),)
        return hb

    def h():
        vs = [any_array(s, dt, (lo, hi)) for (s, dt, lo, hi) in declarations(case)]
        out = []
        if case.orders == "fixed":
            for kind, i in case.script:
                if kind == "assume":
                    assume(case.assumes[i](*vs))
                elif kind == "assert":
                    out.append(assert_(case.asserts[i](*vs)))
                else:
                    out.append(nonvacuity(case.nonvacuities[i](*vs)))
        elif order == "before":
            for a in case.assumes:
                assume(a(*vs))
            for a in case.nonvacuities:
                out.append(nonvacuity(a(*vs)))
            for a in case.asserts:
                out.append(assert_(a(*vs)))
        else:  # "after": every assume is traced BELOW every obligation
            for a in case.nonvacuities:
                out.append(nonvacuity(a(*vs)))
            for a in case.asserts:
                out.append(assert_(a(*vs)))
            for a in case.assumes:
                assume(a(*vs))
        return tuple(out)

    return h


def orders_for(case):
    return ("fixed",) if case.orders == "fixed" else ("before", "after")


# --- the oracle ---------------------------------------------------------------

_N_UNIFORM = 20000


def _samples(decls, rng):
    """Joint samples over every declaration: uniform draws, all corners of
    each declaration (others at their midpoint), and a coarse grid."""
    pts = []
    for _ in range(_N_UNIFORM):
        pts.append([
            rng.uniform(lo, hi, size=s).astype(np.float64)
            for (s, _dt, lo, hi) in decls
        ])
    mids = [
        np.full(s, (lo + hi) / 2.0, dtype=np.float64)
        for (s, _dt, lo, hi) in decls
    ]
    for j, (s, _dt, lo, hi) in enumerate(decls):
        n = int(np.prod(s)) if s else 1
        for mask in range(2 ** n):
            v = np.array(
                [hi if (mask >> k) & 1 else lo for k in range(n)],
                dtype=np.float64,
            ).reshape(s)
            row = list(mids)
            row[j] = v
            pts.append(row)
    # a coarse product grid on the first declaration
    s0, _dt0, lo0, hi0 = decls[0]
    n0 = int(np.prod(s0)) if s0 else 1
    if n0 <= 3:
        axis = np.linspace(lo0, hi0, 21)
        grid = np.meshgrid(*([axis] * n0), indexing="ij")
        flat = np.stack([g.ravel() for g in grid], axis=1)
        for row in flat:
            r = list(mids)
            r[0] = row.reshape(s0)
            pts.append(r)
    return pts


def oracle(case, seed=20260808):
    """Per-obligation ground truth over the DECLARED box under EVERY assume
    of the harness, regardless of trace position — the ruling, stated as
    the oracle.

    Returns dict with
      n_points, n_admitted (points satisfying every assume),
      per obligation: n_violating_admitted (admitted points where the
      obligation is FALSE) and n_admitted_true.
    """
    rng = np.random.default_rng(seed)
    decls = declarations(case)
    pts = _samples(decls, rng)
    # stack into a leading batch axis, one array per declaration, and
    # evaluate under vmap: 29k+ points per case is not a per-point loop
    batch = [
        jnp.asarray(np.stack([row[j] for row in pts]))
        for j in range(len(decls))
    ]

    def _admit(*vs):
        ok = jnp.asarray(True)
        for a in case.assumes:
            ok = ok & jnp.all(a(*vs))
        return ok

    admitted = np.asarray(jax.vmap(_admit)(*batch))
    preds = list(case.nonvacuities) + list(case.asserts)
    obs = []
    for pfn in preds:
        truth = np.asarray(
            jax.vmap(lambda *vs, p=pfn: jnp.all(p(*vs)))(*batch)
        )
        obs.append({
            "violating_admitted": int(np.sum(admitted & ~truth)),
            "true_admitted": int(np.sum(admitted & truth)),
        })
    return {
        "n_points": len(pts),
        "n_admitted": int(np.sum(admitted)),
        "obligations": obs,
    }
