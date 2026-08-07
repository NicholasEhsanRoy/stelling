# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The sibling sweep: every construction, its stelling harness, and its
numpy twin. Each entry: name, specs, harness factory, assume twin, assert twin."""
import numpy as np
import jax
import jax.numpy as jnp
from stelling import harness as H

S0   = ((0,),  "float64", (-1.0, 1.0))
S0_2 = ((2,0), "float64", (-1.0, 1.0))
S0_3 = ((0,3), "float64", (-1.0, 1.0))
SC   = ((),    "float64", (-1.0, 1.0))
S1   = ((1,),  "float64", (-1.0, 1.0))
S3   = ((3,),  "float64", (-1.0, 1.0))
S21  = ((2,1), "float64", (-1.0, 1.0))

CASES = []

def case(name, specs, hfun, afun, cfun, note=""):
    CASES.append(dict(name=name, specs=specs, harness=hfun,
                      assume=afun, assertf=cfun, note=note))

def _decl(specs):
    return [H.any_array(sh, dt, b) for (sh, dt, b) in specs]

# ---- 1. the reported defect -------------------------------------------------
def h1():
    k, z = _decl([SC, S0])
    H.assume((k >= 0.5) & (z >= 2.0))
    return (H.assert_(k > 0.0),)
case("A1_size0_right", [SC, S0], h1,
     lambda k, z: (k >= 0.5) & (z >= 2.0),
     lambda k, z: k > 0.0, "the reported construction")

# ---- 2. size-0 in the LEFT operand position ---------------------------------
def h2():
    k, z = _decl([SC, S0])
    H.assume((z >= 2.0) & (k >= 0.5))
    return (H.assert_(k > 0.0),)
case("A2_size0_left", [SC, S0], h2,
     lambda k, z: (z >= 2.0) & (k >= 0.5),
     lambda k, z: k > 0.0)

# ---- 3. BOTH operands size-0 ------------------------------------------------
def h3():
    k, z1, z2 = _decl([SC, S0, S0])
    H.assume((z1 >= 2.0) & (z2 >= 0.5))
    return (H.assert_(k > 0.0),)
case("A3_both_size0", [SC, S0, S0], h3,
     lambda k, z1, z2: (z1 >= 2.0) & (z2 >= 0.5),
     lambda k, z1, z2: k > 0.0)

# ---- 4. size-0 nested deeper, left-assoc ------------------------------------
def h4():
    k, m, z = _decl([SC, SC, S0])
    H.assume(((k >= 0.5) & (m >= 0.5)) & (z >= 2.0))
    return (H.assert_(k > 0.0), H.assert_(m > 0.0))
case("A4_nested_left", [SC, SC, S0], h4,
     lambda k, m, z: ((k >= 0.5) & (m >= 0.5)) & (z >= 2.0),
     lambda k, m, z: np.array([k > 0.0, m > 0.0]))

# ---- 5. size-0 nested deeper, right-assoc -----------------------------------
def h5():
    k, m, z = _decl([SC, SC, S0])
    H.assume((k >= 0.5) & ((m >= 0.5) & (z >= 2.0)))
    return (H.assert_(k > 0.0), H.assert_(m > 0.0))
case("A5_nested_right", [SC, SC, S0], h5,
     lambda k, m, z: (k >= 0.5) & ((m >= 0.5) & (z >= 2.0)),
     lambda k, m, z: np.array([k > 0.0, m > 0.0]))

# ---- 6. size-0 mixed with | -------------------------------------------------
def h6():
    k, z = _decl([SC, S0])
    H.assume((k >= 0.5) & ((z >= 2.0) | (z <= 0.0)))
    return (H.assert_(k > 0.0),)
case("A6_mixed_or_inside", [SC, S0], h6,
     lambda k, z: (k >= 0.5) & ((z >= 2.0) | (z <= 0.0)),
     lambda k, z: k > 0.0)

# ---- 7. top-level | ---------------------------------------------------------
def h7():
    k, z = _decl([SC, S0])
    H.assume((k >= 0.5) | (z >= 2.0))
    return (H.assert_(k > 0.0),)
case("A7_top_or", [SC, S0], h7,
     lambda k, z: (k >= 0.5) | (z >= 2.0),
     lambda k, z: k > 0.0)

# ---- 8. rank-1 sibling ------------------------------------------------------
def h8():
    v, z = _decl([S3, S0_3])
    H.assume((v >= 0.5) & (z >= 2.0))
    return (H.assert_(v > 0.0),)
case("A8_rank1_vs_0x3", [S3, S0_3], h8,
     lambda v, z: (v >= 0.5) & (z >= 2.0),
     lambda v, z: v > 0.0)

# ---- 9. shape (1,) sibling (broadcast 1-vs-0) -------------------------------
def h9():
    w, z = _decl([S1, S0])
    H.assume((w >= 0.5) & (z >= 2.0))
    return (H.assert_(w > 0.0),)
case("A9_shape1_vs_0", [S1, S0], h9,
     lambda w, z: (w >= 0.5) & (z >= 2.0),
     lambda w, z: w > 0.0)

# ---- 10. rank-2: (2,1) against (2,0) ----------------------------------------
def h10():
    a, z = _decl([S21, S0_2])
    H.assume((a >= 0.5) & (z >= 2.0))
    return (H.assert_(a > 0.0),)
case("A10_rank2_2x1_vs_2x0", [S21, S0_2], h10,
     lambda a, z: (a >= 0.5) & (z >= 2.0),
     lambda a, z: a > 0.0)

# ---- 11. size-0 comparison whose OPERANDS are not both size-0 ---------------
def h11():
    k, m, z = _decl([SC, SC, S0])
    H.assume((m >= 0.5) & (k >= z))      # k >= z is bool[0]
    return (H.assert_(m > 0.0),)
case("A11_size0_via_operand", [SC, SC, S0], h11,
     lambda k, m, z: (m >= 0.5) & (k >= z),
     lambda k, m, z: m > 0.0)

# ---- 12. size-0 assume alone (no `and`) -------------------------------------
def h12():
    k, z = _decl([SC, S0])
    H.assume(z >= 2.0)
    return (H.assert_(k > 0.0),)
case("A12_size0_alone", [SC, S0], h12,
     lambda k, z: z >= 2.0,
     lambda k, z: k > 0.0)

# ---- 13. legitimate conjunction control (NO size-0) -------------------------
def h13():
    k, v = _decl([SC, S3])
    H.assume((k >= 0.5) & (v >= -0.5))
    return (H.assert_(k > 0.0),)
case("B1_legit_conjunction", [SC, S3], h13,
     lambda k, v: (k >= 0.5) & (v >= -0.5),
     lambda k, v: k > 0.0, "must stay VERIFIED")

# ---- 14. legitimate scalar-vs-rank1 broadcast control -----------------------
def h14():
    k, v = _decl([SC, S3])
    H.assume((k >= 0.5) & (v >= 0.0))
    return (H.assert_(k + jnp.sum(v) > 0.0),)
case("B2_legit_broadcast", [SC, S3], h14,
     lambda k, v: (k >= 0.5) & (v >= 0.0),
     lambda k, v: k + np.sum(v) > 0.0, "must stay VERIFIED")

# ---- 15. size-0 inside a cond branch ----------------------------------------
def h15():
    k, z = _decl([SC, S0])
    def tb(a):
        H.assume((a >= 0.5) & (z >= 2.0))
        return a
    y = jax.lax.cond(k > -0.5, tb, lambda a: a * 0.0 + 5.0, k)
    return (H.assert_(y > 0.0),)
case("A13_cond_branch", [SC, S0], h15,
     lambda k, z: (k >= 0.5) & (z >= 2.0),
     lambda k, z: np.where(k > -0.5, k, 5.0) > 0.0,
     "assume lives in a cond branch")

# ---- 16. size-0 with an eq comparison ---------------------------------------
def h16():
    k, z = _decl([SC, S0])
    H.assume((k == 0.5) & (z >= 2.0))
    return (H.assert_(k > 0.0),)
case("A14_eq_narrowing", [SC, S0], h16,
     lambda k, z: (k == 0.5) & (z >= 2.0),
     lambda k, z: k > 0.0)

# ---- 17. size-0 sibling narrows a TRANSFER output ---------------------------
def h17():
    k, z = _decl([SC, S0])
    y = k * 2.0
    H.assume((y >= 1.0) & (z >= 2.0))
    return (H.assert_(k > -0.6),)
case("A15_narrow_transfer_out", [SC, S0], h17,
     lambda k, z: (k * 2.0 >= 1.0) & (z >= 2.0),
     lambda k, z: k > -0.6)

# ---- 18. size-0 sibling + an EMPTY-MEET conjunct: false harness-defect alarm -
def h16():
    k, z = _decl([SC, S0])
    H.assume((k >= 2.0) & (z >= 2.0))     # k>=2 impossible on [-1,1]
    return (H.assert_(k > 0.0),)
case("A16_false_unsat_alarm", [SC, S0], h16,
     lambda k, z: (k >= 2.0) & (z >= 2.0),
     lambda k, z: k > 0.0, "assume is SATISFIABLE (bool[0]); alarm would be false")

# ---- 19. does the subset narrowing mint a REFUTED? --------------------------
def h17():
    k, z = _decl([SC, S0])
    H.assume((k >= 0.5) & (z >= 2.0))
    return (H.assert_(k < 0.0),)
case("A17_refuted_direction", [SC, S0], h17,
     lambda k, z: (k >= 0.5) & (z >= 2.0),
     lambda k, z: k < 0.0, "REFUTED is sound here: subset witness is admitted")

# ---- 20. strict-boundary collapse under a size-0 sibling --------------------
def h18():
    k, z = _decl([SC, S0])
    H.assume((k > 1.0) & (z >= 2.0))     # (1,1] collapses -> false alarm
    return (H.assert_(k > 0.0),)
case("A18_false_collapse_alarm", [SC, S0], h18,
     lambda k, z: (k > 1.0) & (z >= 2.0),
     lambda k, z: k > 0.0, "assume is SATISFIABLE (bool[0])")

# ---- 21. size-0 sibling narrows via a nonvacuity point ----------------------
def h19():
    k, z = _decl([SC, S0])
    H.assume((k <= -0.5) & (z >= 2.0))
    return (H.assert_(k < 0.0),)
case("A19_le_narrowing", [SC, S0], h19,
     lambda k, z: (k <= -0.5) & (z >= 2.0),
     lambda k, z: k < 0.0)
