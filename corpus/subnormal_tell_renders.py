# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every rendered verdict quoted about the real-mode subnormal TELL.

The rule this exists to satisfy: **a figure is quoted only with the script
that produced it.** Run it against two trees and diff the two outputs; that
diff IS the claim, and nothing has to be taken on report.

    PYTHONPATH=<tree>/src JAX_PLATFORMS=cpu COLUMNS=200 \\
        python corpus/subnormal_tell_renders.py > before.txt
    PYTHONPATH=<other>/src JAX_PLATFORMS=cpu COLUMNS=200 \\
        python corpus/subnormal_tell_renders.py > after.txt
    diff before.txt after.txt

Run it in BOTH x64 cells (`JAX_ENABLE_X64` unset and `=1`); the cell picks
which format group A's declarations use, and the tell is a per-format
instrument, so a single cell is half the measurement.

**THE POPULATION IS THE CONTROL, NOT THE SUBJECT.** Two groups:

* **Group A — 9 harnesses × 2 semantics = 18 rendered verdicts.** The
  population the tell landed against (0.2.0 B18). Seven of the nine are
  ordinary programs that must not move one byte, and the two that do move
  are the two subnormal-sensitive ones. The `ieee` renders must be
  byte-identical throughout: the tell is a real-mode note and the ieee dial
  already models the flush.
* **Group B — 4 harnesses × 2 semantics = 8 rendered verdicts.** Added by
  the B18 fixup, and the group the first cut had nothing like. It walks the
  per-format axis the tell turns on: float16 (which this target does NOT
  flush — it keeps gradual underflow, measured eager and jit), against
  bfloat16 and float32 (which it does). Group B's two float16 rows are
  SILENCE cases; a tell there asserts a flush that does not happen.
* **Group C — 4 mixed-dtype rows, and they are NOT rendered verdicts.**
  A comparison whose two operands have different float dtypes cannot be
  reached through the public harness API at all: jax promotes, so
  ``x_f32 > jnp.float16(0.0)`` traces to ``gt a 0.0:f32[]`` and the mixture
  is gone before the propagator sees it. It reaches the propagator from
  hand-built and deserialized IR, which `propagate._ieee_cmp_get_min_normal`
  is written for. **That is exactly why an end-to-end population could not
  have caught the mixed-dtype defect, and why the pinning tests for it are
  hand-built IR** (tests/test_subnormal_tell.py). Group C drives the same IR
  here and prints the notes, so the claim travels with its script.

Group A's declarations follow the x64 cell (float64 when x64 is on, float32
when it is off) because that is what the public API gives a harness that
asks for float64 in each cell. Groups B and C name their formats explicitly,
which is the point of them.
"""

import os
import sys

import stelling

print("stelling.__file__ =", stelling.__file__)
print("PYTHONPATH        =", os.environ.get("PYTHONPATH"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

print(
    "jax", jax.__version__, "| x64 cell:", jax.config.jax_enable_x64,
    "| numpy", np.__version__, "| COLUMNS", os.environ.get("COLUMNS"),
)

from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402

X64 = jax.config.jax_enable_x64
DT = "float64" if X64 else "float32"
# the entirely-subnormal band for THIS cell's dtype
SUB = (1e-320, 1e-300) if X64 else (1.4e-45, 1e-40)

# ---- GROUP A: the population the tell landed against ----------------------


def h_subnormal():                       # THE SUBJECT
    x = any_array((), DT, SUB)
    return assert_(x > 0.0)


def h_subnormal_insensitive():           # band-touching, flush changes nothing
    x = any_array((), DT, SUB)
    return assert_(x < 1.0)


def h_normal_scalar():                   # ordinary: control
    x = any_array((), DT, (1.0, 2.0))
    return assert_(x > 0.0)


def h_normal_straddle():                 # contains 0: haze is the identity
    x = any_array((3,), DT, (-10.0, 10.0))
    return assert_(jnp.sum(x) >= -30.0)


def h_normal_array():                    # ordinary array work: control
    x = any_array((4,), DT, (0.5, 2.0))
    return assert_(jnp.sum(x * x) >= 1.0)


def h_normal_assume():                   # ordinary with an assume: control
    x = any_array((3,), DT, (-10.0, 10.0))
    assume(x >= 1.0)
    return assert_(jnp.sum(x) >= 0.0)


def h_int():                             # integers have no band: control
    n = any_array((), "int32", (1.0, 5.0))
    return assert_(n > 0)


def h_tiny_normal():                     # just ABOVE the band: control
    lo = 2.0 ** -1022 if X64 else 2.0 ** -126
    x = any_array((), DT, (lo, lo * 4.0))
    return assert_(x > 0.0)


def h_sub_eq():                          # eq over the band
    x = any_array((), DT, SUB)
    return assert_(x != 0.0)


GROUP_A = [h_subnormal, h_subnormal_insensitive, h_normal_scalar,
           h_normal_straddle, h_normal_array, h_normal_assume, h_int,
           h_tiny_normal, h_sub_eq]

# ---- GROUP B: the per-format axis ----------------------------------------
#
# float16's subnormal band (0 < |x| < 2**-14 = 6.1e-05) is by far the widest
# of the four in absolute terms AND the one format this target does not
# flush — measured, eager and jit, `x > 0` is True at every subnormal float16
# magnitude. So float16 is where a size test, a shared band, or an assumed
# flush is loudest, and all three of those were live for one commit.


def h_f16_subnormal():                   # SILENCE: the target keeps these
    x = any_array((), "float16", (1e-6, 1e-5))
    return assert_(x > 0.0)


def h_f16_bottom_of_band():              # SILENCE: same, at the band floor
    x = any_array((), "float16", (2.0 ** -15, 2.0 ** -14))
    return assert_(x > 0.0)


def h_bf16_subnormal():                  # FIRES: bfloat16 is flushed
    x = any_array((), "bfloat16", (1e-40, 1e-39))
    return assert_(x > 0.0)


def h_f32_subnormal():                   # FIRES: float32 is flushed
    x = any_array((), "float32", (1.4e-45, 1e-40))
    return assert_(x > 0.0)


GROUP_B = [h_f16_subnormal, h_f16_bottom_of_band, h_bf16_subnormal,
           h_f32_subnormal]

# ---- GROUP C: the mixed-dtype rows, from hand-built IR --------------------

from stelling import ir  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


def _aval(dtype):
    return ir.Aval(kind="ShapedArray", shape=(), dtype=dtype)


def mixed(dtype, lo, hi, rhs_dtype):
    """``assert_(x > rhs)`` with the two operands in different formats."""
    x = ir.Var(id=0, aval=_aval(dtype))
    pred = ir.Var(id=1, aval=_aval("bool"))
    out = ir.Var(id=2, aval=_aval("bool"))
    return ir.ClosedJaxpr(jaxpr=ir.Jaxpr(
        constvars=(), invars=(), outvars=(out,), eqns=(
            ir.JaxprEqn(
                primitive="stelling_any", invars=(), outvars=(x,),
                params=(("shape", ()), ("dtype", dtype),
                        ("lo", lo), ("hi", hi))),
            ir.JaxprEqn(
                primitive="gt",
                invars=(x, ir.Literal(val=0.0, aval=_aval(rhs_dtype))),
                outvars=(pred,)),
            ir.JaxprEqn(
                primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        )))


GROUP_C = [
    # label, query, what must happen and why
    ("float64 [1e-10,1e-9] > float16 0.0", mixed("float64", 1e-10, 1e-9, "float16"),
     "SILENT: the float64 box is 298 decades clear of its OWN band; only the "
     "widest band in the equation (float16's 2**-14) reaches it"),
    ("float32 [1e-30,1e-20] > float16 0.0", mixed("float32", 1e-30, 1e-20, "float16"),
     "SILENT: same shape one format down — the box sits 7.9 decades above "
     "float32's own 2**-126 band, and 15.8 decades below the float16 band "
     "the equation-wide maximum would have applied to it"),
    ("float64 [1e-320,1e-300] > float16 0.0", mixed("float64", 1e-320, 1e-300, "float16"),
     "FIRES, and is the control for the two above: the float64 operand IS "
     "inside float64's own band, and float64 is measured to flush"),
    ("float16 [1e-6,1e-5] > float64 0.0", mixed("float16", 1e-6, 1e-5, "float64"),
     "SILENT: the float16 operand is deep inside float16's own band, and "
     "this target does not flush float16"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
TELL = "SUBNORMAL-SENSITIVE DEFINITE ANSWER"


def drive(group, name):
    tells = 0
    for h in group:
        print()
        print("=" * 78)
        print("HARNESS:", h.__name__)
        print("=" * 78)
        for sem in ("real", "ieee"):
            v = check(h, vacuity_mode="inputs-only", semantics=sem)
            print(f"--- semantics={sem}  -> {v.status}")
            rendered = v.render().replace(HERE, "<CORPUS>")
            print(rendered)
            tells += sum(1 for ln in rendered.splitlines() if TELL in ln)
    print()
    print(f"### {name}: {len(group)} harnesses, {2 * len(group)} rendered "
          f"verdicts, {tells} tell line(s)")
    return tells


def drive_ir(rows):
    fired = 0
    for label, q, why in rows:
        print()
        print("=" * 78)
        print("MIXED IR:", label)
        print("   expected:", why)
        print("=" * 78)
        p = propagate(q)
        notes = [n for n in p.notes if n.startswith(TELL)]
        fired += len(notes)
        print(f"--- semantics=real  -> obligation "
              f"{p.obligations[0].status}, {len(notes)} tell note(s)")
        for n in notes:
            print("note:", n)
        pi = propagate(q, semantics="ieee")
        print(f"--- semantics=ieee  -> obligation "
              f"{pi.obligations[0].status}, "
              f"{sum(1 for n in pi.notes if n.startswith(TELL))} tell note(s)")
    print()
    print(f"### GROUP C: {len(rows)} mixed-dtype IR rows, {fired} tell note(s)")
    return fired


if __name__ == "__main__":
    a = drive(GROUP_A, "GROUP A")
    b = drive(GROUP_B, "GROUP B")
    c = drive_ir(GROUP_C)
    print()
    print(f"### TOTAL: {2 * (len(GROUP_A) + len(GROUP_B))} rendered verdicts "
          f"+ {len(GROUP_C)} mixed-dtype IR rows, {a + b + c} tell(s)")
    sys.exit(0)
