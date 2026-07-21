# Checking the preconditions your solver assumes

Numerical methods rest on properties their implementations assume and
never check: the coefficient field a CG or Cholesky solve needs to be
positive, the mass/shift scalar that removes a nullspace, the argument a
`log`/`sqrt`/division needs to be in-domain, the operating envelope a
scheme is only valid inside. These live in the **inputs** — before the
expensive computation — and when they fail, they fail **silently**: the
solver converges to something, the flags look healthy, and the answer is
wrong.

Install: `pip install stelling` into the environment that already has
your JAX (stelling never touches your resolver); add
`pip install "stelling[solvers]"` if you want the SMT step below.

Stelling checks these **over a declared range, not at a point**. A test
runs your code at some inputs; a verdict here holds for *every* value in
the envelope you declare — including the corner your test suite never
visits.

## Two worked examples

Check that a variable diffusion coefficient built from your own
construction is positive everywhere over the parameter range your
application can produce:

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from stelling.preconditions import field_positive, check

def face_coefficients(theta):
    # YOUR code's construction, not a simplified stand-in — e.g. the
    # conservative face averages a finite-volume Laplacian actually uses:
    a = theta                                  # your parameter -> coefficient path
    a_plus  = 0.5 * (a + jnp.roll(a, -1))
    a_minus = 0.5 * (a + jnp.roll(a, 1))
    return a_plus, a_minus                     # one obligation per value

def harness():
    _, obligations = field_positive(
        (64,), "float64", (1e-6, 1e2),         # the envelope you support
        transform=face_coefficients,
    )
    return obligations

print(check(harness, vacuity_mode="inputs-only").render())
```

Check that a configuration scalar can never be the singular value —
posed over the **admissible range**, not just the default, because the
question is whether your configuration space *admits* the bad value:

```python
from stelling.preconditions import scalar_nonzero, check

def harness():
    _, obligation = scalar_nonzero("float64", (0.0, 1.0))   # the config range
    return (obligation,)

# an explicit time budget opts in to the SMT step (never on by default):
print(check(harness, vacuity_mode="inputs-only", solver_timeout_ms=20_000).render())
```

## Reading the verdict

`vacuity_mode` is required — `"inputs-only"` is the standard choice
(your declared ranges widen; transcribed constants hold still). On a
VERIFIED, check() re-runs the query with the bounds widened: if an
obligation still discharges with the bounds gone, the verdict tells
you (a note and a stamped line) that the envelope was not
load-bearing — the claim is a theorem or the envelope is mis-posed.
A VERIFIED from check() has always already passed this check.

- **VERIFIED** — the property holds at every point of the declared
  envelope, judged by outward-rounded interval arithmetic (and the SMT
  step, if you opted in). The stamp on the verdict names every
  assumption this rests on: versions, the query's content hash, the
  arithmetic semantics, precision configuration, solver invocations or
  their recorded absence, and coverage.
- **REFUTED** — the property definitely fails. When the SMT step found
  it, the verdict carries a **concrete witness** — the input value at
  which your precondition breaks — independently confirmed by exact
  rational replay before it is shown to you.
- **UNKNOWN** — the analysis could not decide, and says why: the notes
  quote exactly what was dropped, declined, or too wide. An UNKNOWN is
  never silently rounded to either answer.

The second example REFUTES with witness `0`: the configuration range
admits the singular value and nothing in the code forbids it. That is a
statement about your config space, not about your default.

## What this checks — and what it doesn't (yet)

**In scope:** *input-side* preconditions — pointwise or scalar
properties of the data your solve consumes, stated over declared
envelopes. This is deliberately the portable core: no adaptivity, no
mesh machinery, no method internals.

**Out of scope today, stated plainly:**

- **Properties of the solve's behaviour** — conditioning over the
  envelope, residual-implies-error — are a different, planned layer.
  The boundary is exactly the solve: this module checks what goes *in*.
- **Array obligations escalate for small static shapes.** Interval
  propagation judges array obligations elementwise (the first example
  is an array, and VERIFIED means every element), and the SMT step now
  takes **small, statically-shaped** array obligations too — one
  per-obligation emission budget (512 element terms and root
  conjuncts) with anything above it declining, the numbers quoted. A
  REFUTED array obligation's witness names **which element violates**,
  with per-element values, replay-confirmed before you see it. General
  dynamic or arbitrarily-large array reasoning stays out of scope — the
  sound, tractable thing is bounded static-shape emission, and the
  budget line in the decline tells you when you have left it.
- **Float-exact claims by default.** The default verdict is about exact
  real arithmetic over your declared sets, and the stamp says so; an
  opt-in `ieee` mode judges binary64 behaviour and stamps its own scope.
  If your precondition is a float-boundary fact, read the stamp's
  semantics line before trusting either mode blindly.

Each limit shows up in the verdict itself — as a quoted decline, an
UNKNOWN with its reason, or a stamped assumption — never as a silent
pass. That is the design: the tool tells you when it could not earn the
claim, so a VERIFIED means exactly what it says.

## Precision configuration

`check()` records the live `jax_enable_x64` state in the stamp. If your
code assumes float64, enable it before tracing (as in the examples) —
and note that stelling checks the obligation under the configuration the
trace *ran under*, which is only meaningful if it matches the
configuration your production code runs under.

## Posing guidance — what not to pose (learned from the field)

Three rules from the first unguided field tests, each of which removed a
class of false alarms without silencing a single real finding:

1. **Pose inputs, not loop state.** A denominator produced by the
   iteration and fed back (a Lanczos/bidiagonalization `rho`, a running
   residual) is not an input — it is constrained by the algorithm's own
   invariants, and posing it as a free box manufactures a false REFUTED.
   The class boundary is the solve: if the quantity exists only inside
   the loop, its guarantees are the algorithm's business, not an input
   precondition.
2. **A tag can be an instruction, not an assertion.** `unit_diagonal`
   tells the backend to *ignore* the stored diagonal — the stored values
   carry no claim, so posing "diagonal == 1" flags nothing real. Before
   posing a tagged property, read the downstream use: if the tagged data
   is never read, there is no precondition. (A tag that *asserts* a
   property the code then relies on — `well_posed=True` guarding an
   unchecked division — is the opposite case, and exactly what should be
   posed: asserted-never-verified is the class's home ground.)
3. **Sentinels are values, not hazards.** `conlim = 0` meaning
   "disabled" and `rcond = 0` meaning "cut only exact zeros" are
   documented conventions; pose the precondition over the non-sentinel
   range, or at the guard's own strength (`>= 0`, not `> 0`), or the
   alarm is about the convention, not the code.

## Reading a CI verdict: gate or triage

A REFUTED whose witness is a value the *input space genuinely admits*
(an unguarded config scalar, an unchecked asserted tag) is gate-grade.
A REFUTED that depends on whether a caller invariant holds is a flag for
a human to triage. The rules above move most of the second kind out of
the alarm stream before it reaches you; what remains fired for a reason.
