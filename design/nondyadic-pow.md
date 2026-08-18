# Non-dyadic `pow` exponents — the value lost, and what soundness costs

**Status:** DESIGN NOTE, opened 2026-08-18. Not built. Scoped to 0.3.0
by the principal's ruling the same day, after an argument that building
it before the falsification probe (roadmap item: B10) inverts the
dependency between a new discharge route and the check that catches it.

**Why this note exists.** Every claim below is either a measurement
taken on 2026-08-18 against `main` at `3482822`, or an argument that
names the measurement it rests on. The design it describes is one
factor away from re-minting **audit 0.2.0 S1** — the false VERIFIED on
`x**0.1 <= 1e30` over `[1, 1e300]`, caused by rationalising a binary64
exponent with `limit_denominator(128)` and emitting `aux^10 = x` about
it. The difference between the sound construction and that defect is a
single multiplicative term. This note exists so that term is never
implicit.

## 1. What is actually declined, measured

`x ** e` declines at EMISSION when the exact rational of the binary64
`e` (`obligation.pow_exponent_rational`) has a denominator above
`RATIONAL_POW_DEGREE_CAP = 128`. Since every binary64 is dyadic, the
reachable denominators are exactly `2, 4, …, 128`.

Measured, through `obligation.rational_pow_problem`:

```
gamma, diatomic/air (7/5)   e=1.4                  q=2^51  DECLINED
gamma, monatomic (5/3)      e=1.6666666666666667   q=2^52  DECLINED
cube root                   e=0.3333333333333333   q=2^54  DECLINED
power law                   e=0.7                  q=2^52  DECLINED
sqrt                        e=0.5                  q=2^1   EMITTABLE
3/2                         e=1.5                  q=2^1   EMITTABLE
5/2                         e=2.5                  q=2^1   EMITTABLE
1/128 (cheap corner)        e=0.0078125            q=2^7   EMITTABLE
```

**THE DECLINE IS AT ESCALATION ONLY, AND THAT NARROWS THE LOSS
SHARPLY.** Interval propagation handles every exponent. Measured, both
rows on the same four exponents, `x` on `[1, 100]`:

| property | 0.5 | 1/3 | 0.1 | 0.7 |
|---|---|---|---|---|
| `x**e >= 0.99` — intervals suffice | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| `x**e <= x` — needs relational reasoning | VERIFIED | declined | declined | declined |

So the lost class is precisely: **relational properties over non-dyadic
exponents.** Cube roots verify today whenever interval arithmetic can
close the property; they fail only when the property needs the solver.

**An instrument note, recorded because it cost a wrong first reading.**
The first probe used `x**e >= 1.0` on `[1, 100]` and returned UNKNOWN
for the non-dyadic rows — which looked like the decline and was not.
`1.0 ** 0.1` is exactly `1.0`, and outward-rounded interval evaluation
puts the lower endpoint one ulp BELOW it, so `>= 1.0` fails on the
interval for a reason that has nothing to do with the exponent. The
dyadic control failed the same way. A property stated at the exact
boundary measures the rounding, not the feature.

## 2. Why the corpus cares

`gamma = 7/5` (air, diatomic) and `gamma = 5/3` (monatomic) are the two
standard adiabatic indices in compressible flow, and both are declined.
Any isentropic `p ∝ rho^gamma` relation is in the lost class, and those
are exactly the RELATIONAL properties — `p` tied to `rho` through the
exponent — that fall to the solver rather than to intervals. JAXFLUIDS
is in `corpus/`. This is the evidence line; it is a field instance, not
imagined demand (L16).

## 3. The construction

The obstruction is not `1/3`. It is that `x ** 0.3333333333333333`
denotes the exact dyadic `6004799503160661/2^54`, whose honest
algebraic encoding is `aux^(2^54) = x^6004799503160661` — infeasible.
S1's defect was substituting `1/3` for it SILENTLY.

The principled version substitutes DELIBERATELY and carries the error.
Pick a simple rational `r = p'/q'` near the true exponent `e`, emit the
feasible algebraic relation `aux^q' = x^p'`, and bracket the gap. Since

    x^e = x^r * exp((e - r) * ln x)

then with `delta = |e - r|` and `L = max|ln x|` over the input domain:

    x^e  in  x^r * [1 - eta, 1 + eta],   eta = exp(delta * L) - 1 ~= delta * L

computed with OUTWARD rounding. This is a containment, not an equality:
sound by construction, and it degrades to a decline when `eta` grows
rather than lying. It keeps the relational tie between the output and
`x` through `aux`, which is the whole reason it beats intervals.

Scale, worked for `gamma = 1.4` against `r = 7/5`, where
`delta <~ 1.55e-16`:

| domain | `L` | `eta` | as a multiple of eps = 2^-52 |
|---|---|---|---|
| `[1, 1e300]` | 691 | ~1.1e-13 | ~480x |
| `[1e-6, 1e6]` | 13.8 | ~2.1e-15 | ~9.6x |
| `[1, 100]` | 4.6 | ~7.1e-16 | ~3.2x |

**Two prerequisites, both already anticipated in the tree's own prose.**
`7/5` and `5/3` have ODD denominators, so this needs the odd-`q` arm
that `rational_pow_problem` currently fails closed on — and that
docstring already states the hazard: the emission's `aux >= 0` root
guard is written for an EVEN `q`, an odd `q` has a single real root and
needs no guard, *"which is exactly why a wrong encoding there would be
silent."* And `eta` must compose with whatever budget relates the
encoding to jax's actual floating-point `pow`; the current dyadic path
emits an exact algebraic relation, and how that relates to the computed
float is UNVERIFIED IN THIS NOTE and must be established before
designing against it.

## 4. The gauge — the shipping condition, and three sharpenings it needs

**The condition (principal, 2026-08-18):** a dedicated, hardcoded
mutation that sets `eta` to 0 during the SMT emission of a non-dyadic
`pow`. If the gauge does not instantly go red and catch the false
VERIFIED, the suite is inadequate and the feature cannot ship.

The instinct is right and is the same move that made B7's bar/gauge
work: gauge the load-bearing factor directly, and read a surviving
mutant as a statement about the SUITE. Three things it needs on top.

### 4.1 `eta = 0` does not automatically mint a false VERIFIED

With `eta` dropped the model says the output is EXACTLY `x^r` when the
truth is a band around it. That becomes an observably wrong verdict
only for a property whose margin at the worst point is SMALLER than
`eta * x^r`. State the property with ordinary slack and the mutant
survives — not because the suite is careless, but because nothing in it
is `eta`-sensitive.

So the condition, read precisely, is a demand that the suite CONTAIN A
KNIFE-EDGE PROPERTY. It is constructible:

```
x in [1, 1e300],  assert  x**1.4 <= 1e300**1.4 * (1 + 1e-14)
    with eta:    upper bound is x^r*(1 + 1.1e-13) > threshold  ->  NOT VERIFIED
    with eta=0:  maximum is exactly 1e300^r      < threshold  ->  VERIFIED
```

That flips on exactly the factor being gauged. Note the flip is
VERIFIED vs NOT-VERIFIED, not VERIFIED vs REFUTED: with `eta` the
solver finds a violating point in the band that jax may not actually
produce, so the honest arm is UNKNOWN-or-REFUTED. The gauge assertion
is on the disappearance of VERIFIED, not on the arrival of REFUTED.

### 4.2 The battery's DOMAIN is load-bearing, and a friendly one kills it

`eta ~= delta * L`. From the table in section 3: on `[1, 100]`, `eta`
is about 3x machine epsilon — indistinguishable from the rounding it
sits next to, so no property can discriminate and the mutant survives for a
reason that says nothing about the code. **A gauge written on a
friendly `[1, 100]` domain goes green under `eta = 0` and reports
nothing.** The battery has to run on wide domains. This is a real
design constraint and it is not obvious from the mutation alone.

### 4.3 Necessary, not sufficient

Each of these is untouched by `eta = 0` and independently mints the
same class:

- `eta` applied with INWARD rounding instead of outward;
- `eta` applied to one endpoint and not the other;
- `L` taken as `|ln(max x)|` instead of `max|ln x|` — silent on
  `[1, 100]`, wrong the moment the domain dips below 1;
- `eta` computed from an ASSUMED `delta` rather than the real
  `|e - r|` — this is S1 wearing a hat;
- the choice of `r` itself, which `eta = 0` never perturbs.

`eta = 0` should be the FIRST ROW of that battery, with the whole
battery required to have no unexplained survivors — the bar
`fidelity.gauge` already enforces elsewhere.

### 4.4 Two traps specific to shipping behind a flag

- **The mutation must revert the SHIPPED emission line, not a test
  double.** This tree has the lesson recorded: B6 audit 3 found a guard
  that reverted with ZERO suite reds because it was proven by
  construction (`obligation.py @@ -2972`).
- **The battery must run with the flag ON.** A flag-off suite makes the
  mutant unreachable and it "survives" for a reason unrelated to
  detection. This wants the `DRIVEN_*` treatment — the gauged set
  derived from the fixture table and asserted equal to measured reach —
  so a flag that silently disables coverage is itself a red.

## 5. Sequencing

**After the falsification probe, not before.** The probe is the
downstream check for exactly this class: a wrong discharge with no
witness to replay is the S4 asymmetry, and `eta = 0` is a textbook
instance of it. Built after, the probe covers the feature for free and
the shipping condition gains a second, independent enforcer. Built
before, the mutation battery is the only thing between the feature and
a false VERIFIED.
