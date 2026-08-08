# PREREG — `assume` is a precondition on the WHOLE QUERY

**Branch** `fix/query-scoped-assume` off `main` = `e8b9377`.
**Worktree** `/home/nick/MSF/.wt-mechc/qsa`.
**Written before the first `src/` edit.** Outcomes are APPENDED below the
line at the bottom; nothing above that line is edited afterwards.

Interpreters (no installs into either, ever):

- `/home/nick/venvs/stelling-jax/bin/python` — jax **0.11.0**
- `/home/nick/venvs/stelling-jax010/bin/python` — jax **0.10.2**

Every run under `export JAX_PLATFORMS=cpu
PYTHONPATH=/home/nick/MSF/.wt-mechc/qsa/src`, with `stelling.__file__` and
`jax.__version__` printed and checked before each comparison (both venvs
carry an editable install pointing at `main`; the trap is real).
`JAX_ENABLE_X64=1` wherever a float64 declaration is traced.

Anything reasoned but not run is labelled **SUSPECTED**.

---

## The change, stated so it can be wrong

An `assume` the checker cannot represent exactly is dropped, and the run
records that. Today the withholding that follows is read **at the assert**
by the interval leg, so it sees only the assumes traced *before* it. The
ruling: an `assume` is a precondition on the **whole query**. The
detectably-empty case already kills the run whole (`UnsatisfiableAssumptionError`
takes obligations written above the line with it, because if the assumed
region is empty every obligation is vacuously true); the possibly-empty
case is the same fact known less precisely and gets the same scope.

Shape to build: **one shared predicate** — *given this run's whole assume
state, is a set-level refutation certified?* — in `stelling/exactness.py`
beside `certifies_nonemptiness`, layered on it (`certifies_nonemptiness`
answers *is this region inhabited*; the new one answers *is a refutation
certified on this run*), consulted by **both** legs, with a routing pin
that forces it False and asserts **both** legs withhold.

---

## Claims, each with the observation that would falsify it

### H1 — one shared point, and both legs really consult it

`stelling.exactness.certifies_set_refutation` exists, and both the
interval leg and the affine leg reach it through the module attribute.

**Falsifier A (routing pin).** With
`monkeypatch.setattr(exactness, "certifies_set_refutation", lambda **k: False)`:
(i) a **no-assume** query the interval leg refutes still reports
`violated-over-set`; or (ii) a **no-assume** query the affine leg refutes
(interval-undecided, affine-decided: `assert_(x - x >= 0.5)`) still
reports `violated-over-set` under `refine="affine"`. Either observation
says that leg holds a private copy. The pin is not satisfiable by an inert
control: both queries carry **no assume at all**, so the unpatched run
must REFUTE on that leg (asserted in the same test as the positive
control).

**Falsifier B (mutation, separate worktrees, `python -B`, `__pycache__`
cleared).** Two mutants, one per leg, each replacing the call to the
shared predicate with an inlined copy of the same expression. If the
routing pin stays GREEN under either mutant, the pin does not pin routing
and H1 is unsupported.

### H2 — the order-dependence is gone

For every corpus case and every `refine ∈ {None, "affine"}`, the
per-obligation status is **identical** with the assume traced before the
assert and after it.

**Falsifier.** Any (case, refine) pair whose before-order and after-order
per-obligation status vectors differ. Measured on my own corpus, scored
**per obligation**, not per query.

### H3 — positive control: what should still refute still does

Neighbouring cases that must keep refuting: (a) no assume at all; (b) the
only assume is a **certified** narrowing of a declared input (exact box);
(c) the only assume is definitely TRUE over the boxes in force (the F8
channel — it restricts nothing). All three, on both legs, in **both**
trace orders.

**Falsifier.** Any of those returning `unknown` where `main` returned
`violated-over-set`. A cost clause with no positive control produces an
unfalsifiable zero; this is that control.

### H4 — one-sided: `discharged` is never touched

The multiset of obligation indices with status `discharged` is identical
between `e8b9377` and the branch across the whole corpus, on both legs,
both orders, and under `assume_mode="inert"`.

**Falsifier.** Any obligation `discharged` on exactly one side.

### H5 — nothing moves toward VERIFIED

Every per-obligation status change is `violated-over-set` → `unknown`.

**Falsifier.** Any obligation moving `unknown` → `violated-over-set`, or
anything → `discharged`, or `discharged` → anything.

### H6 — the cost, measured on my own corpus and counted honestly

Report, per obligation: the count that was `violated-over-set` on
`e8b9377` and is `unknown` on the branch, split by an **oracle** into

- **(a) wrong REFUTEDs closed** — the sampled oracle finds **0** points of
  the declared box satisfying the conjunction of *all* the harness's
  assumes (region empty ⇒ the claim is vacuously true ⇒ REFUTED was
  wrong), and
- **(b) real loss** — the oracle finds at least one sampled point that
  satisfies every assume **and** violates the obligation, so the
  refutation was genuine and is given up.

The prior figures 92 / 168 / 252 were measured on three different corpora,
none is a rate, and none is inherited: my number is my own.

**Falsifier / void condition.** If column (b) is **0** while the corpus
contains cases constructed to be genuine refutations under a dropped
assume (the `_mixed_nonempty` shape: `assume((x >= 0.) & jnp.all(x >= 0.5))`
over `x ∈ [-1,1]^3`, region `[0.5,1]^3` NON-empty, `assert_(x > 5.)`
violated everywhere in it), then the oracle is not discriminating and the
measurement is **void**, not a zero-cost result.

**Oracle.** Per case, the same source function is run twice: once with the
stelling harness primitives, once with numpy/jax on concrete samples where
`any_array` returns a sample, `assume` records `bool(all(pred))` and
`assert_` records `bool(all(pred))`. Sampling per declaration: 20 000
uniform draws over the declared box + every corner + a grid, under
`JAX_ENABLE_X64=1`. The oracle applies **every** assume in the harness
regardless of trace position — that is the ruling, stated as the oracle.

### H7 — byte-identity where nothing should move

For every corpus case with **no assumes**, every case whose assumes are
**all certified**, and **every** case under `assume_mode="inert"`, the
full `Propagation` rendering (obligation statuses **and details**, notes,
assumptions, coverage counts) is byte-identical between `e8b9377` and the
branch. The guard returns having done nothing.

**Falsifier.** Any differing rendering, including note ORDER.

### H8 — the cause is named per flag, not generically

A run whose only defect is a **DROPPED** assume (nothing narrowed, or the
narrowing target was exact) must **not** quote the
over-approximated-intermediate mechanism in the sentence whose job is to
explain the withholding. A run whose only defect is an **uncertified
narrowing** must quote that mechanism and not the drop.

**Falsifier.** The drop-only run's `WITHHELD from REFUTED` note containing
the substring `over-approximated intermediate`; or the narrowing-only
run's note containing `DROPPED`.

### H9 — both series green, collection identical

`pytest -q -ra` (never `-rs`) passes on both interpreters in this
worktree, and `--collect-only -q` ids are byte-identical between them.

**Falsifier.** Any failure or error on either series; any nonempty id
diff. Wall-clock figures, if any, carry a load average.

### H10 — the deferred single-series claim at `design/constraining-assume.md:174`

That line ("jax verified 0.11.0 throughout") is settled: either
re-measured on 0.10.2, or the scope is stated in the file **with the
reason measured**, not asserted.

**Falsifier.** The file left carrying a claim whose series scope a reader
cannot determine.

### H11 — hygiene

`reuse lint` ends rc=0; `docs/supported-primitives.md` regenerated so its
`propagate.py:LINE` citation matches; a `SOUNDNESS.md` entry exists with
the direction of the flip stated; the worktree is clean at the end.

**Falsifier.** `reuse lint` rc≠0; the supported-primitives drift test red;
no SOUNDNESS entry; `git status` dirty.

### H12 — the docstring

`stelling.harness.assume`'s docstring no longer says assumptions are
"inert, conservative", and says both **what `assume` does** and **what it
scopes over**. `assert_`'s set-wise sentence gains the missing clause that
assumes narrow the set it quantifies over.

**Falsifier.** The string `inert` surviving in `assume.__doc__`; or
`assume.__doc__` not naming the scope.

---

## What this change is NOT

- Not the planned "certificate" that recovers most of the loss.
- Not a change to `solvers.py`'s leg (already whole-run; reported, not
  edited).
- Not two-sided: `discharged` is never withheld (H4).

---

# OUTCOMES (appended after the fact; nothing above this line is edited)
