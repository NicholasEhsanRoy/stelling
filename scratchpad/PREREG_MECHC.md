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

All numbers below were DRIVEN. Anything reasoned and not run is marked
**SUSPECTED** in place.

Corpus as built: **23 cases**, 84 trace-keys (case × order × `refine`),
168 runs (× `assume_mode`), **184 obligation-runs**. Baseline tree
`/home/nick/MSF/stelling/src` at `e8b9377`; branch tree
`/home/nick/MSF/.wt-mechc/qsa/src`; `stelling.__file__` and
`jax.__version__` stamped into every ledger and checked before every
comparison.

## H1 — one shared point, and both legs really consult it — **MET**

`stelling.exactness.certifies_set_refutation(*, nonemptiness_certified,
assume_dropped)`. Consulted by
`propagate._withhold_uncertified_refutations` (interval leg, once at the
end of the run) and by `affine.refine_propagation` (affine leg), both by
module attribute.

**Falsifier A — not observed.** `test_both_legs_consult_the_shared_-
set_refutation_point` forces the shared answer False; both legs withhold,
and both positive controls (queries with NO assume) refute on their own
leg unpatched, asserted in the same test.

**Falsifier B — not observed; both mutants red the pin.** Separate
worktrees `/home/nick/MSF/.wt-mechc/m1` and `m2`, `python -B`,
`__pycache__` cleared, `stelling.__file__` verified into each worktree.

| mutant | change | routing pin | rest of the suite |
|---|---|---|---|
| M1 | interval leg: call replaced by an inlined copy of the same expression | **FAILED** ("must reach exactness.certifies_set_refutation, not a private copy") | 1 failed, 2356 passed, 2 skipped |
| M2 | affine leg: same, inlined | **FAILED** ("must reach the SAME shared point") | 1 failed, 2356 passed, 2 skipped |

Both mutants are behaviour-preserving — the inlined expression is
logically identical to the call it replaces — so **no verdict test can
see them**. Under each mutant **exactly one test of 2359 fails, and it is
the routing pin**: the pin is not merely sensitive to the bypass, it is
the only thing in the tree that is.

## H2 — the order-dependence is gone — **MET AS SCOPED, MISSED AS WRITTEN**

Measured, per obligation, both `refine` legs: **order-dependent rows 16
of 38 at `e8b9377`, 2 of 38 on the branch.**

The clause as pre-registered said *identical*, and **2 rows are not**.
They are `certified_input_assume` (`x ∈ [0,1]`, `assume(x >= 0.9)`,
`assert_(x <= 0.5)`) at `refine=None` and `refine="affine"`:
`violated-over-set` with the assume first, `unknown` with it last.

**That is the forward-only NARROWING, not the withholding, and the
pre-registration should have said so.** Narrowing is applied in equation
order (`design/constraining-assume.md`), so an assume traced after an
obligation does not narrow the box that obligation is judged over. The
ruling implemented here is the one the task states — *read the whole-run
flag at the end* — and it is about the withholding. The residual
direction is sound in both cells: a definite violation over the WIDER box
is a violation at every point of the narrowed region, and *certified*
means that region was shown inhabited. It costs an UNKNOWN and cannot
mint a verdict.

**The stronger claim that IS met, and is the one that matters:** no
obligation's `violated-over-set` face depends on trace order on any run
whose assume state is uncertified — 0 such rows, from 16.

## H3 — positive control — **MET**

Unchanged from `e8b9377` to the branch, both orders, both legs:

| control | case | status |
|---|---|---|
| (a) no assume, interval-decided | `no_assume_interval` | REFUTED both orders, both legs |
| (a) no assume, affine-decided | `no_assume_affine` | affine REFUTED both orders |
| (b) certified narrowing of a declared input | `certified_assume_definite_violation` | REFUTED both orders, both legs |
| (c) definitely-true assume (F8 channel) | `f8_definitely_true_assume` | REFUTED both orders, both legs |
| (c) definitely-true dropped conjunct | `harmless_relational` | REFUTED both orders, both legs |

## H4 — `discharged` never touched — **MET**

`discharged` obligation-runs: **28 at `e8b9377`, 28 on the branch**, and
the diff is empty per obligation, not only in count. At the verdict
layer, **VERIFIED 12 → 12**.

## H5 — nothing moves toward VERIFIED — **MET**

All **18** moved obligation-runs are `violated-over-set` → `unknown`. No
`unknown` → `violated-over-set`, nothing to or from `discharged`. Verdict
layer: 16 moves, all REFUTED → UNKNOWN.

## H6 — the cost, on my own corpus — **MET, and the void condition did not fire**

| | `discharged` | `violated-over-set` | `unknown` |
|---|---|---|---|
| `e8b9377` | 28 | 94 | 62 |
| branch | 28 | 76 | 80 |

Of the 18 moved obligation-runs:

- **(a) 12 wrong REFUTEDs closed** — oracle finds **0** admitted points
  (`plain_empty`, `redundant_empty`, `drop_empty_sum`,
  `drop_empty_scalar`, `two_obligations_across_the_assume` ob#0,
  `nonvacuity_before_drop` nv#0; each at both `refine` legs);
- **(b) 6 legitimate REFUTEDs lost** — oracle finds admitted points that
  really violate the obligation: `mixed_nonempty` (560 of 29 269),
  `restricting_relational` (4 995 of 29 277), `drop_nonempty_sum`
  (29 269 of 29 269), each at both legs;
- **(c) 0 neither.**

**The void condition did not fire: column (b) is 6, not 0**, and it
includes the pre-registered `_mixed_nonempty` shape by name. The loss is
real and is the class a later certificate is meant to recover.

**Not a rate.** 12:6 on a corpus of 23 harnesses chosen to REACH the
defect. The prior 92 / 168 / 252 came from three other corpora and are
not comparable with this or with each other.

## H7 — byte-identity where nothing should move — **MET**

110 runs compared on obligation statuses AND details, notes **including
their order**, stamped assumptions, and coverage counts — every
no-assume run, every all-certified run, and **every** run in
`assume_mode="inert"`: **0 differ.**

## H8 — the cause is named per flag — **MET**

- drop-only run (`plain_empty`): the withhold note says *"an assume was
  DROPPED rather than applied"* and does **not** contain
  `over-approximated intermediate`; `narrowing_uncertified is False`.
- narrowing-only run (`_obligations_across_an_uncertified_assume`): the
  note says *"narrowed an over-approximated intermediate"* and does
  **not** contain `DROPPED`; `assume_dropped is False`.

Both pinned (`test_the_withholding_sentence_names_the_mechanism_that_-
actually_fired`, `test_the_query_scoped_withholding_names_the_mechanism_-
that_fired`).

## H9 — both series green, collection identical — **MET**

**2357 passed / 2 skipped** on jax 0.11.0 (138 s at load 0.74) and on jax
0.10.2 (140 s at load 3.06), same checkout, run sequentially.
`--collect-only -q` ids **byte-identical** between the series, 2359 ids.
Additionally the 168-run corpus ledger is **run-for-run identical** on the
two series (0 disagreements).

## H10 — the deferred single-series claim — **MET, by measurement**

`design/constraining-assume.md:174` ("jax verified 0.11.0 throughout")
is scoped in place, and the scope is measured, not asserted:

```
0.11.0 venv:  import mime -> /home/nick/MSF/msf/MIME/src/mime/__init__.py
0.10.2 venv:  import mime -> ModuleNotFoundError: No module named 'mime'
```

No install into either interpreter is permitted, so that row **cannot**
be reproduced on 0.10.2 from this tree. The file now says the row is
one-series and why, and notes that every other claim in it is
series-independent by construction.

## H11 — hygiene — **MET**

`pre-commit run --all-files` passes, `reuse lint` **Passed** (rc=0),
`docs/supported-primitives.md` regenerated (its `propagate.py:LINE`
citations moved by the edit), SOUNDNESS entry added with the direction
stated, worktree clean.

## H12 — the docstring — **MET**

`stelling.harness.assume.__doc__` no longer contains `inert`; it states
what `assume` does (narrows where representable, DROPPED where not and
the query is then judged over a superset) and what it scopes over (the
whole query, in both the empty and the possibly-empty case), and names
the one positional thing — forward-only narrowing — with why that
direction is safe. `assert_.__doc__` gains the missing clause.

---

## Out of scope, recorded

`solvers.py`'s escalation leg still keys on `propagation.assume_dropped`
alone rather than on the shared point. It is already whole-run (a boolean
read once at the end of escalation) so its scope is right, but it does
not see `narrowing_uncertified` and it does not consult
`certifies_set_refutation` — the same fragility the affine leg had. **Not
changed here** (the ruling names two legs) and **not measured**:
**SUSPECTED** that routing it through the shared point is a no-op on
verdicts, because a run that narrows an over-approximated intermediate
raises `coverage.constrained`, which `escalate()` already refuses on. Not
run, so not claimed.
