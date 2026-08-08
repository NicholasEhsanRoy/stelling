# PREREG — the non-emptiness certificate

Written **before** the first `src/` edit, on branch
`fix/nonempty-region-certificate` off `main` @ `681c6ef`.
Baseline: 2369 passed / 2 skipped on both jax series, `--collect-only`
ids byte-identical, `reuse lint` rc=0.

Outcomes are appended **below** the pre-registered clauses, never edited
into them.

## The problem

A run whose assume state does not certify a set-level refutation
(`stelling.exactness.certifies_set_refutation` returning False, from
`narrowing_uncertified` or `assume_dropped`) withholds **every**
`violated-over-set` from REFUTED, query-wide. The withholding is there
because the *assumed region may be empty*, in which case every
obligation is vacuously true.

If the assumed region is demonstrably **non-empty**, that reason is
gone and the refutation stands.

## The mechanism I intend to build

A **point witness**: one point of the declared set at which **every**
`stelling_assume` predicate of the query is definitely true.

Implementation intent (re-derived on `main`, not merged from the spike):

* reuse the existing probe machinery — `_Propagator.pin`, `_pinned`,
  `_probe_point`, `_member_bounds` — so the witness is a **member of
  the declared set** (a value of the declaration's own dtype, inside
  its declared box), not merely a real number in the interval;
* evaluate the assume predicates by **stelling's own real-mode
  interval arithmetic**, whose endpoints are computed in `Fraction`
  and correctly directed-rounded (`interval._exact_down` /
  `_exact_up`). A predicate box of `[1, 1]` on every element means the
  predicate is true at the true value of that point, because every
  transfer output encloses it. No second arithmetic;
* require **every** `stelling_assume` equation in the query
  (statically, sub-jaxprs included) to have been evaluated definitely
  true on the same probe run — an assume the probe never reached (an
  untaken cond branch) does not count;
* feed the answer into the **existing** shared decision as a new
  keyword argument, not through a new channel.

## Pre-registered claims, each with its falsifier

### C1 — routing (Gate 1)

**Claim.** The certificate is a new *input* to
`exactness.certifies_set_refutation`, consulted by **both** the
interval leg and the affine leg; and the per-probe witness decision is
itself a named function in `stelling.exactness` that
`stelling.propagate` consults rather than reimplements.

**Falsifier.** A new routing pin that **captures the keyword arguments**
at the shared point (the existing pins use `lambda **k: False` and are
argument-blind by their own admission) fails to observe the new keyword
arriving from one of the two legs; or a behaviour-preserving mutant that
inlines the witness predicate at its consumer leaves the whole suite
green. Either observation falsifies C1.

**Pre-registered mutant set (must redden exactly the named pin):**
`INLINE_WITNESS` — replace the `exactness` call that decides the witness
with an inline expression of identical value in `propagate.py`;
`INLINE_SETREF` — replace the shared-decision call with
`region_inhabited or certifies_set_refutation(<old two kwargs>)` at each
leg. Expected: `INLINE_WITNESS` reddens the witness routing pin;
`INLINE_SETREF` reddens the kwarg-capturing pin on the corresponding
leg. Expected additionally: neither mutant changes any verdict on the
corpus (they are behaviour-preserving), so a pin that only checks
verdicts cannot see them.

### C2 — one-sidedness

**Claim.** A failed search leaves the withholding **exactly** as it was:
same statuses, same notes, same details. Failing to find a point is not
a proof of emptiness.

**Falsifier.** A query whose assumed region is empty
(`y = x - x; assume(y >= 0.5)`), or whose predicate the arithmetic
cannot confirm (`assume(sqrt(x) >= <the exact boundary>)`), whose
obligation moves off `unknown`, or whose notes/detail differ from the
pre-certificate run byte-for-byte.

### C3 — `discharged` is never touched

**Claim.** Over the corpus, **per obligation**, the count of obligations
that move toward `discharged`/VERIFIED because of the certificate is
**zero**.

**Falsifier.** Any obligation whose status is `unknown` or
`violated-over-set` without the certificate and `discharged` with it; or
any query whose verdict is UNKNOWN/REFUTED without and VERIFIED with.

**Positive control (required — a zero with no positive control is
unfalsifiable).** A deliberately unsound mutant of my own instrument
that *can* reach `discharged` (`WITNESS_TWO_SIDED`: let the certificate
also lift a withheld `discharged`, plus `WITNESS_ALWAYS`: certify every
run inhabited, on a corpus row whose region is empty and whose
obligation is vacuously true). **Pre-registered expectation: the oracle
flags at least one row on each of those two mutants.** If the oracle
reports zero on the mutants too, the zero on the real build means
nothing and C3 is unmeasured.

### C4 — recovery is sound

**Claim.** Every refutation the certificate gives back is sound: the
assumed region really is non-empty at the exhibited point.

**Falsifier.** An independent oracle — brute-force sampling of the
declared set in `Fraction`/`float` outside stelling, checking the assume
predicates and the obligation predicate — that finds a recovered row
whose assumed region it can show empty, or whose obligation it can
satisfy at an admissible point.

### C5 — the ledger

**Claim.** On a corpus built by me, scored **per obligation**, the
certificate recovers a positive number of withheld refutations, and the
empty-region rows stay withheld.

**Falsifier.** Zero recovered (the build is pointless), or any
empty-region row recovered (the build is unsound).

### C6 — the cap (Gate 2)

**Claim.** Work is bounded by the **declared size**: above a declared
element count the search does not run at all, and the search stops at
the first witness. Timings are reported with load averages.

**Falsifier.** A measured time at n=4096 that is not bounded by the cap;
or a cap whose cost in recovered refutations is not measured. I
pre-register that the cap **does** cost recoveries at large declared
sizes and that I will report the number rather than choose a cap that
makes it zero by construction.

### C7 — both series stay green

**Claim.** Both venvs green, `--collect-only` ids byte-identical between
them, `reuse lint` rc=0.

**Falsifier.** Any non-green run, any id diff, any lint failure.

### C8 — what the corpus cannot see

**Claim.** I will state the corpus's structural blind spots explicitly
before scoring, and check at least: (a) does it contain a VERIFIED
query at all; (b) does it contain a non-`float64` declaration; (c) does
it contain a top-level obligation; (d) does it contain a row where the
certificate must FAIL; (e) does it score per obligation.

**Falsifier.** A blind spot found after the fact that the list did not
name.

## Things I expect to be TRUE and will label SUSPECTED unless run

* the interval-arithmetic witness check is strictly weaker than an
  exact-rational one at the boundary (`0.1 + 0.2 >= 0.30000000000000004`
  reads INDETERMINATE, not FALSE) — **to be measured**;
* the certificate cannot help a query whose assume predicate is
  `sqrt`/`sin`/`exp`/`log` *without slack*, and can help one *with*
  slack — **to be measured**;
* branch-scoped assumes are never certified by this build (the probe
  takes one branch, the static requirement covers both) — **to be
  measured**.

---

# OUTCOMES (appended; nothing above this line is edited)

**Recorded 2026-08-08.** Branch `fix/nonempty-region-certificate`, off
`main` @ `681c6ef`. All figures jax 0.11.0 unless stated; load averages
are in the results files they come from
(`scratchpad/cert/RESULTS_*.txt`).

## What was built, against what was pre-registered

Built as described, with **one addition the pre-registration did not
anticipate and one correction it forced**:

* **addition (Gate 2).** The pre-registration promised a cap "by declared
  size". A single size cap turned out not to bound the cost: a search that
  FINDS NOTHING walks the whole grid, and at the cap that is 469 ms
  against a 23 ms propagation — 95% of the whole `check()` pipeline. A
  second bound was added, the probe count itself scaling with the declared
  size. Both are reported below with what each costs.
* **correction.** The pre-registration said the certificate's arithmetic
  would be "stelling's own, in `semantics`". It is — and the consequence I
  had not thought through is that **the two dials are not ordered**:
  `sqrt` of the point 0.25 is exactly 0.5 in binary64, so ieee certifies
  `>= 0.5` where real (which bumps outward unconditionally) does not,
  while three other rows go the other way. Measured, and now pinned
  (`test_the_certificate_speaks_the_dial_the_query_was_judged_on`).

## The clauses

### C1 — routing (Gate 1): **MET**

The certificate is a third keyword argument to
`exactness.certifies_set_refutation`, consulted by both legs; the
per-probe decision is `exactness.certifies_point_witness`, consulted by
the propagator.

Two pins, both new, both of a shape this file did not previously contain:

* `test_the_witness_route_is_the_shared_primitive_too` forces the witness
  decision in BOTH directions — an inhabited region stops being certified,
  an *uncertifiable* one starts — which an inlined subset test cannot
  follow;
* `test_every_reach_of_the_shared_point_names_the_certificate` wraps the
  shared decision in a RECORDER and inspects the keyword arguments. It is
  the first pin in `test_exactness_lift.py` that is not argument-blind;
  the file's docstring says so and says exactly how much of its own blind
  spot this closes.

Pre-registered mutants, each in its own worktree under
`/home/nick/MSF/.wt-cert/mut/`, `python -B`, `__pycache__` cleared:


| mutant | whole suite (jax 0.11.0) | reddens |
|---|---|---|
| `M1_inline_witness` — inline `required <= witnessed` in `propagate.py` | 6 failed, 2388 passed, 2 skipped | `test_the_witness_route_is_the_shared_primitive_too` **plus 5** |
| `M2_inline_setref_interval` — `p.region_inhabited or certifies_set_refutation(<old two kwargs>)` in `propagate.py` | **1 failed**, 2393 passed, 2 skipped | `test_every_reach_of_the_shared_point_names_the_certificate`, and nothing else |
| `M3_inline_setref_affine` — the same in `affine.py` | **1 failed**, 2393 passed, 2 skipped | `test_every_reach_of_the_shared_point_names_the_certificate`, and nothing else |

M2 and M3 redden **exactly** the pin they were written against and no
other test in 2396 — which is the Gate-1 result, and also the measurement
that the pin is doing work no verdict test does.


The pre-registration said each mutant should redden "exactly the named
pin". `M1` reddens **six** tests, and the extra five are not a surprise
once looked at: they are the tests that CLOSE the certificate's route
(`monkeypatch.setattr(exactness, "certifies_point_witness", ...)`) as a
control while they measure something else, and an inlined predicate makes
that patch inert. Every one of the six fails for the same single reason —
the route is gone. That is a stronger observation than the
pre-registration asked for, not a weaker one, but the pre-registered
wording was wrong and is recorded as such.

Also as pre-registered: **no verdict on the corpus moves under any of the
three mutants** — all 32 shipped verdicts identical to the real build
under each (`RESULTS_mutants.txt`). A pin that only checked verdicts
would see nothing at all.

One thing the mutant ledger shows that the suite counts do not: under
`M1`, the ledger's own CONTROL arm collapses. That arm is implemented by
patching `certifies_point_witness` to False — the pre-certificate
behaviour — and an inlined predicate makes the patch inert, so the
"before" column reads REFUTED where it should read UNKNOWN. The shipped
"after" column is unchanged. That is the same single effect as the five
extra test failures, seen from the other side: `M1` does not change what
stelling answers, it removes the ability to ask what stelling would
answer without the certificate.

### C2 — one-sidedness: **MET**

`test_a_failed_certificate_search_changes_nothing_at_all` compares the
WHOLE `Propagation` object — statuses, details, notes and their order,
assumptions, coverage — between the run with the search and the run with
the certificate's route closed, on both declining rows. Equal.
`test_a_failed_search_is_not_a_claim_of_emptiness_anywhere_in_the_notes`
adds the text half: nothing on a declining run says the region is empty,
was searched, or was shown to be anything.

The design change this forced: **every declining path is silent**,
including the cap. A "searched and found nothing" note would have been
honest disclosure and would have made this clause unpinnable
byte-for-byte; the withholding sentence the run already carries explains
the withholding completely and was true before the certificate existed.

### C3 — `discharged` never touched: **MET, with the positive control**

Per obligation, 32 rows, 35 obligations, oracle 20 000 samples per row:

| | real build | `two_sided` | `certify_everything` |
|---|---|---|---|
| recovered (`unknown` → `violated-over-set`) | **13** | 0 | 17 |
| moved toward `discharged` | **0** | **13** | 0 |
| left `discharged` | **0** | 0 | 0 |
| other moves | **0** | 0 | 0 |
| oracle-confirmed WRONG VERIFIED | **0** | **13** | 0 |
| recovery on a row with no admissible point | **0** | 0 | **2** |

Verdict layer, per query: VERIFIED **4 → 4**, REFUTED 2 → 15, UNKNOWN
26 → 13; 13 queries move, **0 toward VERIFIED**. Under `two_sided`:
VERIFIED 4 → **17**.

The positive controls fire on both failure modes the clause is about, so
the two zeros in the real column are falsifiable observations rather than
an absence of measurement.

### C4 — recovery is sound: **MET**

The oracle vetoes are `wrong_refuted` (an admissible sampled point at
which the obligation is TRUE) and `vacuous_recovery` (a recovery on a row
where no admissible point was found in 20 000 samples). Both **0** on the
real build; the second is **2** on `certify_everything`, which is the
proof that it can fire.

**What protects the other 8 empty rows from `certify_everything` is not
this certificate**, and saying so is part of the honest reading: their
probes die of `UnsatisfiableAssumptionError` at the pinned point, because
a relational or arithmetic assume undecidable over BOXES is decidable at a
POINT and its meet comes out empty. The certificate is the second line
there, not the first.

### C5 — the ledger: **MET**

13 refutations recovered; every one of the 10 EMPTY-region rows stays
withheld.

### C6 — the cap: **MET, and the pre-registration was too weak**

Pre-registered: a size cap, and that it would cost recoveries. Both hold —
and a size cap alone did **not** bound the cost, which the pre-registration
did not anticipate. Measured (`RESULTS_cap.txt`, load 0.06–0.44):

| n | probes | failing search, size cap only | failing search, both bounds |
|---|---|---|---|
| 256 | 16 | 29 ms (20x) | 29 ms (20x) |
| 1024 | 4 | 116 ms (22x) | 30 ms (5.6x) |
| 4096 | 3 | **469 ms (20x, 95% of `check()`)** | **95 ms (4.3x)** |

A SUCCEEDING search costs 3.7x at every size — it stops at the first
witness.

**What the bounds cost in recovered refutations**, each turned off in
turn over n = 64 … 16384: the probe budget **0**, the size cap **1 per row
above it** (2 of the 7 sizes tested). On the corpus itself (max declared
size 64) both cost **0** — which is a fact about the corpus and is
reported as one.

The floor of 3 probes is measured, not fitted: across the 17 corpus rows
that witness at all, the first witnessing probe index is 0, 1 or 2 in
**17 of 17** (`RESULTS_probe_index.txt`) — those are the declared box's
low corner, high corner and midpoint. One probe alone recovers 18%.

### C7 — both series: **MET** (figures in the final section below)

### C8 — what the corpus cannot see: **MET on the checklist, and the list
grew**

Checklist: (a) VERIFIED queries — 4; (b) non-`float64` declarations —
`int32` ×2, `float32` ×2; (c) top-level obligations — all 35; (d) rows
where the certificate MUST fail — 10 EMPTY-region rows plus 2
uncertifiable-but-inhabited; (e) scored per obligation — yes, and the
per-query verdict counts are printed BESIDE the ledger, never instead.

Blind spots the checklist did not name, found while building and recorded
here:

1. **`jnp.all(...)` is out of reach entirely.** It lowers to `reduce_and`,
   which has no interval transfer in either registry, so its predicate is
   ⊤ at a POINT exactly as over a box. That is the single most common
   dropped-assume idiom and the certificate cannot see it;
   `r11_all_reduction_inhabited` is in the corpus to measure that rather
   than hide it. This is also why the recovery here (13) is far below the
   spike's reported 84-of-92 on its own corpus.
2. **The corpus's assumed regions are half-space-shaped.** A region whose
   only members sit off the corner/midpoint grid needs a later probe and
   is lost at the budget floor — so the "17 of 17 at three probes" figure
   is about these regions, not about regions in general.
3. **The oracle samples and therefore never proves a region EMPTY.** Its
   two verdicts are existence claims. "No admissible sample in 20 000" is
   printed as exactly that.
4. **The oracle evaluates in binary64 while stelling judges in ℝ.** The
   one row where that could matter is the deliberate boundary row and it
   is scored by hand.
5. **Independent per-element sampling cannot see a wide elementwise
   assume.** `2x >= 0.5` over `[0,1]^64` has admissible probability
   `0.75^64 ≈ 1e-8`; the first oracle reported `r31_wide_declaration` as
   EMPTY for that reason alone. Fixed by correlated and corner fills, and
   the row is kept as the instrument's own control.
6. **The corpus contains no `while`/`scan` body with an assume in it**,
   and no multi-branch `switch`. Branch-scoped assumes are covered by two
   tests but not by the ledger.

## The three SUSPECTED items, now measured

* **exact arithmetic** — CONFIRMED. `0.1 + 0.2 >= 0.30000000000000004` at
  the declared point pair: box
  `[0x1.3333333333333p-2, 0x1.3333333333334p-2]`, straddles, INDETERMINATE,
  no witness. TRUE in binary64, FALSE in ℝ, "not established" here.
* **transcendentals** — CONFIRMED, and the wording was too strong. With
  SLACK the certificate fires and is sound (`sqrt(x+1) >= 1.2` at
  `x = 0.5`); without slack it declines (`sqrt` of the point 0.25 against
  `>= 0.5`). The boundary is the MARGIN, not the primitive.
* **branch-scoped assumes** — CONFIRMED, by two independent mechanisms
  (the static requirement, and `_reachability_witnesses` certifying
  nothing on a constrained-or-dropped run). Both pinned.

## Two things measured that no clause asked for

* **I1 — a certifying probe narrows nothing.** The soundness argument for
  reading each assume's witness answer BEFORE `_assume_constrain` is that
  a `[1, 1]` predicate's meet is a no-op. Measured over the corpus: **148
  certifying probe runs inspected, 0 narrowed anything**
  (`RESULTS_invariant.txt`), and pinned by
  `test_a_certifying_probe_narrows_nothing`.
* **I2 — the two semantics dials are NOT ordered.** Recorded above under
  the corrections.

## A third thing no clause asked for, found by looking at the second leg

**The certificate was DEAD on the affine leg and is now live.** The
search's gate first asked only "did the interval leg withhold a
violation?" — which is never true on a query the interval leg cannot
decide, and the refinement judges exactly those. So on the one class the
refinement actually runs on (`assume_dropped`, `coverage.constrained ==
0`) the certificate was never computed, and `region_inhabited` arrived
False by construction: the same documented-dead situation
`nonemptiness_certified` is in on that leg, freshly created by this
change and invisible to every pin, because the routing was correct and
only the VALUE was constant.

Measured: `assume(x >= y)` over `x, y ∈ [-1,1]^3` (relational, dropped,
region inhabited at `x = y = 0`) with `assert_(x - x >= 0.5)` —
interval-undecided, affine-violated — returned UNKNOWN from the
refinement. The gate now also fires on an `unknown` obligation when
nothing was constrained, which is precisely when the refinement will run,
and the same query returns REFUTED. Cost: 1.4 ms → 30 ms on a
256-element declaration whose interval leg withheld nothing, inside the
bounds already measured. Pinned with its empty-region twin.

This is the clearest thing the whole exercise turned up about routing
pins: a pin can prove an argument ARRIVES and still say nothing about
whether it ever arrives non-constant. `test_exactness_lift.py` says so
about `nonemptiness_certified` in its own docstring; the same trap was
one gate-condition away from swallowing this argument too.

## Both series, and the lint (C7)

| | jax 0.11.0 | jax 0.10.2 |
|---|---|---|
| whole suite | **2398 passed, 2 skipped** (140.30 s, load 0.59/2.85) | **2398 passed, 2 skipped** (139.97 s, load 2.85/3.74) |
| `--collect-only` ids | 2400, byte-identical between the two (`diff` empty) | |
| `reuse lint` | rc=0, 317/317 files with copyright and license information | |

Baseline at `main` @ `681c6ef` was 2369 passed / 2 skipped on both, so the
branch adds 31 tests and removes none.

## Where to reproduce

* `scratchpad/cert/ledger.py [--mutant two_sided|certify_everything]` —
  the per-obligation ledger with its positive controls;
* `scratchpad/cert/cap_timing.py` — the bounds' cost in milliseconds and
  in recovered refutations;
* `scratchpad/cert/probe_index.py` — which probe index witnesses, per row;
* `scratchpad/cert/invariant.py` — I1 and I2;
* `scratchpad/cert/apply_mutant.py NAME` — the three routing mutants,
  applied in place inside their own worktree.

Raw outputs are the `RESULTS_*.txt` beside them, each with the load
average it was taken under.
