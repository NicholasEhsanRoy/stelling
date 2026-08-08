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

---

# OUTCOMES ROUND 2 — a blinded audit of this entry, reproduced and repaired
# (appended 2026-08-08; nothing above the rule is edited)

Branch `fix/shared-point-pin-both-directions`, off `main` @ `0ad22bb`,
own worktree at `/home/nick/MSF/.wt-pin/W`. Baseline verified in that
worktree before any edit of mine: **2398 passed / 2 skipped** on
jax 0.11.0, `reuse lint` rc=0.

**Provenance of this entry.** A predecessor died to an API limit with 711
uncommitted insertions in this worktree — prose, two source files, two
test files and ten measurement scripts — and committed none of it. I
inherited that work and committed it verbatim as a checkpoint
(`CHECKPOINT: inherited worktree, run once and green, not yet verified`)
so that the next limit would cost nothing. Everything after that
checkpoint is mine. **Every figure in this section is one I ran myself,
in a tree whose `stelling.__file__` and `jax.__version__` I printed
before the run.** Where my number differs from the inherited draft's, my
number is what stands and the difference is named. Where the audit's
reported number differs from mine, the same.

Every mutant is a SEPARATE worktree created from `0ad22bb` by
`scratchpad/pin/mutants.py`, run with `python -B` and `__pycache__`
cleared, in two families:

* **OLD/** = mutated source + the **`0ad22bb` tests** — *do the pins that
  were already in the tree see this?*
* **NEW/** = the same source + the **working tree's tests** — *does the
  pin I added?*

The mutation is applied by exact-text replacement and asserted to have
landed, because a mutation that silently failed to apply is
indistinguishable from a pin that missed it.

## F1 — the routing pins closed only the RESTRICTIVE direction

**REPRODUCED.** Every patch of a shared decision in
`tests/test_exactness_lift.py` was `lambda **k: False` /
`lambda *a, **k: False`, or a delegating recorder. Nothing ever forced
`certifies_set_refutation` or `certifies_nonemptiness` TRUE.

A conjunct is observable only through the answers it VETOES. So a leg can
keep a private copy of the run-level rule and demote the shared function
to an `and`-veto — the call still happens, unconditionally, as the FIRST
operand, with all three real keyword arguments, so a recorder sees
exactly what it expects and a forced `False` still makes the whole
conjunction `False` and still withholds.

| mutant | what it does | OLD family, whole suite, jax 0.11.0 |
|---|---|---|
| `M4_affine_and_private` | affine leg: `not (shared(...) and _affine_own_set_refutation(p))` | **2398 passed, 2 skipped, 0 failed** |
| `M5_both_and_private` | the same on BOTH legs | **2398 passed, 2 skipped, 0 failed** |
| `M6_nonemptiness_and_private` | `certifies_nonemptiness(...) and (def_true or id in exact)` | **2398 passed, 2 skipped, 0 failed** |
| `CTRL_unmutated` | nothing | 2398 passed, 2 skipped (the control) |

M6 is mine to add: the audit named only M4/M5, and the `certifies_nonemptiness`
route turns out to have exactly the same hole.

**FIX — three pins, each forcing its decision in the GRANTING direction.**
NEW family, `tests/test_exactness_lift.py` only (the control tree passes
16/16, so the red is the mutant and not the test):

| mutant | test that reddens | result |
|---|---|---|
| `M4` | `test_both_legs_follow_the_shared_point_in_the_TRUE_direction` | 1 failed, 15 passed — affine half, `'unknown' == 'violated-over-set'` |
| `M5` | the same test | 1 failed, 15 passed — interval half fails first |
| `M6` | `test_the_nonemptiness_route_is_pinned_in_the_TRUE_direction` | 1 failed, 15 passed — `narrowing_uncertified` stayed True |
| `M7` | `test_the_TRUE_direction_is_ONE_SIDED_too` | 1 failed, 16 passed — an undecided obligation was decided |

## F1 continued — the full inventory of shared-point pins

Every `monkeypatch.setattr` of a `stelling.exactness` decision in the
tree, with the direction it forces. Named rather than line-numbered,
because the lines move.

| pin | decision | before | after |
|---|---|---|---|
| `test_assume_certification_routes_through_the_shared_primitive` | `certifies_nonemptiness` | False only | False only |
| `test_routing_pin_covers_the_f8_channel_too` | `certifies_nonemptiness` | False only | False only |
| **NEW** `test_the_nonemptiness_route_is_pinned_in_the_TRUE_direction` | `certifies_nonemptiness` | — | **True** |
| `test_both_legs_consult_the_shared_set_refutation_point` | `certifies_set_refutation` | False only | False only |
| `test_the_shared_point_is_one_sided_on_both_legs` | `certifies_set_refutation` | False only | False only |
| **NEW** `test_both_legs_follow_the_shared_point_in_the_TRUE_direction` | `certifies_set_refutation` | — | **True** |
| **NEW** `test_the_TRUE_direction_is_ONE_SIDED_too` | `certifies_set_refutation` | — | **True**, scored on what must NOT move |
| `test_every_reach_of_the_shared_point_names_the_certificate` | `certifies_set_refutation` | delegating recorder | unchanged — forces no direction BY DESIGN, and is the only pin that sees a dropped keyword |
| `test_the_witness_route_is_the_shared_primitive_too` | `certifies_point_witness` | **False AND True** | unchanged — already whole |

AFFECTED and now fixed: `certifies_nonemptiness`, `certifies_set_refutation`.
NOT affected: `certifies_point_witness`, already pinned both ways, which
is the contrast that named the fix.

**Four further `lambda **k: False` patches of `certifies_point_witness`
are CONTROLS, not pins** — `_close_the_certificate`
(`test_exactness_lift.py`), `_without_certificate`
(`test_nonempty_certificate.py`), `_no_certificate`
(`test_vacuous_refutation.py`), and one in
`test_membership_idiom_hint.py`. Each closes the certificate's
independent route so a DIFFERENT mechanism is observable underneath it;
forcing them True would observe nothing. They still carry
restrictive-direction routing signal, which is the `M1_inline_witness`
"reddens six" result recorded in round 1.

**The query-scoped-assume pins are a different KIND and the audit's
question does not apply to them.** `test_an_obligation_ABOVE_the_assume_
is_withheld_with_the_ones_below`, `test_the_order_of_the_assume_no_longer_
moves_the_verdict`, the four `_ORDERING_ROWS` cells and
`test_the_run_level_decision_takes_no_position_argument` force NO shared
decision: three of them assert behaviour end to end and the fourth
inspects the function's signature for the absence of any parameter naming
an obligation, an equation or a position. There is no direction to force
in a pin that forces nothing, and adding one would not make them
stronger — `test_the_run_level_decision_takes_no_position_argument` is
already the sharpest available statement of query scope, because it
forbids the argument rather than checking a consequence of not having it.
They were audited and left alone, which is a result and not an omission.

**The one-sidedness pin is a weaker finding than the other two, and the
difference is recorded rather than smoothed.** `M4`/`M5`/`M6` are
invisible to the WHOLE SUITE. `M7` is invisible to
`tests/test_exactness_lift.py` — 14 of 14 pre-existing tests pass on it —
but not to the suite, which reddens **2 failed, 2396 passed, 2 skipped**
(`test_the_certificate_reaches_the_affine_leg_as_a_LIVE_argument` and
`test_solver_dispatch.py::test_inert_relational_assume_escalates_normally`).
So the new pin closes a hole in the ROUTING FILE, not a hole in the tree.

## F1 minor — the M2/M3 mutant names

**REPRODUCED.** `scratchpad/cert/apply_mutant.py` shows `m2`/`m3` keep
`exactness.certifies_set_refutation(...)` and drop the `region_inhabited`
keyword; neither inlines anything, so "inline" is the wrong word and the
prose beside them ("each leg lifting the withholding locally") was
already right. A GENUINE inlining — the call gone, the rule written out —
reddens **2** tests on each leg, not 1:

    OLD/M2g_genuine_inline_interval   2 failed, 2396 passed, 2 skipped
    OLD/M3g_genuine_inline_affine     2 failed, 2396 passed, 2 skipped
    (test_both_legs_consult_the_shared_set_refutation_point and
     test_every_reach_of_the_shared_point_names_the_certificate)

`SOUNDNESS.md` now says so and gives the names to read them by.

## F2 — "BRANCH-SCOPED ASSUMES ARE NEVER CERTIFIED" is false

**REPRODUCED** (`scratchpad/pin/f2_repro.py`). An `assume` inside a
`lax.cond` branch with nothing at top level (0 top-level
`stelling_assume` equations, 1 static required id): probe 1, the declared
box's HIGH corner, forces the branch, and the assume is evaluated,
witnessed and certified — `region_inhabited: True`, the note *"probe
point 1 of the declared set satisfies every assume"*, obligation back at
`violated-over-set`.

**The recovery is sound**: a 20 000-sample oracle over the EXECUTED
program finds **20 000 of 20 000** points admissible AND violating. (The
audit reported 2755; that is not what this construction gives, and the
direction — sound recovery — is what the finding turns on.)

Corrected in all three places (`SOUNDNESS.md`, `_region_witness`,
`exactness.certifies_point_witness`), which now separate the static
requirement (declines an assume the probe walked AROUND) from
`_reachability_witnesses` returning ∅ (which is what actually protects
branch-scoped violations).

**And the cost, measured.** The cost twin — the same assume, made EMPTY
inside its branch, so the region is inhabited only via the UNTAKEN side —
is not recovered: **8 of the 16 probes walk around it with an EMPTY
witness map**, `region_inhabited: False`, obligation `unknown`, and the
same oracle finds **9933 of 20 000** admissible violating points (seed 0;
the inherited draft said 9880 from a different seed). A sound refutation
lost to the static requirement, in exactly the shape the old sentence
claimed the rule prevented.

Pinned by `test_an_assume_the_probe_walks_INTO_is_witnessed_and_certified`
and `test_a_region_inhabited_only_via_the_UNTAKEN_branch_is_not_recovered`.

## F3 — constant arguments at shared points

**REPRODUCED, with my own counts**, in a worktree that wraps each call
site in a recorder DELEGATING to the real decision, whole suite jax
0.11.0 (2402 passed / 2 skipped in that tree):

* `affine.py`'s reach of the shared point: **31 reaches, all 31 with
  `nonemptiness_certified=True` and `coverage.constrained == 0`**, and
  `narrowing_uncertified` False at every one. Over
  `scratchpad/pin/corpus_pin.py`: **70 of 70**. The other two arguments
  are LIVE on the same runs — `assume_dropped` True at 12 of 31 and 40 of
  70, `region_inhabited` at 2 and 8 — so what is constant is one argument
  of three, not the call. Structurally constant and not merely
  empirically: `narrowing_uncertified = True` has exactly one assignment
  site, inside the `if narrowed:` block whose head sets `any_constrained`
  and calls `counter.record_constrained`, and `refine_propagation`
  declines wholly on `coverage.constrained` before the loop.
* `certifies_point_witness`'s `bool(required_assumes)` guard: **2203
  calls from `_region_witness`, non-empty at 2203 of 2203**, and **1544
  of 1544** over the corpus — dead in production, because
  `if not required: return False` runs first. The only empty-set calls
  anywhere in the tree are the **2** from
  `test_the_point_witness_decision_is_one_sided_and_static`, established
  by recording each caller rather than inferred.

Three different figures were in circulation for these two facts — the
audit's 26/26 and 2153/2153, the inherited draft's 29/29 and 2169/2169,
and a stale 25 in the module docstring of `test_exactness_lift.py`. None
was mine; all three are now replaced by the counts above, stated AT the
call site and in the function's own docstring.

## F4 — "31 tests added, none removed"

**REPRODUCED exactly.** `--collect-only` id diff, `681c6ef` (2371 ids) vs
`0ad22bb` (2400 ids): **31 added, 2 removed, net +29**, matching
2369 → 2398. Removed:
`test_doc_examples.py::test_doc_example[harness-api.md:614]` and
`[harness-api.md:660]`; `docs/harness-api.md` gained 11 lines and lost 2
(`git diff --numstat`), so both blocks shifted 9 lines and re-entered as
`[:623]` and `[:669]`, which are among the 31 added. The claim is in
`SOUNDNESS.md`, added by `ef41164`, and is restated there.

**The same wrong sentence is in THIS FILE, above the rule line** — *"the
branch adds 31 tests and removes none"*, in round 1's own "what shipped"
table. It is left standing there because a pre-registration is not
editable after the fact. This paragraph is the correction, and the rule
is why the correction lives here instead of up there.

## F5 — a boundary paragraph scoped to one dial

**REPRODUCED** (`scratchpad/pin/f5_repro.py`). With `x0` and `x1` each
declared as their own point, against `x0 + x1 >= 0.30000000000000004`:

    real : region_inhabited=False  narrowing_uncertified=True   status=unknown
    ieee : region_inhabited=True   narrowing_uncertified=False  status=violated-over-set

Sound under `ieee` — jax executes binary64, in which
`0.1 + 0.2 == 0.30000000000000004` exactly, while in ℝ `1/10 + 2/10 =
3/10` is strictly below it. The sentence is now scoped inside itself in
`SOUNDNESS.md` and in `_region_witness`.

**The unnamed other direction, measured over the whole corpus** (95 rows
× 2 assume modes = 190): **11 certify under `real` and not under `ieee`,
0 the other way**. The 11 are 4 `float32` rows, 4 `int32` rows, 2
`float64` `reduce_sum` rows and one nested-branch row — every one an
assume that narrows an over-approximated intermediate. On a separate
dtype grid the same split shows as `float32` and `int32` certifying under
`real` only, `float64` under both, `float16`/`bfloat16` under neither.
The `ieee`-only direction is the boundary point above, which the corpus's
grid does not contain.

## F6 — one search is capped, the other is not

**Non-contradiction: CONFIRMED, structurally and by measurement.**
`_region_witness` gets past its gate only when `narrowing_uncertified or
assume_dropped`; `narrowing_uncertified` implies `any_constrained` (one
assignment site, inside the `if narrowed:` block that sets it); and
`any_constrained or assume_dropped` is exactly when
`_reachability_witnesses` returns ∅ BEFORE probing. The certificate can
therefore only fire on runs where the reachability search certifies
nothing, which is why they cannot contradict each other and why a
branch-scoped violation is never restored by the certificate.

**The combined worst case DOES NOT REPRODUCE as posed.** The audit says a
query reaching both pays up to `16 + _certificate_probe_count(n)`
propagations. No query can reach both. Measured over **508 propagations**
— `corpus_pin.py` plus a size grid built to reach both, including 32 rows
putting a branch-scoped violation beside a narrowing assume and beside a
dropped one: **0 pay for both; worst combined probe count 16**; worst for
either search alone also 16. Were they NOT exclusive the sum would peak
at **32**, at small n where the certificate's budget is loosest — not the
19 you get by evaluating the expression at the size cap.

**The underlying complaint stands and is corrected.**
`_reachability_witnesses` is uncapped at any declared size
(`scratchpad/pin/f6_repro.py time`, jax 0.11.0, load 1.18 before / 1.16
after):

    n        propagate ms   bare walk ms   ratio   reach probes
    16              1.6           0.1      16.1x        16
    64              3.3           0.2      19.0x        16
    256             9.7           0.5      21.0x        16
    1024           34.1           1.5      22.3x        16
    4096          126.6           6.2      20.4x        16
    16384         549.9          25.7      21.4x        16

n = 16384 is four times the size cap and the probe count does not move.
The cost sentence in `SOUNDNESS.md` and the comment block at
`_CERT_MAX_ELEMENTS` now say which search they bind.

**DECISION: the older search is NOT capped on this branch, because
capping it moves verdicts.** Over 21 branch-violation rows at
n = 4 … 16384: **15 reachability keys asked, 3 lost** under a
`_certificate_probe_count` cap — the `x[0] > x[1]` guard at n ≥ 4096,
first certified by probe index 3 (the plain anchors put every element at
the same value and cannot witness a relation between two of them) against
a budget floor of exactly 3. Each loss is `violated-over-set` →
`unknown`. Safe direction, real cost; recorded rather than taken.

**A methodological correction inside this finding.** My first capping
instrument scored the keys the search FOUND and reported **0 lost**,
which is wrong by construction — a key that is asked and never certified
within the budget cannot appear in the found set. Scored on
`p.branch_violations`, the keys the branch pass ASKS about, the answer is
the 3 above. The wrong instrument is recorded because a zero from it
would have been a comfortable and false result.

## No verdict moved

See the entry below this one.

## Where to reproduce

* `scratchpad/pin/mutants.py build` — every mutation worktree, both
  families, from `0ad22bb`
* `scratchpad/pin/corpus_pin.py run OUT.json` / `diff A.json B.json`
* `scratchpad/pin/f2_repro.py`, `f5_repro.py`,
  `f6_repro.py count|time|capcost`
