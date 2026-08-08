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
