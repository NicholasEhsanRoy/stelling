<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## 0.2.0 — unreleased

### New transfers and precision improvements

- **`is_finite` transfer**: returns definite-true for bounded intervals,
  definite-false for point-at-infinity (`[inf, inf]`), unknown otherwise.
  Unlocks the `jnp.where(jnp.isfinite(x), ...)` pattern that MADDENING's
  Aitken relaxation depends on — `select_n` can now prune unreachable
  branches when the selector's `isfinite` result is decidable.

- **`int64→float64` point-interval conversion rule**: when an integer
  constant is cast to float64 and is exactly representable (in [-2^53,
  2^53]), the interval passes through instead of declining to top.
  Unblocks 41 jax-md `safe_mask` sites.

- **Boundary-aware division, REAL MODE ONLY, and only where a strict
  `assume` excludes the zero**: when the divisor has zero at exactly one
  boundary (`[0, hi]` or `[lo, 0]`) **and** a strict `assume` certifies
  the divisor is nonzero, compute a meaningful result instead of
  declining. True straddles and point-at-zero still decline with an
  actionable message — and so, since the B5-1 fix below, does a
  zero-touching divisor with no certificate.

  **The certificate, and what carries it.** `assume(d > 0)` narrows `d`
  to the CLOSED `[0, hi]` — an interval cannot hold an open bound — so
  the box alone can never say whether its zero endpoint is a value the
  program reaches. The propagator records the exclusion separately and
  carries it through `mul`, `div`, `add`/`add_any`, `neg`, `abs`,
  `square`, `integer_pow`, `reduce_sum` and `dot_general`, which is what
  keeps the row's headline shape — `assume(x > 0); 1 / jnp.sum(x*x)` —
  decidable in all four of its spellings. **A subtraction breaks the
  chain** (two positives can differ by zero), as does every primitive not
  in that list: those decline, naming the remedy.

  **Under `semantics="ieee"` the tightening is WITHDRAWN entirely**: an
  IEEE format has two zeros and an interval endpoint has no sign bit, so
  a divisor box reaching zero divides to `[-inf, inf]` there — and the
  transfer now says so, quoting `interval.IEEE_ZERO_DIVISOR_TOP` as its
  decline reason instead of returning ⊤ as an ordinary result. That ⊤ was
  counted "known", so a reader was told "none fell to ⊤ … compatible with
  a precision near-miss" about a `[-inf, +inf]` box while the same
  verdict's `top_despite_coverage` line named `div ×1`. See the S10 and
  B5-1 entries under Soundness fixes; the two kernels disagree
  deliberately.

- **Div-straddle decline**: when float division has a divisor spanning
  zero (true straddle), the transfer now declines with a message naming
  the interval and suggesting remedies, instead of silently returning
  `[-inf, inf]`.

### Float32 / float16 / bfloat16 IEEE mode

- **Format-parametric IEEE semantics**: the existing `semantics="ieee"`
  mode (previously binary64-only) now supports all four catalogued
  formats. Each operation rounds interval endpoints outward to the target
  format's ULP grid, models per-format subnormal flush, and handles
  format-specific overflow.

- **IEEE assume-bump** (`_format_nextafter`): `assume(x > k)` in IEEE
  mode narrows to `[nextafter_fmt(k, +inf), hi]` — the smallest
  representable value strictly above k in the target format. Works for
  all k, all formats. **The `assume(b > 0); a / b` pattern does NOT
  produce a decidable quotient in ieee mode** (it does in real mode):
  `nextafter_fmt(0, +inf)` is the format's smallest subnormal, which the
  DAZ haze immediately hulls back to 0, and a zero-containing divisor is
  ⊤ under ieee since the S10 fix. An assume whose bound is above the
  format's subnormal band (`assume(b > 1e-30)` in float32, say) keeps its
  quotient.

- **float16 and bfloat16 constants are readable** (audit 0.2.0 M12).
  `propagate._STRUCT_FMT` had no entry for float16's `<f2` or bfloat16's
  `<V2`, so every constant in those formats bound ⊤-maybe-NaN and *any*
  harness mentioning a scalar — including the ubiquitous
  `assert_(y > 0.0)` — answered UNKNOWN. Sound, and it made two of the
  four catalogued formats unusable for the ordinary shape of a harness.
  float16 decodes through `struct`'s `e` code (IEEE binary16, exact);
  **bfloat16 needs the aval**, because its dtype `.str` is `<V2` — an
  anonymous 2-byte VOID that every 2-byte structured dtype spells, so the
  byte string alone does not identify the format. The decoder therefore
  takes the aval's dtype NAME and reads `<V2` only under `"bfloat16"`;
  anything else stays ⊤-with-a-note rather than being read as a float.
  Verdicts move **UNKNOWN → VERIFIED/REFUTED** on float16 and bfloat16
  harnesses with constants, in both `real` and `ieee` semantics.

- **A mixed-format comparison gets the WIDEST operand band, never the
  alphabetically-first** (audit 0.2.0 M13). `_ieee_cmp_get_min_normal`
  sorted the operands' float dtypes and took `[0]`, and
  `bfloat16 < float16 < float32 < float64`, so a `{bfloat16, float16}`
  comparison was hazed with bfloat16's `2**-126` where the float16
  operand needs `2**-14` — 112 decades too narrow, and the band is what
  keeps a verdict sound for a flushing target. The rule is now a maximum
  over the operands' formats, which is sound for every one of them
  because the haze HULLS with 0 rather than replacing. Reachable only
  through hand-built or deserialized IR (jax promotes before it
  computes). The *arithmetic* face still declines a mixed equation, and
  the asymmetry is deliberate: an arithmetic result needs a grid to round
  onto, a comparison produces a bool and uses only the band.

- **The two mode-wide IEEE assumption stamps are format-parametric**
  (audit 0.2.0 M14). `IEEE_ENDPOINT_ASSUMPTION` and
  `SUBNORMAL_INDETERMINACY_ASSUMPTION` are binary64 sentences and were
  stamped verbatim on narrow-format verdicts, where both are false: the
  endpoints **were** outward-rounded to the target grid (that is the whole
  of `_ieee_round_box`), and the band applied was the format's, not
  `2**-1022`. The `semantics:` line disclosed the parametric mode
  correctly, so the two `assumes:` lines contradicted the line above them.
  Both sentences now name the formats the query contains and their own
  bands; a binary64-only run stamps the identical text it always did.
  Disclosure only — no verdict moves.

- **A binary IEEE kernel with no format-parametric row declines** (audit
  0.2.0 M15). `_ieee_arith`'s fallback used the binary64 kernel — whose
  haze band is `2**-1022` — for a narrow format, and `_ieee_round_box`
  afterwards **cannot** recover the missing haze: outward rounding onto
  the format grid does not hull with 0. Measured, float32 `x + x` at
  `x = 2**-140` came back `[1.4349e-42, 1.4349e-42]` where jax computes
  `0.0`. Dead today, and the hazard was that the fifth binary kernel
  registered without a `_FMT_BINARY_OPS` row would be a silent
  regression: `_FMT_BINARY_OPS` and `IEEE_TRANSFERS` are two hand-written
  lists that must agree, the coupling `affine.py`'s `AFFINE_SUPPORTED`
  already names as load-bearing. An import-time census now refuses the
  import when they disagree in either direction, and the runtime arm
  declines as a second guard.

### The eager construction-site detector (Mode 2), DEFAULT-OFF

- **`--stelling-eager-truncation=error` — an out-of-range integer constant
  narrowed at array construction now RAISES at the line that wrote it.**
  `jnp.full((), 256, jnp.int8)` is `0`: the 256 is destroyed before any
  primitive is bound, so the overflow tripwire — which watches jax's
  const-fold rule — never sees it, and no verdict downstream can tell that
  the `0` it certified was written as a `256`. `SOUNDNESS.md`'s
  integer-literal wrap entry is that defect and its cost is a wrong VERIFIED.

  Six of the seven `unwatched` routes in
  `tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` narrow at one line
  inside jax, and this attaches there: `jnp.full`, `jnp.full_like`,
  `lax.full`, `lax.full_like`, `lax.convert_element_type` and
  `jnp.stack`-of-`full`, plus `lax.select`-of-`full`, `jnp.take`'s
  `fill_value`, and a scoped `with jax.disable_jit():`. Two numpy routes
  remain and are named: `np.asarray(N).astype(dt)` is permanently unhookable
  (`np.ndarray.astype` is an immutable type attribute) and
  `jnp.asarray(np.array(N), dtype=dt)` is a second spelling into the same
  residue. `EAGER_COVERAGE`, beside `GATE_COVERAGE`, is the measured
  inventory and a test holds the residue to exactly those two.

  **It carries an ORIGIN QUESTION, and the first version of this entry said it
  needed none.** With `jit` on, jax's own threefry PRNG mask reaches the
  const-fold site and not this one, which is what that claim was measured on.
  With `jit` OFF — `jax.disable_jit()`, `JAX_DISABLE_JIT=1`, and the public
  `chex.fake_jit()` / `chex.fake_pmap_and_jit()` that install it — jax
  evaluates the mask eagerly and it arrives here as a written scalar, and the
  detector raised `4294967295 -> -1 (int32)` **inside jax's own PRNG**,
  naming a line the user never wrote a constant on. Measured on jax 0.11.0 and
  0.10.2, byte-identically, over a 32-workload census across 24 third-party
  packages: with `JAX_DISABLE_JIT=1`, **9 truncations, 9 fires before and 1
  after** (8 of them jax's own: `jax.random` ×4, flax linen, flax nnx,
  equinox, `chex.fake_jit`); with `jit` ON and jax's defaults untouched, **2
  truncations, 2 fires before and 1 after** — because `chex.fake_jit()`
  installs `disable_jit` around a test body, so this is reachable in the
  DEFAULT configuration through a public API. The one remaining fire in every
  row is a control of this repository's own that must fire. Over chex's own
  installed `fake_test.py`: **2 failed / 32 passed before, 34 passed after.**

  **The answer is a LOOKUP, not a predicate, and that is the second attempt.**
  The first asked a general question of the data — *is the narrowed integer
  among the arguments of the call that crossed out of non-jax code into jax?*
  — and an audit found it wrong in both directions: it **suppressed a constant
  the user really wrote** whenever the call carried it in a `functools.partial`,
  a `jax.tree_util.Partial`, a bound method, a closure cell or a
  registered-dataclass pytree — silently, in the DEFAULT `jit`-on
  configuration, on `jax.tree.map(partial(jnp.full_like, fill_value=300),
  tree)` and under `jit`, `vmap`, `lax.map`, `lax.scan` and `lax.fori_loop`
  alike — and it **raised on jax's own mask** whenever its container scan hit
  a depth, breadth or budget limit, which a params-shaped pytree does.

  A sweep of 649 conversions across `jax.random.*` and `jnp`'s integer ops
  over six integer dtypes, under `JAX_DISABLE_JIT=1`, finds **exactly one**
  eager truncation of jax's own in existence — re-derived as shipped code by
  `_adapter_jax.eager_jax_constant_sweep`, over a wider surface, at 675
  conversions and 13 truncation events all of which are that one row, on both
  jax series. So the one thing jax writes is
  written down, at jax's own site, in `_adapter_jax._JAX_EAGER_CONSTANTS`:
  `("_src/random/threefry2x32.py", "_threefry_seed") -> 4294967295, uint32
  into int32`. A narrowing is jax's when one of those functions is in the
  unbroken run of jax frames beneath the caller AND the value, the SOURCE
  dtype and the target dtype are that row's. Everything else is the caller's.
  That is the same shape — a narrow map plus a canary that reddens when it is
  incomplete — that `_KNOWN_HASHES` already argues for one screen up in the
  same file.

  **The source dtype is in the key because without it a row suppresses the
  CALLER'S constant.** At that one site the two collide:
  `jax.extend.random.threefry_prng_impl.seed(np.int64(2**32 - 1))` narrows
  twice under `_threefry_seed` — the caller's seed and jax's mask, both
  `4294967295 -> -1` at `int32` — and a three-field row suppressed **both**,
  then printed *"written by jax … the threefry PRNG's 32-bit mask"* at the
  caller's own line. It was a value collision and not a general quiet
  (`seed=8589934592` and `seed=2147483648` alarmed correctly throughout), and
  the two differ in exactly one field the hook can see: all 13 of jax's own
  truncation events arrive from `uint32` and a caller's seed from `int64`.
  Driven before and after on both routes into that entry point. What remains
  is the shape rather than the instance — a row is a value lookup, not a proof
  of authorship — and it is disclosed in `report.EAGER_UNCOVERED`.

  It **fails closed**: a jax release that adds a second internal eager
  truncation has no row, is therefore the caller's, and RAISES at a line
  inside jax rather than disappearing. Three things arrive first — the sweep
  runs as a test on both jax series, arming drives the row and reports
  `origin-blind:jax-attributed-to-you` rather than attaching if it stops
  holding, and the alarm prints jax's own frames and asks the reader to report
  it. It needs no container scan, so the depth, breadth and budget constants
  and the "inconclusive" bucket are gone rather than documented; and it
  removes the previously-disclosed false alarm on `jax.random.PRNGKey(2**32 -
  1)`, where jax's mask and the caller's seed are the same integer — a correct
  verdict about a program whose seed is nonetheless **already dead**, which is
  now disclosed rather than left to read as a clean bill of health.
  `PRNGKey` and `key` cast the seed with `jnp.asarray(np.int64(seeds))` inside
  jax's own `random_seed`, a NUMPY-level cast this detector has never sat on —
  `jit` on or off, before this work and after it — so the seed produces **zero**
  observations at the hook. Measured on jax 0.11.0, x64 off:
  `PRNGKey(2**32 - 1) == PRNGKey(-1)`, `PRNGKey(2**32) == PRNGKey(0)`,
  `PRNGKey(2**33 + 5) == PRNGKey(5)`. A seed that does not survive is exactly
  what this instrument exists to report and it structurally cannot report this
  one; closing it needs a hook at a numpy cast rather than at jax's array
  constructor.

  **The `jit` claim is now the narrow one.** *"A call boundary exists whether
  or not a trace is in progress, which is why the answer does not depend on
  `jit`"* was **false**: widened from four programs to fourteen,
  `jax.jit(partial(jnp.full_like, fill_value=300))(x)` gave no alarm with
  `jit` on and raised with it off, on the same observed conversion, because
  which frame is "the outermost jax frame" depends on how many wrapper frames
  jax installs. **And the sentence that replaced it carried a false clause of
  its own** — *"the verdict is a function of the value, the dtypes and which
  jax functions are in the run, and `jit` changes none of the three"*. The
  third clause is false, measured on jax 0.11.0 with one fresh subprocess per
  cell over 36 programs: of the 25 observations that occur in both modes, **6
  (in 5 programs) present a different run of jax frames**, and it differs in
  BOTH directions — `jit` on inserts tracing frames
  (`jit(partial(full_like, fill_value=300))(x)`: 8 frames on, 2 off) and `jit`
  off inserts jax's eager dispatch, which a trace does not contain
  (`jnp.take(x, [9], mode="fill")`: 25 frames on, 31 off). The real invariant
  is a **constraint on rows**: the verdict is stable across `jit` exactly when
  the function a row names is in the run under both modes or in neither, which
  is why the one row holds — `_threefry_seed` is a PRNG leaf that neither
  jit's machinery nor eager dispatch contains. A row keyed on a function only
  one mode's run contains would flip the verdict. Driven as an equality over
  **19 programs** covering `jit`, `vmap`, `tree.map`, `lax.map`, `lax.scan`,
  `lax.fori_loop`, five carrier shapes and two pytrees big enough to have
  exhausted the old scan's budget: 0 of 19 verdicts differ. Suppressions are
  counted and printed with their sites, their source dtype, the jax function
  that wrote them and what the constant is.

  **Off by default, and NOT turned on by `--stelling-overflow`.** Two dials,
  because the tripwire is a report over a session and this is a rule: a
  session it is armed on either contains no undeclared truncation or does not
  finish. With it off, nothing is patched, no jax is imported for it, and
  every program is byte-identical.

- **`stelling.intentional_wrap(value, dtype)` and
  `stelling.EagerTruncationError`, both public.** `intentional_wrap` returns
  the wrapped integer — `intentional_wrap(0xFF, "int8")` is `-1` — so the
  value that reaches jax is the value jax would have produced anyway, and a
  declared program is byte-identical to an undeclared one. It needs no jax
  and no numpy, and every declaration is recorded and printed with its site.
  The dtype is half the declaration, and what that buys is narrower than
  "a declaration used at a different width fires": measured over 98
  (declaration, misuse) pairs, 45 fire and 53 pass silently — but in every
  silent case the declared value is IN RANGE at the other dtype, so no
  narrowing happens there and no truncation is hidden. Writing the wrong
  constant is a bug this instrument does not claim to catch.

  **The exception inherits directly from `BaseException`**, so an ordinary
  `except Exception:` cannot swallow a soundness alarm — the handler shape
  that is everywhere in numerical Python. "Uncatchable" is not achievable and
  is not claimed; `design/eager-truncation-detector.md` carries the argument,
  the measured blast radius and the cost (cleanup written in
  `except Exception:` rather than `finally:` will not run). The radius, with
  the configuration it was measured in now stated: 122 module imports (174
  scalar integer conversions, 0 truncations) and 33 real workloads across 24
  third-party packages (264 conversions, 1 truncation — a control of this
  project's own that must fire and does) give **0 fires in any third-party
  workload with `jit` on**, every figure identical on jax 0.11.0 and 0.10.2.
  With `JAX_DISABLE_JIT=1`, a 32-workload re-derivation sees 9 truncations, 8
  of them jax's own, attributed and counted rather than raised on, and the one
  remaining fire is that same control. **Compare truncations and not
  conversions across those rows**: the alarm is a `BaseException`, so a fire
  kills the rest of its workload and stops its later conversions being
  counted, and a tree that fires nine times therefore reports a smaller
  denominator than the same tree that fires once.

  **There is no value-based carve-out and there will not be one.**
  `jnp.full((4,), 0xFF, jnp.int8)` and `jnp.full((4,), 255, jnp.int8)`
  produce identical observations at the hook, so intent is not a function of
  `(value, dtype, result)`. Two heuristics were driven over a corpus of real
  narrowings: "a value below the dtype's minimum is deliberate" hard-errors
  correct code 7 times and misses a real bug once; "an all-ones result is
  deliberate" is 5 and 2. And the corpus carries the PROOF rather than only
  the scores: `0xFF` and `255` into `int8` are the same `(value, dtype)` pair
  with opposite intent, so the class of value-based rules is empty rather than
  merely badly-scoring.

- **The dial reaches the exit code and the report on its own.** The eager
  detector's session escalation sat below the tripwire's `state.recorder is
  None` guard, so with `--stelling-overflow=off` — the spelling the docs
  recommend for running this detector alone — a rule that could not stay
  attached exited **0**. Under xdist, `pytest_testnodedown` returned on the
  tripwire's condition and dropped every worker's eager payload (`-n 2` on a
  fully green suite reported `NOT ARMED [no-worker-reported]` and exited 1),
  and `_capture_eager` then overwrote the merged worker snapshot with the
  controller's own zeros. All three are fixed and driven; a controller now
  prints its workers' figures and a single process's figures identically.

  **The first thing the reachable escalation caught was in this repository's
  own suite.** A canary test stubbed the tripwire's half of the canary and not
  the eager one, so `canary.main()` called the real `disarm_eager()`: a
  session run with `--stelling-eager-truncation=error` lost its detector at
  that test and ran every later file unwatched, printing `NOT ARMED
  [detached]` and exiting **0**. It now stubs both, and both files that drive
  the canary assert the process's arm state is what they found it.

- **`expected_truncation` is dynamically scoped, and now says so.** It was
  described as "lexically bounded" in four places. A `with` block looks
  lexical and no context manager can be: a region held across an `await`
  licensed a truncation in a SECOND asyncio task on the same loop. The region
  stack is now a `contextvars.ContextVar`, which isolates asyncio tasks as
  well as threads; a generator suspended inside a region still licenses its
  resumer, which nothing in Python fixes, and that residue is disclosed and
  driven rather than claimed away.

- **`_tripwire.arm()` on an already-armed process returns the recorder that
  is actually recording.** It returned a fresh, disconnected `Recorder`, so
  any assertion written against it under `-p stelling.overflow` was false by
  construction rather than by measurement.

- **It fails closed on drift.** It patches a private jax function, so arming
  verifies the module and attribute, checks `inspect.signature`'s first two
  parameters, and then drives EVERY construction route it claims in both
  directions. A route that stops reaching the site — the silent failure a jax
  release actually produces — is `route-blind:<route>` and it refuses to
  attach. Failure codes: `no-site-module`, `no-site`, `signature-drift`,
  `route-blind`, and `origin-blind:<leg>`. The nightly jax canary arms it,
  drives a live control both ways, and checks its own per-release source-hash
  map.

  **Arming drives the ATTRIBUTION too**, which the route probes cannot: they
  swap a collector in for the observer, so they exercise every route for
  reachability and arithmetic and never reach the function that decides
  whether a narrowing raises. A detector whose origin rule suppressed
  everything passed all of them. One narrowing of each origin now goes through
  the live policy at arm time — a constant at no enumerated jax site, which
  must raise, and `jax.random.key(0)` under `jax.disable_jit()`, which must be
  attributed to jax — and the control puts the user's counters back exactly as
  it found them, so a self-check can never appear in a denominator.

### Verification pipeline

- **`check(..., falsify="sample")` — the falsification probe, DEFAULT-OFF
  and UNRELEASED.** A new keyword on `stelling.preconditions.check`,
  `stelling.contracts.check_contract` and
  `stelling.inductive.check_inductive_step`. With the default `None`
  nothing changes: `stelling.falsify` is never imported and the verdict
  is byte-identical. Set to `"sample"` it runs, after a VERIFIED, the
  check this library has never had — it executes the real program at
  concrete points inside the declared set and tries to find one that
  violates a discharged obligation. `stelling` replays a REFUTED's
  witness through the real program; an `unsat` is a universal claim with
  no witness to replay, so a false VERIFIED had nothing downstream at
  all.

  Two properties are enforced rather than described. It **can only
  refute**: the note it appends is a sentence about work done and carries
  its own disclaimer, so a probed VERIFIED never reads as a better
  VERIFIED. And when it finds a violation it **raises**
  `stelling.falsify.VerifiedFalsified` instead of returning a status,
  because a discharged obligation the program violates is a defect in
  *stelling*, not a finding about the caller's code.

  Under `semantics="real"` a violation is admitted **only** by an exact
  test: exact **rational replay of the same traced jaxpr** at the same
  point (stdlib `fractions`; the probe imports no analysis module), or —
  where the *program* is integral throughout, meaning every operand and
  result dtype in the jaxpr at every depth and not merely the declared
  ones — exact integer arithmetic, which keeps its own branch because
  rational arithmetic does not wrap and routing it through the replay
  would suppress the runtime-wrap catch.
  **Everything else declines**, under `no-exact-reading-of-this-program`,
  with the reason the exact reading was unavailable counted by primitive
  in `ProbeReport.abstentions` and repeated in the stamp line. There is no
  fall-back: an alarm whose message is "stelling is UNSOUND" must not be
  admitted by a heuristic, and the ulp-stability proxy that used to sit
  behind the replay is gone from the firing path.

  That is a deliberate reach cost and it is measured rather than implied.
  A program with one step the replay cannot read cannot be fired on,
  however false its obligation is — irrational steps (`exp`, `log`,
  trigonometry, a fractional `pow`, a non-square `sqrt`) inherently, and
  `dot_general`, `sort`, `cumsum`, `stack`, `rem`, `scatter` and
  `scatter-add` because this module's tables have no reading for them
  yet. Three of the six live
  fixtures in `tests/test_falsify_probe.py` are `scatter` and now decline;
  they are listed there with the primitive that costs each one.

  Every admission is also downstream of the point being **admitted by
  every assume**, and that gate is a reading of the program that can be
  PARTIAL. The executed walk hands a call equation whole to jax, so a
  `stelling_assume` inside a `jit` or a `remat2` body executes without
  ever reaching the list the gate reads — and `propagate` narrows on that
  assume, so the probe was attacking points the analysis had claimed
  nothing about. The gate now declines (`assume-not-fully-executed`)
  unless the executed run saw every assume the program contains at every
  depth. Generalised rather than patched: every quantity the probe reads
  off the program is checked against a census taken at every depth before
  it may license anything, a declaration or an obligation the probe cannot
  see declines the whole probe by name, and a table (`_READINGS`) is held
  to the two dataclasses field-for-field so a new quantity cannot arrive
  without either a guard or a written argument that it needs none.

  The two walkers stay at **different depths on purpose**, and that is
  measured: `Primitive.bind` on a call equation compiles the whole body
  and XLA contracts across it, so a version that walked the body op by op
  computed different floats (5 disagreements over 22 one-line `jit`
  bodies, including sign disagreements) and raised "stelling is UNSOUND"
  on an obligation the real program satisfies. Each walker's reading is
  therefore checked against the census instead. The second
  reach-preserving alternative — keep the call compiled and thread the
  body's intermediates out as extra outputs — was driven too and is **0
  of 3 bitwise-identical** to the plain compiled call on the same
  fixture, returning exactly `0.0` where the plain call returns the
  rounding error of the product: exposing an intermediate is itself the
  change, because the value that has to be materialised is the one XLA
  was contracting away.

  **And the executed float is at the TRACE's granularity, which is not
  the program's.** `_execute` hands jax one equation at a time, so XLA
  never sees two of them together — and `jax.make_jaxpr` INLINES the
  `jit` that `jnp.mean` is built out of, so this is reached with no `jit`
  written anywhere. `jnp.mean` and `jnp.average` disagree between the two
  granularities on 70 and 72 of 200 random points (every other wrapper
  surveyed: 0 of 200), and four lines fire on a correct VERIFIED under
  `semantics="ieee"`, where *"the executed float IS the subject of the
  claim"* is what admits:

      X0    = 1.3102272059107631
      mean3 = lambda x: jnp.mean(jnp.stack([x, x * 2.0, x * 3.0]))
      C     = float(mean3(jnp.asarray(X0, "float64")))   # the program's OWN value
      x     = any_array((), "float64", (X0, X0)); assert_(mean3(x) <= C)

  So the granularity is measured the way the depth is: the same program
  is run at the same point as ONE compiled region
  (`_whole_program_route`), and an executed violation whose truth value
  moves between the two routes declines
  (`executed-float-depends-on-granularity`). The second route is
  consulted only after a violation and can only ever decline. Reach
  re-measured on 31 ordinary one-line `jnp` programs: identical, 31 of 31
  firing under `ieee` and 17 of 31 under `real`, base and fixed.

  **The exact-rational evaluator's ARITHMETIC is now pinned too.** Which
  primitive names the replay claims to read is checked against a live
  trace; what it reads them AS was checked by nothing, and that is the
  one direction in which this evaluator can INVENT a refutation rather
  than lose one. Three one-token mutations each raised "stelling is
  UNSOUND" on an obligation TRUE over ℝ with the whole falsify suite
  green — `math.trunc` → `round` in `_rat_convert`, dropping the
  integer-exponent guard in `_rat_pow`, `Fraction(math.sqrt(a))` in
  `_rat_sqrt` — and the `_rat_pow` one did it on a real `VERIFIED`
  through the public door, while the `_rat_sqrt` one survived the entire
  repository. The readings are now asserted against jax's own arithmetic
  where jax's answer is exact, and against their own algebra where it is
  not (`v * v == a` for a root, `v ** k.denominator == a ** k.numerator`
  for a power), with `_int_ok`'s boundary asserted at the point where jax
  actually wraps.

  Each `_READINGS` guard is also bound to the `if` that takes it rather
  than to the file: the table used to be satisfied by a new field
  declaring any decline reason spelled anywhere in `falsify.py`.

  A point the exact replay places **outside** the assumed region is no
  longer counted under `points_admissible`, and is not reported as a
  declined violation either. The stamp line could say "74 point(s)
  executed, 65 inside the declared set and admitted by every assume …
  declined 39 assume-unsatisfied-over-the-rationals" — a count that reads
  as coverage for 39 points no assume admitted.

  Blind spots, disclosed rather than discovered: the probe cannot see the
  `jnp.full((), 256, jnp.int8)` narrowing (there is no executable form of
  that program, traced or eager, in which `256` survives), it declines
  `bfloat16` and the `float8_*` formats outright, and its 5-second
  wall-clock backstop is thin enough — the deterministic element and
  width budgets already permit about 4.75 seconds — that **whether it
  fires on a given program can depend on the machine**. That bound can
  only decline, never admit, so what varies with the hardware is reach
  and never soundness.

- **Reachability conjunct**: a backward walk from the jaxpr's outputs
  identifies variables that flow to an output. Violated obligations on
  "dead" variables (computed but never observed by the caller) are
  downgraded from REFUTED to UNKNOWN with a note. The fail-safe is
  always REFUTED: obligations that cannot be proven dead keep their
  status.

- **Solver selection API**: `check(..., solver="z3")` or `solver="cvc5"`
  restricts the SMT portfolio to one backend. The verdict explicitly
  discloses degraded redundancy.

- **An `assert_` nested in a `jit` no longer declines solver escalation for
  every OTHER obligation in the query** (audit 0.2.0 **M17**). Escalation
  slices top-level `stelling_assert` equations, and it used to decide
  whether it could map obligations onto them by COUNTING: unequal totals
  meant nothing could be mapped, so *every* unknown obligation declined.
  One `assert_` written inside a `@jax.jit` helper — or a `cond` branch, or
  a `scan` body — therefore cost escalation for the whole query. This is
  the mechanism behind reports that "several asserts that each pass
  individually come back UNKNOWN together"; it was widely attributed to the
  per-obligation element budget, which was never involved.

  The count check was *sound* (equal totals really did mean index `k` is
  assert `k` — nothing was ever mis-sliced); it was simply a whole-query
  answer to a per-obligation question. The walk now records, per
  obligation, the position of the `stelling_assert` equation it came from
  (`ObligationReport.top_level_eqn_pos`, `None` for anything inside a
  sub-jaxpr), and `slice_unknown_obligations` VERIFIES that record against
  the query — the position must name a `stelling_assert`, carry the same
  `source_info`, and be claimed by exactly one obligation — before slicing
  by it. An obligation failing any of those declines individually with the
  reason quoted. The result is FINER than the count check — it answers per
  obligation what the count answered per query — and on the wrong-query
  attack it catches strictly more than the count did, but not all of it;
  see the narrowing under **M17′** below.

  **Measured** on a 246-harness / 684-obligation corpus of multi-assert
  queries with jit-nested asserts (jax 0.11.0, `JAX_ENABLE_X64=1`, z3 +
  cvc5 wheels), run before and after: **244 of 584 previously-undecided
  obligations became decided (41.8%)** — 123 discharged and 121
  violated-with-witness — with **0** regressions and **0** disagreements
  against an exact-`Fraction` oracle computed independently of stelling.
  109 of the 208 nested-containing harnesses moved UNKNOWN → REFUTED. The
  38 all-top-level control harnesses were byte-identical.

  **A nested `assert_` is still not sliceable**, and its own obligation
  still declines — with a reason that now names the actual cause instead of
  an arithmetic mismatch. Every obligation still undecided after this fix,
  on the corpus above, is a nested one.

### SMT emission extensions

- **`is_finite` emission** (guarded): emits constant `true` when the
  operand's propagated interval has finite endpoints; declines when
  infinite (sound: bounded reals are finite by construction). Unblocks
  solver escalation on every harness containing `jnp.isfinite()`.

- **`pow` emission** (integer AND non-integer exponents): integer
  exponents (`x**2`, `x**3`, `x**(-1)`) expand to explicit products.
  Non-integer exponents emit as auxiliary-variable polynomial constraints
  (`aux^q = x^p` with sign constraints) — both z3 and cvc5 handle these in
  QF_NRA. **The rational `p/q` must be the exact value of the traced
  binary64 literal**, which admits `x**0.5`, `x**0.25`, `x**0.75`,
  `x**1.5`, `x**(1.0/64.0)`, `x**(1.0/128.0)` — every dyadic — and
  declines `x**0.1`, `x**(1.0/3.0)`, `x**(1.0/80.0)` to UNKNOWN, because
  those literals are NOT the low-denominator rationals they are written
  as and emitting about a nearby rational is emitting about a different
  function. One cap (128) bounds the degree of the emitted equation on
  both sides, so a large numerator (`x**100.5` → `aux^2 = x^201`) declines
  exactly as a large denominator does. Base must be non-negative (JAX
  returns NaN for `pow(negative, fractional)`).

### Soundness fixes

- **The trace gate now consults the tripwire's displacement check.**
  `live_check() == "foreign-patch"` was a fourth way the gate's watch went
  partial and the gate consulted none of it: rebinding jax's const-fold
  registry entry over stelling's wrapper after arming leaves the recorder's
  identity unchanged and `fires_count()` unchanged, so both of the gate's
  existing partiality tests pass, the fire counter stays at zero because the
  wrapper is never called, and `check()` returned **VERIFIED on a route
  `GATE_COVERAGE` calls `watched`**. It now returns `UNKNOWN — trace NOT
  FULLY OBSERVED`, naming the displaced hook. One instrument,
  `_tripwire.displaced()`, answers for both hooks — the const-fold rule and
  the eager construction site — because two of them would be two chances to
  teach one caller about one hook and forget the other.

**Batch B15 — the trace gate observed part of a program and claimed all of
it** (`fix/B15-trace-gate-observation`). Branched from `a759809`.

- **A warm jit trace cache no longer hides a narrowing from the gate.** The
  armed gate traced through a fresh closure, which defeats
  `jax.make_jaxpr`'s identity cache and so guarantees the OUTER trace is
  re-run. It guarantees nothing about an inner `@jax.jit` helper, whose
  trace cache is keyed on the jitted callable and its avals: a helper any
  earlier trace already warmed is REPLAYED, the const-fold rule never runs
  over its body, and the gate's zero meant "I observed no narrowing" while
  being read as "no narrowing occurred". Measured on jax 0.11.0: one
  harness with a jitted helper, checked four times, gave `UNKNOWN,
  VERIFIED, VERIFIED, VERIFIED`; two DIFFERENT harnesses sharing one jitted
  helper gave `UNKNOWN` and then a **wrong VERIFIED** — about a program
  whose written 40000 had already been destroyed to -25536. `check()` and
  `check_contract()` now empty jax's trace caches before the trace they
  gate, so a verdict's observation is complete **with respect to jax's
  caches, in a single-threaded process**.

  **That qualifier is the claim, and both halves of it were measured.**
  `jax.clear_caches()` empties JAX's caches and nothing else, so a constant
  narrowed into a memo jax does not own survives it: measured on jax 0.11.0,
  `jax.extend.core.jaxpr_as_fun` over a saved jaxpr, a user
  `functools.lru_cache` holding an eagerly narrowed value, and
  `jax.closure_convert` — a **public jax API** that traces at setup and
  hoists the narrowed constant — each return VERIFIED with `fires=0` on a
  program whose executed values are `[-25536, -25436]` where the source says
  `[40000, 40100]`. And jax's trace cache is process-global while the gate's
  fire counter is per-thread, so the eviction-to-trace window is not atomic:
  over 400 gated checks of one harness whose narrowing sits in a shared
  jitted helper, wrong VERIFIEDs were **0/400 single-threaded** and
  **247/400 (61.8%) with four threads calling that helper while the gate
  traced**. The rate scales with the width of the window — how much the
  harness traces before it reaches the shared helper — so it is a range: with
  the same four threads, 1/100 when the helper is traced first, 52/100 after
  50 preceding primitives, 247/400 after 100, 100/100 after 200. Against
  399/400 single-threaded at `a759809`, the eviction is a large improvement
  and not a guarantee. stelling makes no
  thread-safety claim anywhere; run gated checks on one thread. Both sets
  are disclosed, with these numbers, in `report.UNCOVERED`.

  It is an eviction and not a detector because a detector was measured and
  does not work: `jax.jit(f, inline=True)` replays a warm body and leaves NO
  nested jaxpr in the enclosing jaxpr to notice the replay by, and jax
  publishes no per-jit trace counter on a public surface (`_cache_size` and
  `_clear_cache` are private, `clear_cache()` clears rather than reports and
  cannot be enumerated, and `jax.explain_cache_misses` logs MISSES where the
  state that matters is a HIT).

- **The gate has three states where it had two, and the third has its own
  sentence.** Observed-and-clean proceeds; observed-and-narrowed refuses
  with `trace unfaithful`; NOT-FULLY-OBSERVED refuses with a message that
  says no narrowing was seen AND none was seen not to be. The third state
  is reached when the cache eviction fails or when the tripwire is
  disarmed/re-armed mid-trace — the latter previously took the refusal path
  by way of `narrowings = max(narrowings, 1)` and printed *"1 integer
  narrowing(s) detected"* about a trace in which nothing was observed to
  narrow. The refusal was right and the sentence was wrong. When a
  narrowing IS seen while the watch was partial, the count is now published
  as a LOWER BOUND.

- **User-visible cost, and it is real.** While the tripwire is armed —
  opt-in, and nothing changes without it — every `check()` and
  `check_contract()` calls `jax.clear_caches()`, which is process-global and
  drops the caller's own compiled functions. **The call scales with how many
  jitted functions are live**, so a single "populated" figure is not a
  measurement: on jax 0.11.0 (median of 12) it is 0.049 ms empty, 1.4 ms
  with one live jit, 8.3 ms with ten and **41.8 ms (39.6–48.3) with fifty**.
  A caller then pays one re-trace-and-compile per jitted function it re-uses
  (18 ms trivial, 45 ms for a 32-step `scan`, 330 ms for a 200-primitive
  chain). Across this repository's suite and `corpus/` — 1475 armed gated
  traces — it was not measurable above the noise (522.0 s → 519.7 s), which
  is a fact about this suite holding few jits live across a `check()` and
  not a general one. Priced in `docs/overflow-tripwire.md`.

- **What the fix cost in verdicts: nothing.** Measured across the suite and
  `corpus/` at `a759809`: 1475 armed gated traces, of which 88 were NOT
  FULLY OBSERVED, of which **0** actually contained a narrowing — so 0
  verdicts were WRONG and 88 were merely UNGUARDED, and VERIFIED counts are
  312 before and 312 after. The eviction adds observation, not refusals.

- **The tripwire's coverage claim is now an asserted inventory.**
  `tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` declares a bucket
  for each of 32 constant-construction routes — 17 `watched`, 7
  `unwatched`, 3 `loud` (jax raises), 5 `deferred` (the constant reaches the
  jaxpr and the convert transfer declines it) — and the suite MEASURES every
  route by driving it through `check()` twice, comparing, and failing on a
  route whose two calls disagree. Driving it once was the shape that made
  this defect invisible. `report.UNCOVERED` and `docs/overflow-tripwire.md`
  gain the doors the sweep found unnamed (`lax.full`, `lax.full_like`,
  anything built on `full` such as `jnp.stack`, and values numpy narrows
  before jax sees them), and the warm-trace-cache row now records that the
  door is closed for a *verdict* and still open for the *session report*,
  which has no single moment that owns the whole program.

  The second call is a **regression detector for the eviction, not an
  independent control**: with the eviction in place both calls trace cold
  and always agree, so the `unstable:` bucket is unreachable — and against
  `a759809`'s `src` it reports exactly the regression it is for
  (`'@jax.jit helper': declared 'watched', measured
  'unstable:watched->unwatched'`). Its docstring says that now instead of
  implying a control.

- **Three disclosures that this batch narrowed, corrected where they were
  made.** Each is the same defect the batch exists to close, one layer down.

  - The claim the README retracted — *"a VERIFIED with the tripwire armed is
    a statement that the trace is faithful to what was written"* — was still
    standing verbatim in `design/d4-wrap-disclosure.md` (twice) and in the
    STRONGER form on the user-facing `docs/quickstart.md` (*"means both: the
    trace is faithful AND the property holds"*), plus in
    `tests/test_tripwire_gate.py`'s own docstrings, where the test that
    asserts the invariant drives ONE route (`x + N`, `watched`). All four
    now carry the qualifier the watched set requires.
  - `report.UNCOVERED` used to disclose *"anything traced BEFORE the
    tripwire was armed"*; the eviction rewrite replaced it with a narrower
    warm-cache bullet plus a completeness claim over the difference. It is
    **restored**, with the three constructs that live in that difference
    measured beside it.
  - `GATE_COVERAGE`'s comment said an added `unwatched` route "needs a line
    in `report.UNCOVERED` in the same commit, which the second test
    enforces". That test walked a six-entry dict literal typed beside the
    inventory rather than the inventory itself: measured twice, an added
    `unwatched` row passed all seven tests undisclosed. It iterates
    `GATE_COVERAGE` now, and `lax.full_like(x, N)` — a 32nd route found by
    the same sweep, `unwatched`, VERIFIED, `fires=0` — is in the inventory
    and named in the report.

- **`design/d4-wrap-disclosure.md`'s flagship worked example did not run.**
  It wrote the narrowing as `raw_adc + jnp.int16(40000)`, and
  `jnp.int16(40000)` raises `OverflowError` on jax 0.11.0 — so the example
  never traced, never produced the `add a -25536:i16[]` jaxpr it asserts,
  and never produced the refusal it quotes; this batch's own
  `GATE_COVERAGE` classifies `jnp.int16(N)` as `loud`, so the branch shipped
  an inventory contradicting its own narrative. The example is now
  `raw_adc + 40000` on an `int16` array, which is `watched` and does produce
  both, and the `loud`/silent contrast with `jnp.full` is stated where the
  reader meets it.

**Batch B13 — the instruments that read as enforcing something**
(`fix/B13-instrument-reach`). Branched from `3482822`. Nothing here is on
the analysis, transfer or emission path; the one user-visible change is a
refusal message, and it is here because the message was WRONG.

- **A verdict replayed on a different jax now says so.**
  `reproduce.write_reproducer` re-traces the subject and refuses when the
  stored `query_content_hash` disagrees. It reported that disagreement as
  *"this verdict is not about this subject's program"* — sending the reader
  to look for a program difference — when the actual cause could be that
  the jax version moved under an unchanged program. A jax bump can change a
  query hash without changing a verdict (jax 0.11.1 gave `reduce_max` and
  `reduce_min` an `out_sharding` param, measured), so this is reachable by
  doing nothing but upgrading jax. The refusal now ENUMERATES every
  difference the stamp can witness — jax version and precision config, both
  at once when both differ — instead of branching to the first one it
  finds. The direction is unchanged: every path still refuses, and a
  verdict that should replay still replays across a jax bump.

  `SOUNDNESS.md`'s entry for the query-identity break said there was
  nothing to fix in stelling and named no consumer. There is one, and it is
  this. Both sentences are withdrawn there.

- **The const-fold tripwire's "this release has never been read" carve-out
  no longer waves through real jax releases.** `is_release` treated a
  release as a bare `X.Y.Z`; jax has shipped `0.9.0.1` (and `0.0`, `0.1`).
  A four-component or post-release whose const-fold rule had moved landed
  in the never-read state and the suite stayed green against an unread
  rule. It now implements PEP 440's *final release*. Internal — no lane
  and no user-facing behaviour keys on it — but it is a coverage hole in
  the instrument that watches for jax moving underneath the tool.

**Batch B12 — the from_dict document-schema batch**
(`fix/B12-from-dict-structure`; audit 0.2.0 S15, S16). Branched from
`a4e4056`. Every figure below was measured on that tree and on this one,
in `/home/nick/venvs/stelling-jax` (jax 0.11.0, python 3.12.3), 2026-08-18.

- **`ClosedJaxpr.from_dict` now judges the TYPE the code declares at every
  position it stores a document value at** (audit 0.2.0 **S15** and
  **S16**; both reach the released **0.1.0** through `from_dict` and are
  reproduced at the tag — see [SOUNDNESS.md](SOUNDNESS.md)). `ir._encode`
  is a total function from IR to JSON with one arm per stored type, so the
  JSON type at every position of a document is what that function writes
  there; `_decode` judged almost none of it. Two false VERIFIEDs came out
  of that, both from pure JSON with no attacker Python:

  * **S15** — `<eqn>.primitive` had no type rule. `null`, `true`, `0`,
    `-1`, `1.5`, `[]` and `[0]` all loaded, were silently reclassified as
    an unknown primitive, and the `stelling_assert` the equation carried
    DISAPPEARED: a REFUTED two-obligation query returned VERIFIED with
    one. `_REQUIRED_PARAMS`' own comment says *"the primitive name is the
    semantic authority"*, and `_validate_loaded` type-checked it nowhere.
  * **S16** — `stelling_any`'s `lo`/`hi` had no type rule and no emptiness
    rule. A declaration of `(inf, inf)` — the empty real set
    `harness.any_array` refuses at the trace face in as many words —
    returned VERIFIED with 100% coverage; `"0.5"` and `true` loaded and
    moved the declared box; `""`, `"xx"`, `null` and `()` each raw-crashed
    out of the public `propagate()`; and `vacuity.widen` compared the two
    bounds RAW where every analysis reads `float(...)`, so
    `lo:"1.0"`/`hi:1.0` was a point by the reading that decides the
    verdict and not a point by the reading that decides whether to widen.

  **ONE RULE AT TWO DOORS**, because a document position is one of exactly
  two kinds. A DATACLASS FIELD's declared type is its own annotation, read
  with `typing.get_type_hints` and never listed. A SEQUENCE the reader must
  ITERATE in order to recurse has no field to carry an annotation — the
  container is a fact about the ENCODING, gone by the time a field exists —
  so it is judged by `_canonical_shell`, the container reader the module
  already had for the `params` sequence alone, asked now at every sequence
  position. Three leaves the reader consumes itself (`<array>.data`,
  `<complex>.re`/`.im`) have a type gate at the reader; the document's KEY
  SET is judged from the same field list. `lo`/`hi` are
  `_validate_decl_eqn`'s, beside the `shape` and `dtype` it already owned,
  and the value it validates is INSTALLED — which is what makes `vacuity`'s
  raw `!=` and `propagate`'s `float()` one read of one value.

  **THE PARTITION, before and after**, over the B12 census sweep of 5 base
  documents x every structural position x 12 values = **20,424 cells**:

  | | `a4e4056` | this commit |
  |---|---|---|
  | refused, `TranscriptionError` | 3,396 | **13,134** |
  | refused, the reader's 3 declared `ValueError` arms | 4,701 | 4,717 |
  | **RAW escapes** (uncatchable by either) | **4,879** | **0** |
  | accepted, faithful round trip | 5,488 | 2,489 |
  | accepted, declared canonicalization | 38 | 38 |
  | **accepted, SILENT** (re-encodes as something else) | **1,917** | **41** |
  | accepted OUT OF SCHEMA | 4,628 | **36** |

  The 41 silent accepts left are `<eqn>.source_info` and
  `<jaxpr>.debug_info` DELETED — metadata outside `content_hash`, whose
  absence is the form `to_dict(include_metadata=False)` writes. The 36
  out-of-schema accepts left are the two DECLARED canonicalizations:
  `<aval>.shape[*]: true` storing as `1` (30 cells, the "shape extents"
  entry) and `<complex>.re`/`.im` integer parts storing as `float` (6
  cells, the "complex parts" entry, ADDED here rather than repaired).

  **THE HASH.** Over the accepted population, metadata-free serializations
  reached by more than one distinct document fall from **117 to 6**, and
  the documents involved from **1,401 to 229**. Every one of the six is
  now explained by a written commitment: five are a base together with its
  own metadata mutants (the "hash scope" entry), two of those five also
  holding an `<aval>.shape[*]: true` and one a `<complex>.re: true`; the
  sixth is three DIFFERENT bases whose top-level `eqns` have all been
  emptied, which are then the same program and correctly share a hash.
  Checked mechanically rather than read: no residual class contains two
  documents differing at a position no `CANONICALIZATIONS` entry names.

  **NOTHING LEGITIMATE IS REFUSED, measured.** The census's population of
  **170 legitimate documents** — every zero-argument harness in the
  property corpus, the tag probes and five hand-built bases, covering all
  15 tags — **all 170 still round-trip exactly with `content_hash`
  preserved**, and the population statistics are identical to the
  baseline's (4,563 shapes, every container a `list` and every extent an
  `int`; 0 IR-side shape violations; the same tag histogram).

  **TWO CANONICALIZATIONS WERE WRITTEN DOWN RATHER THAN REPAIRED**, each
  with the witness `ir.CANONICALIZATIONS` requires: **"complex parts"** (an
  `int` or `bool` at `<complex>.re`/`.im` is stored as the `float`
  `complex(re, im)` carries) and **"array payload spelling"** (two base64
  spellings denoting one byte string are one document — base64's trailing
  bits are not part of the value, and neither `validate=True` nor
  `binascii`'s `strict_mode=True` treats them as part of it).

  **BEYOND THE CENSUS**, which swept single-position mutations only and
  said it expected a two-position sweep to find more. Two were driven.
  Every ordered pair of the two positions each PAIRING invariant compares,
  x 12 values at each (6 pairs x 144 x 2 bases = **1,728 cells**):
  `a4e4056` gives 366 raw escapes and 96 crashes inside `propagate()`
  after an accepted load; this commit gives **0 and 0**. And the full
  two-position product over the smallest base — every ordered pair of its
  130 positions x 6 values at each, **583,792 documents driven** —
  produces **164,366 raw escapes on `a4e4056` and 0 here**. Var-id
  ALIASING was driven too, a mutation the census's fixed value set cannot
  produce because it needs two ids to MEET: 56 documents, no false verdict
  on either tree — the two that turn a REFUTED base into VERIFIED re-point
  the second assertion's INPUT at a value carrying the first assertion's
  predicate (at that predicate itself in one, at the first assertion's own
  output in the other), so the document then asserts one TRUE predicate
  twice and VERIFIED is true of it as loaded. Both avals agree, so
  `_one_shape_per_value` has nothing to catch and is not being evaded.

  **THE RESIDUAL CLASS.** This rule judges the TYPE at every position and
  never the VALUE: `ir.py` scopes per-primitive shape inference out of the
  load door in writing, and a document whose primitive is a plausible but
  wrong NAME, or whose extents are integers that lie, is still admitted
  and is still the slicer's problem. **And the residue includes an
  UNCATCHABLE CRASH OUT OF A PUBLIC ENTRY POINT, which is a robustness
  regression against `SOUNDNESS.md`'s degrade-don't-crash posture and not
  only a precision one**: an ARITY the type rule cannot see — a well-typed
  but SHORT `<eqn>.invars` for a known primitive — loads, and then
  `propagate()` raises a bare `TypeError` (`gt() missing 1 required
  positional argument`, out of `propagate.TRANSFERS`' `"gt"` entry,
  `lambda eqn, p, ins: [iv.gt(*ins)]`) or `IndexError` (`list index out of
  range`, out of its `"stelling_assert"` entry, `[ins[0]]`), which `except
  TranscriptionError` does not catch. Six comparison witnesses
  (`gt`/`lt`/`ge`/`le`/`eq`/`ne`) for the first and `stelling_assert` for
  the second. **Cited by SYMBOL and not by line on purpose**: the first
  spelling of this sentence quoted two `propagate.py` line numbers, and
  both were pointing at unrelated code on `main` before this batch landed
  — `tests/test_prose_hygiene.py` only catches a citation past the END of
  a file, so a line that still exists and has become something else is
  exactly the claim nothing checks. Pre-existing and identical on
  `a4e4056`; this batch narrows the population that reaches it and closes
  none of it.
  `from_dict` also has two refusal
  SHAPES — `TranscriptionError` for everything this batch adds, and the
  reader's three older `ValueError` arms — and unifying them is a change
  to a public error surface that two tests pin, so it is reported and not
  made here.

  **THE ONE NARROWING OUTSIDE DOCUMENTS** is hand-built IR, where the rule
  is loud: `ir.JaxprEqn(source_info=7)` and an integer `lo`/`hi` are no
  longer constructible. The two slicer-totality tests that needed the
  first install it with `object.__setattr__` now, the way
  `tests/test_aval_lie_both_faces.py` installs a declaration lie, so the
  slicer is still measured with the door not in front of it; the test
  helper that built declarations with integer bounds records `float(lo)`,
  which is what `any_array` would have recorded anyway. The EMPTINESS
  refusal is on the LOAD path only, so the suite can go on building
  `(inf, inf)` and `(nan, hi)` declarations through the constructor — the
  two faces are asking about different things, and
  `_validate_decl_nonempty`'s docstring says which. **How much capability
  that protects is now MEASURED there and not named**: moving the rule to
  `JaxprEqn.__post_init__` turns **11 pre-existing tests red across four
  files** — 7 in `tests/test_ieee_semantics.py`, 2 in
  `tests/test_transfers.py`, 1 in
  `tests/test_ieee_zero_divisor_and_mul_exact.py` and 1 in
  `tests/test_undecided_detail.py`, which is where the `(nan, hi)`
  declaration actually is. This paragraph and that docstring both credited
  the whole of it to the ieee file alone.

  **Suite**: 3798/10 and 3799/9 at `a4e4056`, this batch's base; **3869
  passed / 10 skipped** with `JAX_ENABLE_X64=1` and **3870 / 9** without
  at the branch tip. **ON THE MERGE INTO `main`, which brings B11 with
  it: 3905 / 10 and 3906 / 9**, against `main`'s own **3834 / 10** and
  **3835 / 9** at `5f7168d`. Both pairs measured for this merge — the
  merged ones on the merged tree, `main`'s on a `git clone --shared` tree
  at `5f7168d` — because this batch's counts were taken on a tree without
  B11 and `main`'s on one without B12, and neither is a count of the tree
  that ships. The delta is exactly the 71 node ids of the new
  `tests/test_document_schema.py` — in both cells and against both bases,
  so the two batches are additive to the unit (3798 + B11's 36 + this
  batch's 71 = 3905); the skip sets are byte-identical to `main`'s in
  both environments and still differ from each other by exactly
  `test_tripwire_arm.py:643`. No pre-existing test changed status, in
  either batch's direction. **Each rule was reverted ALONE** and
  the new file re-run, so the coverage is attributed rather than assumed:

  **BOTH UNITS ARE GIVEN, because the file has 30 test FUNCTIONS and 71
  NODE IDS and a table in one unit beside a sentence in the other cannot
  be reconciled by a reader who is not told** — which is what this table
  did until B12's own review, in a batch whose subject is writing one
  identity across two faces.

  | reverted alone | test functions red | node ids red |
  |---|---|---|
  | the field-annotation rule (`_matches_spec` to "everything matches") | 7 | 13 |
  | the `lo`/`hi` TYPE rule and its install | 4 | 22 |
  | the `lo`/`hi` EMPTINESS refusal | 2 | 7 |
  | the sequence-container rule (`_doc_sequence` back to `tuple(v)`) | 3 | 3 |
  | the document-KEY rule | 4 | 4 |
  | the `<array>.data` / `<complex>` leaf gates | 2 | 2 |

  **THE ROWS DO NOT PARTITION AND ARE NOT MEANT TO**: they sum to 22
  functions / 51 node ids over a UNION of 15 / 44, because several
  functions are red under more than one revert; the control with nothing
  reverted is 0 / 0. **Every figure in the table, the sum, the union and
  the control were re-derived on the MERGED tree**, with the merged file,
  and every one is unchanged. *"Reverted alone" means BEHAVIOURALLY: the
  rule stops firing and its `_load_check` / `_doc_refuse` call stays where
  it is.* Deleting the body instead also deletes the call, which the
  load-only enumeration reads off the AST — that variant is +1 function
  and +1 node id on the three decoder rows, and it is the enumeration
  correctly reporting a rule that left the call graph rather than a rule
  that stopped refusing. On `a4e4056` itself the whole file is **55 of 71
  node ids red, which is 23 of 30 functions** — a figure the rows cannot
  be summed to, and not only because they overlap: 8 of those 23 are red
  under NO single revert, being the ones that read the new API's own
  correspondence (`_spec_of` over every field, `_doc_keys` against
  `_encode`'s own output, the load-only rules off the call graph) or the
  corrected sentences, none of which exist at `a4e4056` to read. Of the
  16 node ids green there, 11 assert that a legitimate document still
  loads, 3 are checks `a4e4056` already satisfies for other reasons, and
  2 pin `_encode`'s straight-through behaviour at its non-recursing
  slots, which is identical on both trees and is the point of them.

  **The message-totality control gained a THIRD knob**, and that is a
  finding rather than a maintenance chore. The field-annotation rule sits
  IN FRONT OF most of the quote sites `tests/test_ir_message_totality.py`
  measures, so with it shipped a hostile leaf is refused at its one message
  expression and six deeper ones are never composed: the door-removed row's
  per-message figure FELL from 9 to 5 while its escape count rose from 27
  to 87. That is the exact silent shrinkage that file's own docstring warns
  a one-knob control would suffer, arriving through a fix. `_neutered_sweep`
  takes `schema=False` now, the union the record quotes is taken over
  CONFIGURATIONS rather than over the deepest one, and the headline
  quote-site figure is **13** (was 11): 11 the sweep reaches in one
  configuration or the other, plus the 2 only the driven rows reach.

  **SIX SENTENCES THIS BATCH SHIPPED WERE READ AGAINST THE CODE BESIDE
  THEM AND CORRECTED**, five of them added by this batch (item 2 also
  correcting a pre-existing copy of the same claim) and the first one
  falsified by it without being touched. They are listed because the
  pattern — a repair whose own prose overstates it — is the one this
  campaign has caught over and over, repeatedly inside the fix meant to
  close the previous instance.
  1. *"`to_dict` / `from_dict` must round-trip losslessly"* was
     unconditional and is not true: `_validate_required_params` and
     `_validate_decl_nonempty` run on the LOAD path only, so their subjects
     are CONSTRUCTIBLE AND NOT RELOADABLE. The params-less form was already
     so at `a4e4056`; **this batch widened the class by two — `(inf, inf)`
     and `(nan, hi)` — which are exactly the declarations
     `_validate_decl_nonempty`'s own docstring promises stay
     constructible**. Fails closed, mints no verdict, leaves `content_hash`
     alone. The bound is now stated at both paragraphs and pinned by
     `tests/test_document_schema.py`, which enumerates the load-only rules
     from `ir.py`'s call graph so a third cannot arrive silently.
     **THE CALL GRAPH IS NOW THE WHOLE LOAD PATH.** That closure was seeded
     from `_validate_loaded` alone, so it never saw `_decode` — and five of
     `ir.py`'s own refusals live there. Seeded from both, driven with a
     synthetic third rule added seven ways: direct and via-helper were
     already red; decoder-side was GREEN and is now red; and the three
     edges an `ast.Call` walk cannot follow — a module-level alias, a
     dispatch table, a lambda — are red on a third assertion, that EVERY
     refusal in `ir.py` is reached by some closure. The seventh, a rule on
     the load path *and* a constructor, stays green, correctly.
     The decoder-side refusals are enumerated in their own bucket: they
     judge a DOCUMENT, never an object a constructor built, so they do not
     widen this bound.
  2. The field rule's widest exception was licensed with *"no document can
     reach one: `_decode` has no tag for it **and `_encode` refuses to
     encode one**"*. The second half is false: `_encode` refuses a
     registered value only in the arms where it RECURSES, and at **18
     measured positions** — `<eqn>.primitive`, `<aval>.kind`/`.dtype`/
     `.weak_type`, `<var>.id` and the rest, enumerated in `ir.py` — it
     writes the object straight through and `to_dict()` does not raise.
     The conclusion survives on `_decode` alone. **THERE WERE THREE
     PARAGRAPHS AND NOT TWO**, and this line said two: the third is the
     comment introducing `ir._LIBRARY_STORED_TYPES` — the first thing a
     would-be registrant reads, and above BOTH of the pair the first pass
     corrected (`_register_stored_type`'s docstring and the door narrative
     below `_encode`) — and it carried an extra clause that is more
     strongly false, *"outside `content_hash` and `to_dict` entirely"*.
     All three now rest the conclusion on `_decode`. TWO EARLIER LOG
     ENTRIES carry the original wording — this file's own *"THE DOOR'S OWN
     DISPATCH WAS BUILT FROM THE TWO MOST OVERRIDABLE TESTS IN PYTHON"*
     entry (audit 0.2.0 B6 audit 7, S14) and the 2026-08-15 B6 entry in
     `SOUNDNESS.md` — and both are marked in place
     rather than rewritten, the way this project has marked a rotted claim
     before (`SOUNDNESS.md`, the `Script.stamp_options` parenthesis: *"the
     wording is left standing because a log that edits itself is not
     one"*).
     **AND `content_hash` DOES NOT RAISE AT ALL 18** — it raises at 14 and
     answers at 4: `<eqn>.source_info[*]` and the three `<dbg>` slots,
     which are exactly the metadata `to_dict(include_metadata=False)`
     omits, so the hash is a correct function of a scope that deliberately
     excludes them. No soundness consequence; the two sentences that said
     *"`content_hash` does still raise"* unqualified are scoped.
     `tests/test_document_schema.py` drives all 18 and checks its own
     position set against `_encode`'s AST, so a slot added to the encoding
     later cannot go undriven and the enumeration cannot quietly grow. (A
     hand-written enumeration was wrong on its first attempt, in this same
     review — which is why it is now checked against the AST.) **THAT AST
     CHECK COMPARED BARE KEY NAMES**, so it delivered less than this line
     claimed: a new `<aval>.cls` slot was undriven and green, while a new
     `<aval>.zzz` was red. It now compares the full `<tag>.key`, in both
     directions, and it reads the two positions whose VALUE recurses and
     whose KEY does not — `<eqn>.params` and `<ntuple>.fields`, which its
     `"_encode(" not in unparse(v)` test dropped, leaving them hand-listed
     on both sides of the comparison, driven but never derived. **The pin
     is a SHAPE and no longer two literal strings**: it finds every
     paragraph of `ir.py` that argues the WRITING side excludes a
     registered value, and requires each to scope the claim. The literal
     pin missed the third copy because that copy says *"such a type"*
     where the two it was written for say *"it"* — a pin that lists
     spellings is the defect it is pinning, one level up.
  3. The revert table above was in test FUNCTIONS and the sentence beside
     it in NODE IDS. Both units are given now.
  4. `_doc_keys`' heading said *"THE LAST OF THE READER'S RAW ESCAPES"*.
     Scoped to the census sweep it measured: a
     `{"k":"tuple","items":[…]}` chain deeper than the interpreter's
     limit is still a raw `RecursionError` from pure JSON, on this tree
     and on `a4e4056` alike.
  5. The residual-class paragraph named a plausible-but-wrong primitive
     NAME and lying extents, but not the uncatchable crash out of
     `propagate()` that a short-but-well-typed `<eqn>.invars` still
     produces. Named now, in both logs.
  6. The scope argument for keeping the emptiness rule off the constructor
     credited the whole protected capability to
     `tests/test_ieee_semantics.py`, in `ir.py` and in both logs. Moving
     the rule to `JaxprEqn.__post_init__` in fact turns 11 pre-existing
     tests red across FOUR files, and the `(nan, hi)` form the sentence
     named is built in `tests/test_undecided_detail.py`. The capability is
     now pinned in one test that constructs a witness for each of the two
     refusals in each direction, so the argument no longer rests on
     filenames.

*The next two blocks are two independent soundness batches that branched from
the same commit (`dee8bc2`), were developed in parallel, and were merged into
`main` on 2026-08-16. **B7** landed on `main` first, at `198a2b5`; **B6** merged
on top of it, so B6 is the newer arrival of the two and leads them here.
Neither batch's figures were measured on a tree containing the other.
Where a figure survived the merge unchanged it is left as it was read; where the
merge moved it, the entry says so and carries the merged-tree value. B6's later
audit rounds continue in a second block at the END of this section, where B6
placed them.*

*The merged tree — `a4e4056`, which is also B12's base above, not this
commit — is **3798 passed / 10 skipped** with `JAX_ENABLE_X64=1` and
**3799 / 9** without it, as CI runs — zero failures in both, skip sets
unchanged and still differing by exactly `test_tripwire_arm.py:643`. The two
batches are additive to the unit: the shared base `dee8bc2` is 3453/3454, B6
adds 297 and B7 adds 48. Neither batch deleted or renamed a test of the other's.*

**Batch B6 — the plan-structure batch** (`fix/B6-plan-structure`; audit 0.2.0
S12, S12&prime;, S12&Prime;, S14, M17&prime;). Every figure in these six entries was
measured on a B7-free tree unless it says otherwise.

- **`dot_general` shape well-formedness is now ONE definition, shared by
  the interval transfer and the SMT emission** (audit 0.2.0 **S12**;
  reaches the released **0.1.0** through `ir.ClosedJaxpr.from_dict` — see
  [SOUNDNESS.md](SOUNDNESS.md)). `interval.dot_general` checked contracted-
  and batch-extent agreement and raised; `obligation._dot_general_plan`
  re-derived the same geometry from the **LHS alone**. On `lhs=(2,) @
  rhs=(4,)` the transfer refused the equation while the emission returned a
  two-term linear combination over a four-element constant operand —
  **dropped addends, with no decline** — and because a refused transfer
  binds ⊤, and ⊤ leaves the obligation `unknown`, the truncating plan is
  exactly what solver escalation then ran. Measured: the four-term sum lies
  in `[4, 8]` and `<= 4.5` does not hold; the truncated two-term sum lies in
  `[2, 4]` and it does — a **false VERIFIED**. On `lhs=(4,) @ rhs=(2,)` the
  same loop indexed off the end of the constant operand and raised a raw
  `IndexError` out of `slice_obligation`, whose caller catches only
  declines.

  Neither face owns a shape predicate now: both call
  `interval.dot_general_geometry`, which is the single definition of dim
  ranges, duplicate dims, list pairing, extent agreement, and the derived
  output shape and contraction ranges.
  `tests/test_dot_general_both_faces.py` asserts the two faces AGREE over
  a well-formed and a malformed half — agreement, not "the emission
  declines these forms", because two copies of a predicate that happen to
  match is the arrangement that produced the defect.

  **No traced query is affected**: jax refuses to trace the equation
  (`dot_general requires contracting dimensions to have the same shape`).
  **No well-formed query changes verdict**: the oracle refuses exactly what
  the transfer already refused. `from_dict` still accepts the document, by
  decision — `ir.py` scopes per-primitive shape inference out of the door in
  writing, and a rule there would leave the two faces free to disagree on
  any hand-built query.

  Also in this fix: `slice_obligation` can no longer raise. An unexpected
  exception becomes a quoted `internal error` decline (UNKNOWN), the same
  posture `solvers.escalate` already takes around `_dispatch_obligation`,
  and its range test is two-sided, so an index past the start of the assert
  list declines instead of raising `IndexError`.

  **The claim "the two faces cannot hold different opinions about whether
  an equation is admissible" was too strong, and the residue was a live
  soundness defect — see the next entry.**

- **The emission may not model a DIFFERENT ARRAY than the propagation did:
  one shape per value, checked for every primitive at once** (audit 0.2.0
  **S12′**; reaches the released **0.1.0** through
  `ir.ClosedJaxpr.from_dict` — see [SOUNDNESS.md](SOUNDNESS.md)). The S12
  fix above gave `dot_general` a shape oracle both faces call. **The oracle
  is shared; its ARGUMENTS are not**: `interval.dot_general` asks it about
  the shapes of the propagated BOXES, `obligation._dot_general_plan` asks
  the same function about the shapes recorded on the equation's INVAR
  AVALS. Leave the declaration and the constant operand alone, edit only
  those avals — which `from_dict` accepts — and the two faces disagree
  again, in the asserting direction.

  Worse than S12's own presentation, and this is what makes it hard to
  recognise: there the transfer REFUSED and left a ⊤ in the coverage
  record. Here it does not refuse. It agrees the contraction has four
  terms, prints the box `[4, 8]`, and the verdict comes back **VERIFIED at
  `4 eqns: 4 known (100%)`** on the claim `Σ <= 4.5`, whose truth in exact
  rationals is `8 <= 9/2` — false. The same lie also mints a **false
  REFUTED**, at a point where the predicate is true, carrying the sentence
  *"confirmed by independent exact-rational replay"* — honest about the
  arithmetic and false about the plan, because replay re-derives the same
  truncated plan. A witness is independent of the SOLVER, never of the
  plan, and that distinction is now stated where the claim is made.

  **It is a class, not a row**: `reduce_sum` truncates identically through
  `_group_reduce_sum`, and two further shapes reach it from inside a
  `jax.jit` body. So the fix is not a third `dot_general` shape rule but
  one cross-check in the slicer — `_Slicer._one_shape_per_value`, over
  every equation of every slice before any plan is built: **no equation may
  be modelled at a shape that disagrees with the shape the value actually
  has.** Two witnesses to "actually has", complementary because each is
  blind where the other sees: the value's BINDING SITE (needs no
  propagation, so it reaches inside transparent call bodies, where no
  interval environment holds a box at all) and the PROPAGATED BOX (the one
  witness a consistently-applied lie cannot forge, blind outside the top
  level). An operand the slicer cannot bind at all declines rather than
  passing.

  **And "the shape the value actually has" means the shape the EMISSION
  MINTS TERMS FROM.** The first spelling of the binding witness read the
  producing equation's outvar aval, which is the record of the binding for
  every producer but one: `slice` mints one SMT constant per element of a
  `stelling_any`'s **`shape` param**, never per element of that outvar's
  aval. A declaration saying four elements in its param and two in its aval
  therefore minted four symbols, summed the two the reference asked for,
  and came back `discharged` on `8 <= 4.5` — inside a `jit` body, where the
  box witness is blind by construction. The three sites in the emission
  path that need a declaration's element count — the budget, the
  input-term construction and the check — all call
  `_Slicer._declared_shape`, so none can implement a different rule from
  the others. **That is not sole readership and it is not a single read**:
  `propagate._declared_element_count` reads the outvar aval for the
  certificate search's cap (sound — the cap only gates whether the search
  runs, and the search re-derives its witness honestly), and each call
  re-reads the param, an object that answers differently between calls
  being caught by `ClosedJaxpr.content_hash()` rather than here. Both
  claims were made in the first spelling and both are struck.
  `ir._validate_decl_eqn` was closed alongside it: it compared a
  declaration's two self-descriptions only `if isinstance(shape, tuple)`,
  so a `list` skipped it entirely, and it now compares the extents whatever
  holds them and refuses a `shape` param it cannot read at all rather than
  passing it. **The door is not the containment** — a declaration with no
  `shape` param at all stays legal, as hand-built IR requires, and the
  slicer closes that form on its own.

  **Cost, measured** over every obligation slice the test suite builds, by
  a stated method: wrap the check, mirror its short-circuits, attribute
  every count to the test file that produced it, run the whole suite, and
  partition on `declines > 0`. The partition lands on exactly two files,
  and both hand the check malformed IR on purpose. Over the
  well-formed remainder — **74,330 equations; 86,792 operand references,
  all 86,792 with a binding found; 161,122 atoms, 160,137 of them with a
  propagated box** — **zero** disagreements on either witness and **zero**
  declines.

  *Those are the MERGED tree's figures, re-derived by the same method at
  the merge (parents `198a2b5` and `dd95333`). B6 published
  10,503 / 13,286 / 23,789 / 23,112 here, read on a
  tree without B7: the corpus this cost is measured over is the suite, and
  B7's `tests/test_pow_row_gauge_jax.py` multiplies it about sevenfold. What
  did NOT move is the finding — the disagreement and decline counts are
  IDENTICAL to B6's tip (15 binding and 9 box disagreements whole-suite, 0
  and 0 on the remainder, 13 declines whole-suite, 0 on the remainder) and
  the partition is the same two files. A cost that scales with the corpus
  and a zero that does not is exactly the shape this measurement was taken
  to show. The full table, both columns, all three trees, is in
  SOUNDNESS.md.*

- **`interval.dot_general_geometry` keeps its documented contract on
  non-integer `dimension_numbers`** (audit 0.2.0 **S12″**). A float or
  string dim passed the range test (`0 <= 0.0 < 1` is True) and then
  indexed a tuple with it, raising a raw `TypeError` — out of the public
  `propagate()`, since both consumers catch `IntervalError` and nothing
  else, and, on the emission side, as an *"internal error"* decline. The
  dims now go through `operator.index` first, exactly as `check_shape`
  already does for extents, **and are BOUND to what it returns** — the
  first spelling called it and discarded the result, so the dims were
  validated and never normalised, and an object that is indexable but
  UNHASHABLE (a 0-d `numpy` array) passed the guard and then raised a raw
  `TypeError: unhashable type` inside `len(set(dims))`, out of the public
  `propagate()`, while the emission declined: the same two-faces split one
  type level up. The returned geometry now holds plain `int`s. The crash
  was pre-existing; the docstring asserting it could not happen was not.

- **A guard that refuses a malformed extent can no longer be stopped by
  the extent** (audit 0.2.0 B6 **audit 3**, F1/F2/F3 — three shapes of one
  mistake, at the five `operator.index` sites the batch had touched). Each
  is in the safe direction and none moves a verdict; they are listed
  because "the guard is closed" was said about all five.

  - **A guard must PRODUCE the value it validated, not merely test it.**
    `_Slicer._declared_shape` called `_shape_problem(shape)` — which bound
    `operator.index(d)`, tested it and discarded it — and then RETURNED a
    second read, `tuple(_op_index(d) for d in shape)`. An extent answering
    `4` and then `-1` was validated at 4 and emitted as `(-1,)`, where the
    element budget takes a negative contribution and `range(-1)` mints no
    symbols at all. This is verbatim the defect the entry above fixed in
    `dot_general_geometry`, one module over, in the same batch.
    `ir._load_extent_problem` carried it too, in both its callers: the
    declaration door compared the RAW param objects with `==` after
    validating them through `__index__`, and the array length check re-read
    them with `int(d)`. All three now read once and hand back what they
    read, so every comparison downstream is `int`-to-`int`.
  - **`operator.index` raises whatever `__index__` raises.** Four guards
    caught `TypeError` alone, so a `ValueError` or `OverflowError` from a
    hostile extent left `ir.JaxprEqn(...)`, `ir.Aval(...)`,
    `interval.check_shape`, `interval.dot_general` and the public
    `propagate()` **raw**, while the emission face declined on the same
    object — the S12″ two-faces split, from a guard written as an
    enumeration of the exception types its author expected.
  - **A refusal message may not itself raise.** Two composers interpolated
    an unguarded `{!r}` of the object being refused, so a hostile
    `__repr__` turned a decided decline into *"internal error:
    RuntimeError: repr refuses"*. In `ir._validate_decl_eqn` it was worse
    than that: `_load_check`'s message is an ARGUMENT and is composed on
    the passing path too, so a **well-formed** declaration whose extent
    merely had a refusing `__repr__` raw-crashed the public constructor.
    Every such quote now goes through a placeholder-substituting read
    (`obligation._safely`, `interval._safe_repr`, `ir._safe_repr`).

  Re-measured over the malformed-`dimension_numbers` corpus that entry
  publishes, extended by the family it did not contain and driven through
  the public `interval.dot_general` on all three trees: **31 of 34 raised
  raw on `dee8bc2`, 6 on `d6b6d0b`, 0 on this tree.**

- **A declaration's `shape` param is accepted by a POSITIVE rule** (audit
  0.2.0 B6 audit 3). Both faces refused `str`/`bytes`/`bytearray` by name,
  because `tuple(b"34")` is `(51, 52)` — a pair of plausible extents the
  declaration never said. `memoryview` and `array.array` read the same way
  and were not on the list: the door ACCEPTED a `memoryview` shape param
  and the slicer sliced a four-element declaration off it. Adding two more
  names is "the container type I happened to enumerate", which
  `ir._validate_param_value` is annotated in this same batch as
  condemning, so the rule is stated the other way round: **a declaration
  records its extents in a `tuple` or a `list`** — the only forms
  `ir._decode` builds and the only forms jax's own params carry — and
  anything else declines. The character sequences fall out of it instead
  of being named by it, and so does whichever sequence type is noticed
  next.

- **`slice_unknown_obligations` can no longer raise** (audit 0.2.0
  **M17′**; a regression of the M17 fix above, caught and fixed before
  release). Its association check called `tuple(...)` on a `source_info` it
  had not established was iterable, and both callers (`solvers.escalate`,
  `affine.refine_propagation`) iterate this function **in the `for`
  header**, outside their own per-obligation nets — so on hand-built IR
  carrying a non-tuple there, `escalate` raised `TypeError: 'int' object is
  not iterable` and every obligation's verdict went with it. The comparison
  is now total (a non-frame-list means the association cannot be CHECKED,
  which gets its own decline sentence rather than the useless *"traced at 7
  but records 7"*) and the per-obligation body is netted **per obligation**,
  so a sibling still gets its own answer.

  Four totality claims in that repair were **not total**, and none of them
  moves a verdict — each is a raise where a decline belongs. `_frames`
  tested `isinstance(v, list)`, which a `list` SUBCLASS whose `__iter__`
  raises satisfies; the association net's own handler could raise while
  composing its message (`str(e)` runs the exception's `__str__`; `getattr`
  with a default swallows only `AttributeError`); a decline sentence read
  its claimant count a second time and raised `KeyError` printing it; and
  the docstring's *"the preamble cannot raise on any object"* is replaced
  with the true argument, which is that both callers read the same objects
  first and the residual is named.
  Also narrowed: the per-obligation association is **finer** than the count
  check it replaced, not *strictly stronger*. Two queries traced from the
  same factory carry identical `source_info` at the same position, so all
  three guards pass and the wrong-query slice comes out — as it did under
  the count. The containment is `make_solver_verdict`'s query-hash
  pairing, which is the same defence the count check had — **and that
  pairing binds the ESCALATION to the query and does not bind the
  PROPAGATION**, so it is not containment for a mispaired propagation.
  Measured on B6's tree, on `main` as it stood when B6 branched (`dee8bc2`),
  and on the released **0.1.0** — and re-verified on the merged tree, where
  `solvers.py` is byte-identical to B6's and `make_solver_verdict` is
  untouched by B7:
  `make_solver_verdict(query_B, propagation_of_A, escalate(B, p_A))`
  returns **VERIFIED** where `B`'s honest verdict is **REFUTED**, with no
  exception anywhere — `escalate` hashes the `closed` it was handed, so
  the gate sees a matching pair while `B`'s obligations are reported with
  `A`'s statuses. `carries_work=False` — an escalation with no records,
  no notes, no spawns and no stamps — exempts the gate entirely and
  reaches the same false VERIFIED with no solver record at all. The
  identity belongs on the `Propagation`, checked wherever a propagation is
  consumed against a query; that is cross-module work and was scheduled as
  its own change. **It landed in B11 below, and this residue is closed** —
  see [SOUNDNESS.md](SOUNDNESS.md) and
  `tests/test_verified_bar.py::test_a_mispaired_PROPAGATION_can_no_longer_mint_a_false_VERIFIED`.

**Batch B11 — the propagation identity** (`fix/B11-propagation-identity`):

- **SOUNDNESS FIX — a `Propagation` from one query could be stamped as a
  verdict about another, minting a false VERIFIED (and a false REFUTED) on
  the released `v0.1.0` and on every 0.2.0 revision up to `207faca`.** Audit
  0.2.0 B6 re-audit UNSOUND-3, disclosed by B6 and closed here. See
  [SOUNDNESS.md](SOUNDNESS.md) for the affected versions, the screen, and
  what to re-run.

  `MispairedEscalationError` bound the ESCALATION to the query. Nothing
  bound the PROPAGATION, because `Propagation` carried no query identity —
  and the discharges do not have to come from an escalation: an obligation
  the interval leg or the affine refinement decides outright arrives already
  `discharged` on the propagation and is reported by INDEX, with no solver
  record anywhere. Two queries traced from one factory carry byte-identical
  `source_info` at identical positions, so every structural check passes.

  **`stelling.propagate.Propagation` now carries `query_sha256`** — a
  REQUIRED field with no default, written at `propagate`'s single
  construction site — and `propagate.unpaired_propagation` is the one
  comparison read by all five sites that consume a propagation against a
  query: `verdict.make_verdict`, `solvers.make_solver_verdict`,
  `solvers.escalate`, `affine.refine_propagation` and
  `obligation.slice_unknown_obligations`. Each fails closed in its own
  vocabulary and none of them raises — the two assemblers return UNKNOWN
  with no obligations and the reason quoted
  (`verdict.unpaired_propagation_verdict`), the other three decline. The
  escalation gate still raises, and the propagation gate is ordered after it
  so an assembly wrong on both legs keeps raising.

  **API.** `Propagation` gains a required field, so a hand-built one now
  needs `query_sha256=`; `propagate()` fills it in. `propagate.__all__`
  gains `query_identity` and `unpaired_propagation`;
  `verdict.__all__` gains `unpaired_propagation_verdict`. No change to
  `check()`, `check_contract()`, `Verdict` or `Stamp`.

  **Cost**, measured on real queries rather than a microbenchmark: one
  `content_hash()` per `propagate()`. Median of n=200 `check()` calls after
  warm-up: **1.548 → 1.766 ms (+14%)** on the README example and
  **2.683 → 3.017 ms (+12%)** on `contracts.conditioning_2x2_field`; **no
  measurable change** on a solver-bound query (146.5 → 141.2 ms, n=15, bands
  overlapping). Both verdict assemblers and `escalate` reuse the hash they
  already took for the stamp, so the gates themselves add none — see
  [SOUNDNESS.md](SOUNDNESS.md) for the table and the attribution.

  **What is NOT closed, by name — FOUR arguments, not one.**
  `obligation.slice_obligation` takes `env`, `assert_position`,
  `top_primitives` and `relational_assumes` from its caller. None is bound
  to the query, and none is visible to the site derivation, because they
  arrive unpacked into scalars rather than as a `Propagation` — there is no
  object left whose identity anything could compare. A foreign `env`
  defeats the div-by-zero straddle guard and produces a slice where the
  honest pairing declines; a foreign `relational_assumes` is worse — it puts
  an axiom that is FALSE of the query being sliced into the emitted script,
  turning `sat` (REFUTED, with a witness) into `unsat` (DISCHARGED). Both
  are measured in
  `tests/test_propagation_identity.py::test_the_slicer_takes_FOUR_unbound_arguments_and_TWO_of_them_are_measured`,
  and the boundary — that each of the four library calls into a slicer
  supplies every channel it supplies from the query it is judging, the one
  exception being `slice_unknown_obligations` passing its own `env` parameter
  through, which is the declared channel — is DERIVED off the live source in
  `::test_NO_library_path_FORWARDS_a_slicer_argument_it_did_not_derive`. So
  the exposure stops at the slice, reaches no verdict, and a new library
  path that forwards any of the four goes red. See
  [SOUNDNESS.md](SOUNDNESS.md).

  **The gates fail closed on the objects they exist to refuse, and that is
  now driven.** Seven hostile identity shapes × the five sites = 35 cells;
  four of them raised a raw exception instead of refusing — a
  `Propagation.__new__` with no fields took `slice_unknown_obligations` and
  `refine_propagation` down on `obligations` and `make_solver_verdict` down
  on `coverage`, and a non-dataclass took `refine_propagation`'s
  `dataclasses.replace` down with a `TypeError`. The pairing gate in
  `slice_unknown_obligations` is now that function's FIRST statement rather
  than the line after the read it guards; `affine` builds its declines
  without a bare `replace` and records a whole-run refusal on the new
  `RefinementReport.declined_wholly`; and `make_solver_verdict` reads the two
  propagation fields its escalation gates consult ONCE EACH through a net, an
  unreadable one refusing rather than raising. All 35 fail closed
  (`::test_EVERY_site_fails_CLOSED_on_EVERY_hostile_propagation_shape`). No
  verdict changes: nothing here is reachable without hand-building a
  propagation.

  **The site derivation is driven backwards.** The AST rule that enumerates
  the five sites recognised a query as "annotated `ClosedJaxpr` or NAMED
  `closed`" and checked for the gate by substring over `ast.unparse(node)`,
  which keeps docstrings — so three of the five gates could be deleted
  outright with the test still green, and a new assembler with an unannotated
  query parameter was invisible. The rule is structural now (both halves
  seeded from the classes and closed under calls in both directions), the
  check is an `ast.Call` node, and the claim is narrowed to what it verifies:
  a function that holds a query and READS the propagation. Each of the five
  deletions and both injections are re-run as tests.

  **Also recorded, as a decision rather than an oversight** (2026-08-18):
  root-object canonicalization — a `ClosedJaxpr` subclass answering `jaxpr`
  from a property, one of the three routes past `ir`'s door that B6
  disclosed — is OUT OF SCOPE by explicit ruling, *"It can be addressed with
  proper CI security on projects that need it. It requires actively
  malicious python to actually happen."* The route is still true of the tree
  and its driven test is untouched; only its status changed. See
  [SOUNDNESS.md](SOUNDNESS.md) and
  `tests/test_canonicalization_routes.py::test_the_ROOT_route_is_recorded_as_a_DECISION_not_as_an_OPEN_item`.

**Batch B7 — the `pow`-row bar and gauge batch** (`fix/B7-bar-gauge`, landed on
`main` at `198a2b5`; audit 0.2.0 M10, S4). Every figure in these eleven entries
was measured on a B6-free tree unless it says otherwise.

- **The VERIFIED bar's re-derivation is given the query's forwarded
  relational assumes** (audit 0.2.0 **M10**; a verdict flip, `UNKNOWN` →
  `VERIFIED` — see [SOUNDNESS.md](SOUNDNESS.md)). `verdict._bar_scope`
  narrows the bar to the decided obligations' own slices only when a recorded
  invocation reproduces both the slice's fingerprint and the script it emits.
  It re-derived the slice without `relational_assumes`, and `smt.emit` takes
  its axioms from `sl.assumes` and from nowhere else — so on any query with a
  forwarded relational assume the re-emitted script was the recorded one minus
  its `(assert …)` axiom lines and an HONEST record failed to be recognised,
  widening the bar to the whole query. The assumes are now re-derived from
  `closed` (never read off an argument: `make_solver_verdict`'s `propagation`
  is not bound to its query by the pairing gate). Conservative before and
  after — the fix can only NARROW — and the residue it does not close, a
  propagation re-run in `propagate`'s default configuration rather than the
  caller's, is disclosed at `_bar_scope` and in SOUNDNESS.md.

- **The `pow` emission row has a fidelity gauge, and stays out of the
  VERIFIED bar** (audit 0.2.0 **S4**). The row's emission now goes through
  three named seams — `smt._pow_integer_body`, `smt._pow_rational_lines`,
  `smt._pow_aux_name` — extracted behaviour-identically so that a mutation
  battery can express an emitted-`pow` wrongness at all;
  `tests/test_pow_row_gauge_jax.py` runs 32 such mutations across both
  exponent branches with zero survivors. One of them declares a single
  auxiliary constant for two elements of a vectorised `pow`, which is
  well-formed SMT-LIB2, collapses `sqrt(x0) - sqrt(x1)` to zero, and silently
  DISCHARGES an obligation that is false at `x = [4, 1]` — the
  missed-violation direction, caught by three of the battery's gates and by
  nothing in the tree before this batch. `tests/test_bar_membership_policy.py` carries the decision not to
  bar `pow` or `is_finite`, the reading of the standing rule it rests on, and
  the cost of the alternative measured on two corpora rather than estimated.
  `docs/gauge-coverage.md` states what the gauge reaches and what it does
  not, including `pow`'s integer-dtype guard, which is recorded UNCOVERED
  because no jax program can reach it.

- **That gauge was overfit to the exponents it drove, and now states its
  arity as a MEASUREMENT.** A blinded adversarial audit instrumented the two
  seams and found the shipped battery reaching integer exponents `{-2, 3}`
  and the single pair `(1, 2)` — an unstated SCOPE, since "both exponent
  branches" is true of the branches and says nothing about the exponents. It
  then wrote three wrongnesses conditioned OUTSIDE that set (wrong only above
  degree 3, wrong only for `q >= 4`, wrong only for `p != 1`); all three
  passed every one of the fourteen gates, and each turned a genuinely REFUTED
  query into VERIFIED on an exponent the admission guard admits — `2^6 = 64`
  under a bound of 40, `81^(1/4) = 3` under 2.9, `4^(3/2) = 8` under 7.9.
  Conditional wrongness is the shape a REPAIRED row regresses in, which is
  the shape a one-point-per-branch battery cannot see.

  The battery now drives integer exponents `[-4, -2, 3, 5]` (both signs at
  both magnitude classes, because `_pow_integer_body` reads sign and
  magnitude separately) and the pairs `1/2`, `1/4`, `3/2` (three of the 448
  admitted pairs, because `_pow_rational_lines` reads `p` and `q`
  separately). All three mutations are in it, so the widening is pinned and
  not merely applied once. **The arity is derived from the fixture table and
  measured at the seams** —
  `test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`
  instruments all three seams, runs every gate against the baseline and fails if
  the reach is not exactly the declared set — so the SCOPE the gauge prints
  cannot go stale the way the prose version did.

- **`docs/gauge-coverage.md`'s coverage figures are recomputed from the
  battery instead of typed.** The page said *"Sixteen of the seventeen are
  caught by more than one gate"* while the table printed directly beneath it
  already showed five single-covered entries; the measurement was **six**.
  Every other cell re-derived exactly — only the summary was wrong, and it
  was wrong in the direction that made the gauge look stronger, since the
  page's own premise is that *"a gauge with one single-covered mutation is
  one edit from a hole"*. Correcting the digit would have left the class
  alone, so the digits are now parsed out of the page and compared against a
  live gauging run by
  `test_the_documented_coverage_figures_are_the_MEASURED_ones`: the battery
  size, survivor and asymmetry counts, the multi/single partition, the
  single-covered set BY NAME, and every gate the mutation table names
  including each `ALONE` exclusivity. The bare gate counts that column
  carried are deleted rather than corrected — five of them were stale too.

- **`pow`'s rational branch no longer has an unreachable arm that reads as
  covered.** The gauge claimed the branch covered "`q` even **and** odd", and
  odd `q` was structurally unreachable: `obligation.pow_exponent_rational` is
  `Fraction` of a binary64, every finite binary64 is a dyadic rational, so in
  lowest terms `q` is a power of two and `q == 1` takes the integer branch.
  Measured: `q` over the whole 448-pair admitted set is exactly
  `{2, 4, 8, 16, 32, 64, 128}`, 0 odd in 500 000 random draws — so
  `smt._pow_rational_lines`' `if q % 2 == 0:` had a dead `else`, presented in
  its docstring as one of two live cases. An untested branch that READS as
  covered is worse than no branch, so the repair is enforcement rather than a
  corrected sentence, at all three altitudes: the DERIVATION refuses to
  return a non-dyadic rational (so a widening cannot happen quietly), the
  ADMISSION guard `rational_pow_problem` DECLINES an odd denominator (a
  decline belongs where an UNKNOWN can be returned), and the EMISSION refuses
  one with the root guard now unconditional (emission can only write a script
  or refuse). `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED`
  pins the unreachability and all three refusals on the standard
  `pow`'s integer-dtype guard already gets. No behaviour changes for any
  exponent jax can produce; the emitted text is byte-identical. Three PROSE
  sites went on describing the odd arm as a live case the encoding handles —
  an admission comment in `obligation.py` on a path where odd `q` declines
  several lines earlier, `_negative_base_harness`'s docstring, and a 0.2.0
  regression docstring. All three now say what is actually true there (`q` is
  even, so the encoding has NO solution at a negative base, and the guard
  stops a trivially-unsat negation coming back VERIFIED), and
  `test_no_source_text_presents_the_ODD_q_ARM_as_a_live_case` scans the tree
  for the claim SHAPE rather than for those three sentences.

- **That gauge measured two of its row's three seams, and the SHAPE axis was
  the one left as prose.** The round above instrumented `_pow_integer_body`
  and `_pow_rational_lines` and left out `smt._pow_aux_name` — the only seam
  handed the array shape — then wrote "any other array shape" into the
  not-reached list. A blinded audit conditioned a mutation on the ELEMENT
  INDEX (correct at elements 0 and 1, the whole of what the two-element
  vector fixture drives; wrong from element 2 on, where it emits
  `aux^6 = x^1` for `x^(1/4)`). It passed all twenty-one gates and minted a
  false VERIFIED: on `x[2]**0.25 - x[1]**0.25 <= 1.9` over `[1, 81]` the truth
  is `81^(1/4) = 3` less `1^(1/4) = 1`, which is 2. The measured/asserted line
  ran through the middle of the instrument.

  `_measured_seam_reach()` now instruments all three seams and the
  `(element, n_out)` pairs are compared against a `DRIVEN_AUX_ELEMENTS` read
  off the fixture table, exactly as the exponents are. **What closes the axis
  rather than sampling it is a new gate**, added here as
  `emission-is-invariant-to-the-array-shape` and renamed further down this
  section to `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` when it
  took on the base-term axis too: `pow` is elementwise, so an
  element's emitted lines cannot legitimately depend on which element it is
  or on how many there are, and the gate asserts that every element's seam
  output at every count in `[1, 2, 3, 4, 5, 6]`, on BOTH branches, is the
  SCALAR output with the symbol names substituted. Emission is text, so a
  shape-conditioned emission wrongness IS a per-element difference in that
  text — which makes the gate evidence about every conditioning function
  inside the range instead of about one more sampled point. It needs no
  solver. The audit's mutation is in the battery and is caught by that gate
  ALONE; delete the gate and `fidelity.gauge` refuses the run with an
  unexplained survivor. The claim is now split honestly in `SCOPE` and on the
  page: element counts `[1, 2, 3, 4, 5, 6]` reach the seam, `[1, 2]` are
  driven end to end to a VERDICT, and every NON-emission stage past two
  elements — transfer, slice, dispatch, replay, verdict — is a sample, said
  to be one.

  The same page and the gauge file now also record what the mechanism cannot
  do: the equality between declared and measured reach is DRIFT protection,
  the anti-vacuity floors are the only part carrying new coverage, and every
  floor is typed at the radius of a mutation someone already wrote. The shape
  invariance closes a RANGE rather than sampling it; its anti-vacuity floors
  are still typed at the radius of the mutations already written.

- **And then the radius moved onto the seam's OTHER ARGUMENT: the BASE TERM.**
  `smt._pow_integer_body(term, exp_val)` takes two arguments and the gauge
  measured the joint reach of one of them. Every fixture in the battery raised
  a DECLARED PROGRAM INPUT to a power — measured across all 22 gates, the
  base-term prefix reaching either exponent seam was `x` and nothing else — so
  a wrongness conditioned on "the base is not an input" was correct on every
  fixture the instrument had. A blinded audit conditioned two mutations that
  way, at a DRIVEN exponent, element 0 and `n_out` 1, and both survived all 22
  gates and minted verdict-level false VERIFIEDs on ordinary jax:
  `(x+1)**3 - (x+1) <= 23` over `[1, 2]` (truth 24) and `(x+1)**0.5 <= 8.9`
  over `[0, 80]` (truth 9), REFUTED becoming VERIFIED on both branches.
  Enumerating the axis properly found three more: the base can also be a
  rational `pow`'s own AUXILIARY (`(x**0.5)**3 <= 7.9` over `[1, 4]`, truth
  `4^(3/2) = 8`, likewise REFUTED → VERIFIED) or an inlined NUMERAL. Five
  survivors in total, none of them the disclosed exponent radius. **Real
  programs almost always `pow` a computed quantity.**

  The base's SPELLING is a coordinate of the joint reach now —
  `(element, n_out, base_kind, exp)` and `(element, n_out, base_kind, p, q)` —
  and the invariance gate sweeps the three spellings `smt.emit` writes for a
  symbol (`input`, `intermediate`, `auxiliary`) against a reference shared
  with the shape sweep, so what is closed is the PRODUCT and not a fourth
  marginal. The gate is renamed
  `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` to say so. The joint
  reach went from 84 and 63 tuples to **252 and 189, still the full product**;
  four battery mutations conditioned on the base kind at a driven exponent pin
  it, each caught by that gate ALONE. `_element_index_of` no longer guesses
  the element index off a trailing `_<digits>` — it parses the emitter's
  naming grammar, which the old regex got wrong for the auxiliary branch's
  single-element spelling (`aux_2` is element 0 of output 2, not element 2).
  A base spelling the grammar does not know is reported as an INCONSISTENCY
  rather than counted, so the NUMERAL case is disclosed in both not-reached
  lists instead of being absorbed into a driven set. No source behaviour
  changes: the seams were already correct, and the whole repair is in the
  instrument.

  **What is meant to end the pattern is not the fourth axis but that the axis
  LIST is no longer written from memory.** A seam is a pure function of its
  arguments, so its arguments are the complete set of things a wrongness in it
  can be conditioned on.
  `test_every_SEAM_ARGUMENT_is_a_gauged_COORDINATE_or_a_named_exemption` reads
  all three seams' signatures with `inspect` and requires every parameter to
  be classified. A future round that adds
  an argument to a seam fails that test instead of shipping an axis nobody
  enumerated. It says nothing about whether a coordinate is swept WIDELY
  enough — the exponent radius is still a finite set of points and the NUMERAL
  base spelling is outside the driven range of `base_kind` — which is the
  half the entry below had to repair.

  **Running the enumeration corrected its own first draft, which is the
  strongest thing that can be said for it.** `_pow_aux_name`'s `out_id` — the
  auxiliary's freshness across two `pow` OUTPUTS rather than across two
  elements — was written down as an undriven residual, on the reasoning that
  no fixture holds two rational `pow`s. Measured, that was already false: this
  round's `auxiliary` probe arm is `(x**0.5)**e`, two rational `pow`s and so
  two `out_id`s. Two distinct values reach the seam across the battery and a
  mutation dropping `out_id` from the name — colliding two outputs'
  auxiliaries while keeping them fresh per element — is CAUGHT. The
  disclosure would have shipped false on the day it was written; `out_id` is
  classified DRIVEN and pinned by
  `emit-rational-aux-collides-across-two-pow-OUTPUTS`.

- **And then the LIST was closed and every RANGE was still open, which is the
  defect the enumeration itself shipped.** A wrongness is conditioned on an
  argument's VALUE, not on its name, so enumerating the parameters closed the
  list and left every parameter's range exactly as open as before. Two entries
  said otherwise. `out_id` was classified DRIVEN under a definition written as
  a UNIVERSAL — *"reaches the seam at more than one value AND a wrongness
  conditioned on it is CAUGHT"* — on a measured reach of `{2, 3}`, of which
  only `2` reaches any verdict-producing gate; what held was the EXISTENTIAL,
  and mutations conditioned on `out_id >= 4`, `>= 5` and `>= 6` survived all 22
  gates. One mints a verdict-level false VERIFIED on three ordinary jax
  operations: `y = (x+1)-1 ; r = y**0.5 ; r[0]-r[1] <= 7.9` over `[1, 81]^2`,
  where the exact truth is `9 - 1 = 8` and the mutated encoding cannot exceed
  `81^(1/4) = 3`. And `aux_name` was classified DERIVED because it is *"a pure
  function of parameters already classified, SO it carries no axis of its
  own"* — the same `so` this batch had already corrected in the monotonicity
  sentence. Derived-ness closes the LIST; the RANGES compose, and a product
  over `element` and `n_out` says nothing about `out_id`.

  The same defect ran one seam over on an argument nobody had called an id:
  both exponent seams read a BASE TERM, and `x0`, `t2`, `aux_2` spell an id
  exactly as `aux_2_0` does. `_base_spelling` keeps the KIND and throws the ID
  away, so three spellings were swept while the ids reaching the seams were
  `{0, 2}` — and `emit-integer-wrong-only-at-a-LATER-BASE-ID` and its rational
  twin survived every gate too, the second minting the same false VERIFIED.

  **The repair is an INVARIANCE over a printed range, because a range is what
  a sample cannot close.** `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT`
  now asserts that the block a seam returns is the REFERENCE block with its
  symbol names substituted, at every id in `[0, 15]`, for every symbol
  spelling, at every driven shape and every driven exponent — one reference
  shared with the emitted half, so it is one claim and not two marginals. The
  FRESHNESS claim needs its own gate, and that is the finding rather than an
  economy: the invariance comparison canonicalises the auxiliary's name to
  `AUX`, so a wrongness that changes only the NAME is invariant there by
  construction. `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` asserts
  INJECTIVITY of the naming seam over the same range — two elements of one
  output never share a name, two outputs never share one — and, end to end,
  that every `declare-const` in an emitted probe script is declared once.
  Writing it corrected a sentence in the process: `_pow_aux_name` is NOT
  injective over its own signature, since `aux_{out_id}_{element}` does not
  spell `n_out` and `(0, 0, 2)` and `(0, 0, 3)` both mint `aux_0_0`. Those
  cannot co-occur — one output has one element count — so the claim asserted
  is the one the row actually makes.

  **And the vocabulary now forces the choice, which is what stops this
  happening a fifth time.** Every seam parameter carries a CLOSURE as well as
  an axis — SWEPT (every value in its range is driven; only `element` is one),
  INVARIANT (a named gate asserts the text is independent of it over a
  declared range) or DISCLOSED (a finite set of driven points, with the rest
  named in `SCOPE`) — plus a RESIDUE naming the part of its range that is not
  closed, whose phrase has to appear in `SCOPE`. DISCLOSED was EMPTY for a
  round and its emptiness was the tell: a vocabulary with nowhere to say
  *"listed, but not closed"* had nowhere to put the exponent, which is the most
  disclosed thing in the file. `exp_val`, `p` and `q` are in it now, and a
  DERIVED argument is required to carry its sources' residues — the `so` above,
  made mechanical. Four battery mutations pin the id range, a fifth test drives
  the whole `>= 3/4/5/6` family through `fidelity.gauge`, and the measured id
  reach is asserted to be a PROPER subset of the closed range so that a sweep
  which began recording itself would fail rather than print its own range as
  the reach. What is NOT closed is stated: an id is unbounded, `[0, 15]` is
  finite, and a wrongness conditioned past the top is disclosed in both
  not-reached lists. No source behaviour changes; the whole repair is in the
  instrument.

  **Carried as a disclosed follow-up, measured harmless today**: the integer
  branch's `n_out` RECOVERY can silently OVERSTATE. Two separate scalar `pow`
  eqns emitted consecutively present as elements 0 and 1 of one two-element
  eqn — not out of order, not gappy — so nothing fires and the recorder books
  an `n_out` of 2 where the truth is two eqns of 1. Checked against jaxpr
  ground truth on this tree: nothing overstated, nothing understated, because
  no gate holds two consecutive scalar `pow`s. The docstring stated that
  guarantee as an INVARIANT and it is a coincidence of the gate set; the honest
  fix keys the run on the eqn, and it is recorded in both the recorder's
  docstring and `docs/gauge-coverage.md` rather than taken here.

- **Two documented claims about this gauge were out of date in its own file.**
  `test_the_documented_coverage_figures_are_the_MEASURED_ones` read
  `reach.shapes`, `.integer` and `.rational` and asserted "the FULL PRODUCT"
  without the `assert not reach.inconsistent` guard its sibling has —
  measured: perturbing `smt.emit` so the recorder's LIFO mis-pairs left this
  test green, printing 63 of 63 off 54 inconsistent entries. The guard is
  there now — **and it is the WHOLE of the defence, which this entry first
  described as one omission among several figures.** Under that same
  perturbation the joint reach still counts its tuples and still equals the
  declared FULL PRODUCT, so the equality the page argues from does not notice a
  broken measurement at all; `assert not reach.inconsistent` is the only thing
  in either test that does. A product equality is evidence about the reach only
  once the reach is known to be a reading. And the bar-independence paragraph
  counted only the nine
  pre-existing tests elsewhere in the suite: measured with `pow` injected into
  `verdict.VERIFIED_BARRED_PRIMITIVES`, the BATTERY does read identically (32
  mutations, 0 survivors either way, every catch set unchanged, baseline
  passes every gate) but **10 demonstration assertions in the gauge file
  itself go red**, because their per-item verdict lines read `check().status`
  — which is the point of them. 21 red in total: 9 pre-existing elsewhere, 2
  detectors in `tests/test_bar_membership_policy.py`, 10 here. **That battery
  size was written `27` against a tree holding 28**: the run was taken one
  mutation early, the substance re-measured true at 28 — 0 survivors and
  identical catch sets both ways — and only the digit was stale, in the two
  files whose whole argument is that a documented count should be written by
  the tree and not by an author. Both copies are machine-checked now, by
  `test_the_DOCSTRING_and_CHANGELOG_battery_SIZE_is_the_one_that_RAN`.
  Separately, the
  shape probe's docstring said `pow` is monotone on a strictly positive box
  "so" the difference of two elements is symmetric about zero; the `so` does
  not carry. Every element of a declared array carries the SAME interval, so
  `r[n-1] - r[0]` is a SELF-subtraction and symmetric whether or not the row
  is monotone (demonstrated with a non-monotone `x*x` on a straddling box,
  equally undecidable). Strict positivity is load-bearing for DOMAIN
  ADMISSIBILITY instead — a negative integer exponent needs a base excluding
  0, a fractional exponent declines on a negative base.

- **The document test accepted wrong documents and rejected a right one.**
  `test_the_documented_coverage_figures_are_the_MEASURED_ones` required only
  that each gate the table NAMED be a real catcher, so it never required the
  naming to be complete: deleting one gate name from the
  `emit-integer-loses-the-reciprocal` row left the suite green while deleting
  exactly the fact the page's narrative argues from. The bare gate counts
  removed last round had been replaced by hedges (`and most others`, `and
  others`) with measured totals behind them from 4 of 21 to 18 of 21, all
  unchecked; the third column was checked in no respect at all and a
  falsified cell passed; and the single-covered comparison was ordered, so a
  correct alphabetised table was REJECTED. The catch column is now a parsed
  SET compared for equality — complete, prose-free, order-insensitive, with
  `ALONE` derived from the measurement rather than trusted — the
  single-covered rows are compared as a set, and every `A^B = C` anywhere in
  the section is decided exactly in `Fraction` arithmetic. Because a complete
  list runs to twenty gate names on one row, the column is GENERATED:
  `python tests/test_pow_row_gauge_jax.py --doc-blocks` prints both blocks
  from a live run.

- **A fixture docstring described an emission this tree had just outlawed.**
  `_rat_denominator_false_harness` said the battery's mutation emits
  `aux^5 = x` with a cap of `81^(1/5)`; the mutation adds TWO to the
  denominator (`q + 1` would be an odd `q`, which the same round made
  `_pow_rational_lines` REFUSE, so the item would have been caught by
  malformedness instead of by the denominator it exists to measure), and
  `q = 5` is not emittable at all. `docs/gauge-coverage.md` had it right,
  because the page is machine-checked and a docstring is not. Both are now:
  `test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`
  reads the emitted `(p, q)` off the LIVE mutation, decides
  `cap <= bound < truth` exactly (raising both sides to `q` so no irrational
  root is ever taken as a float), and fails unless the fixture docstring and
  the doc row both quote what it computed.

- **`exp` and `pow` under `semantics="ieee"` now require a DECLARED libm
  accuracy budget** (audit 0.2.0 **S9** and **S11**; S11 reaches the
  released **0.1.0** — see [SOUNDNESS.md](SOUNDNESS.md)). Under `ieee` a
  verdict is a claim about the float value the program computes, and
  stelling's bracket was built around CPython's `math.exp` — the libm of
  the machine running the analysis. The program runs whatever XLA
  compiled. Measured on jax 0.11.0 / jaxlib 0.11.0, CPU, x86_64,
  exhaustively over every `float32` argument whose result is normal and
  finite (2,237,668,967 of them), XLA's `exp` is out by up to **5.51
  float32 ulps** — not faithfully rounded at all, so no fixed widening is
  sound; in binary64 by up to **1.67 ulps** over 3,000,000 samples, which
  is what leaks past a ±1-ulp bracket. On the *same* backend `bfloat16`
  `exp` is exhaustively **correctly rounded** over every normal finite
  result, while `float16` misses correct rounding on 2 of its 63,487
  arguments (0.500028 ulps) — a factor of eleven between two formats of
  one op, so one number cannot be right for all four.

  Both transfers therefore **fail closed** and are re-enabled by a
  declaration:

      check(harness, vacuity_mode="inputs-only", semantics="ieee",
            libm_budget="xla-cpu-2026-08")

  `"xla-cpu-2026-08"` is a shipped, **named and dated** profile of
  per-`(op, format)` budgets; `stelling.propagate.LibmBudget` states your
  own. Both `check()` and `propagate()` take the keyword. The decline
  carries the measurement that justifies it and a line that **runs as
  written**. The budget widens the bracket by the declared ulps before
  the format rounding, and is stamped as **declared, not verified** —
  because a budget smaller than the backend's real error mints a VERIFIED
  stelling cannot catch. A budget of `0.5` ulps (correctly rounded) widens
  by nothing at all, which is `interval.sqrt`'s own argument generalised;
  `sqrt` is a correctly-rounded basic operation, carries no libm demotion,
  and needs no budget. `semantics="real"` is untouched and refuses the
  argument. Verdicts move **VERIFIED → UNKNOWN** and **REFUTED → UNKNOWN**
  on ieee-mode queries containing `exp` or `pow`; the coverage cost,
  measured at a point argument, is **12 float32 / 6 binary64 / 2 float16
  extra grid steps** of bracket width, and zero for `bfloat16`, the one
  format this backend's `exp` is exhaustively correctly rounded in.

- **Rational-`pow` exponent identity** (audit 0.2.0 S1; see
  [SOUNDNESS.md](SOUNDNESS.md)): the exponent was rationalised with
  `Fraction(e).limit_denominator(128)` and admitted on a *binary64*
  distance test, which measures exactly `0.0` for `0.1`. Verdicts move
  **VERIFIED → UNKNOWN** on every non-dyadic non-integer `pow` exponent;
  affects 0.2.0 development only.

- **No emitted term is a unary `(* t)`** (audit 0.2.0 S2): `q == 1` wrote
  an application SMT-LIB2's `Reals` theory does not define — cvc5 1.3.4
  segfaults on it, z3 reads it as the operand. Every repeated product now
  goes through one renderer (`smt._repeated_product`).

- **The rational-`pow` replay is exact** (audit 0.2.0 S3, M8): it computed
  `Fraction(float(base) ** exp)` while every REFUTED witness claimed
  "independent exact-rational replay". It now extracts exact integer
  `q`-th roots, or declines the witness through the existing
  "witness not independently replayable" channel. The public `check()` no
  longer raises `EmissionInfidelityError` on correct emissions, and the
  replay's `OverflowError` on large operands is gone with the float.

- **The fragment stamp follows the aux encoding** (audit 0.2.0 M9): a
  non-integer `pow` over a declaration-independent base was stamped
  `QF_LRA` while the emission wrote `(* aux aux)`, and both backends
  refused the script.

- **An IEEE divisor box that reaches zero divides to ⊤** (audit 0.2.0
  S10; see [SOUNDNESS.md](SOUNDNESS.md)): `ieee_div`/`ieee_div_fmt` read
  `[lo, 0]` as *"the divisor approaches 0 from below"* and returned a
  one-signed infinity. Under IEEE the divisor does not approach zero, it
  IS zero at that endpoint, and the sign of `x/0` comes from the ZERO's
  sign bit — which an interval endpoint cannot carry. `+0.0 == 0.0`, so
  `+0.0` is a value of `[lo, 0]` and the excluded `-inf` is a value of
  the program. **FALSE VERIFIED in all four formats**, a 0.2.0
  regression against `v0.1.0` (measured: `v0.1.0` returns `(-inf, inf)`
  where the pre-fix tree returned `(2.0, inf)`). Verdicts move
  **VERIFIED → UNKNOWN** wherever an ieee-mode division has a divisor box
  reaching zero. The boundary-aware branch also raised
  `IntervalError("NaN endpoint")` on `[-inf,-inf] / [-inf, 0]`; returning
  ⊤ before any endpoint arithmetic removes that too. Real-mode
  `boundary_div` is a sound kernel over `b ≠ 0` and is not wrong for this
  reason — ℝ has one zero and `a/0` is undefined there — but *reaching*
  it needs a premise the box does not carry, which is the next entry.

- **A real-mode divisor box that reaches zero declines unless a strict
  `assume` excludes the zero** (audit 0.2.0 B5-1; see
  [SOUNDNESS.md](SOUNDNESS.md)). **FALSE VERIFIED, real mode, made
  reachable by the M16 fix below.** With `mul` exact, `Σxᵢ²` floors at
  exactly `0`, so `Σxᵢ² − c` turned from a TRUE STRADDLE (which declines)
  into a ONE-SIDED BOUNDARY — and the one-sided arm was the only one of
  `div`'s four zero-containing shapes that did not decline. It called
  `boundary_div`, which drops `b = 0` from the image, and nothing in the
  verdict disclosed the drop. Measured: `x` declared `[0, 2]²`,
  `1/(jnp.sum(x*x) − 8.0)` boxed to `(-inf, -0.125]` and DISCHARGED
  `q <= -0.125`, while jax at `x = [2, 2]` — a point of the declared box
  — returns `+inf`. The three sibling shapes (`[0,0]`, a true straddle,
  a negative `sqrt` domain) all decline citing the same fact, that ℝ has
  no value there; this one minted a definite verdict from the rest of the
  box. Verdicts move **VERIFIED/REFUTED → UNKNOWN** wherever a real-mode
  division's divisor box reaches zero with no strict assume excluding it.
  See "Boundary-aware division" above for what now licenses the
  tightening and what carries the licence.

- **`boundary_div` answers `inf/inf` instead of raising** (audit 0.2.0
  B5-3). The claim recorded for the S10 fix — "returning ⊤ before any
  endpoint arithmetic removes the `NaN endpoint` raise too" — was true of
  `ieee_div` and false of the real-mode sibling, which was never touched:
  `_boundary_div_lo`/`_hi` fall to `_down(num/den)` on an infinite
  operand, and `inf/inf` is NaN. `boundary_div([inf,inf], [0,inf])`
  raised `IntervalError("NaN endpoint in interval arithmetic")` — caught
  by the dispatcher, so nothing crashed, but the domain's internal
  invariant string was printed as the user-facing reason `div` declined.
  `div`'s own `inf/inf` guard now runs first in both of `boundary_div`'s
  arms; 8 box pairs in the endpoint sweep raised before, 0 after.

- **`mul` is exact when its corner products are representable** (audit
  0.2.0 M16): it was the only arithmetic transfer with no exact-rational
  path, bumping every endpoint outward unconditionally. `[2,3]×[2,3]`
  boxed to `[3.9999999999999996, 9.000000000000002]` for an image that is
  exactly `[4, 9]`, and the exactly-zero corner of `[0,4]×[0,4]` bumped to
  `-5e-324` — below zero, which defeats `reduce_sum`'s nonnegative clamp.
  A sum of squares written `x*x` therefore became a true straddle and the
  division consuming it declined, while `x**2` and `jnp.square(x)`
  verified: one real property, three spellings, two verdicts — on exactly
  the `assume(x > 0)` sum-of-squares shape boundary-aware division was
  added for. Sound in both directions (the weak spelling only lost
  precision), so no verdict was wrong; verdicts move **UNKNOWN →
  VERIFIED/REFUTED** where the lost ulp was what prevented a decision.
  `mul` now takes the same `_exactable`/`Fraction` route `add` and `div`
  already had, confined the same way (an infinite endpoint keeps the bump,
  because `Fraction(inf)` raises and `0·±inf = 0` is an endpoint
  convention). The ieee `mul` kernels deliberately do NOT change: under
  ieee the value IS `fl(x*y)`, which the native corner products already
  compute exactly.

  **`dot_general` follows the same rule, because it now IS the same rule**
  (audit 0.2.0 B5-2). It carried an inlined COPY of `mul`'s four corners
  and M16 converted only the original, so `jnp.sum(x*x)` floored at
  exactly 0 while `jnp.dot(x, x)` floored at `-1e-323` — the M16 defect,
  one level up, in the second copy. Both call `interval._mul_corners` now.
  Measured over `x in [0,4]²`: the contraction returns `(0.0, 32.0)`,
  identical to the reduction, where it returned
  `(-1e-323, 32.00000000000001)`; a `[2,3]`-valued 2×2 matmul returns the
  exact `[8, 18]` where it returned `[7.999999999999999,
  18.000000000000004]`. Verdicts move **UNKNOWN → VERIFIED/REFUTED**,
  never the other way. Only the product corners changed: the accumulation
  already used `_add_lo`/`_add_hi`, and the association-order argument the
  contraction rests on is untouched by this and always was.

- **Relational assumes forwarded to solver**: when `assume(e1 < e2)`
  involves two variable operands (a constraint the interval domain cannot
  apply), the comparison is recorded and emitted as a positive axiom
  alongside the negated obligation. The solver sees the full constraint
  set.

- **SOUNDNESS FIX — a forwarded assume is now resolved by a scope-correct
  identity; it could previously be emitted about the wrong values.**
  See the SOUNDNESS.md log entry for the full account. In brief: a
  relational `assume` traced inside a `jit` / `custom_jvp` body was
  forwarded as its producing comparison equation, whose operand ids belong
  to that body, and `smt.emit` resolved them with a bare integer lookup
  against the slice's *renumbered* table. When the two id ranges met, the
  axiom was emitted about unrelated terms — measured as the CONVERSE of
  the user's own precondition, returning VERIFIED on an obligation false at
  every admitted point. Development-only; no released version is affected.

  What changed, user-visible:

  * `propagation.relational_assumes` now holds
    `stelling.propagate.RelationalAssume` records (the comparison equation
    plus the scope path its operand ids belong to), not bare
    `ir.JaxprEqn`s.
  * `ObligationSlice` carries `assumes` (translated into the slice's own id
    namespace) and `assumes_skipped` (one quoted reason per assume this
    obligation cannot state). The two partition the assumes the slicer was
    given, so *emitted versus requested* is derivable from the slice alone.
  * `stelling.smt.emit` no longer takes a `relational_assumes` parameter —
    the axioms come off the slice. `Script.relational_assumes_emitted` now
    counts assumes emitted **about the terms their operands denote**, and
    `Script.emitted_origins` names *which* ones, by their index in the
    propagation's forwarded tuple (`SliceAssume.origin`).
  * `slice_obligation` gained a `relational_assumes=` keyword;
    `slice_unknown_obligations` passes the propagation's.
  * **Once escalation dispatches, every assume the slice declines to state
    is disclosed** in the verdict notes, naming the assume's source line and
    the reason. Emission previously skipped silently in five places. The
    per-assume disclosure is produced *at dispatch*, so a run refused before
    dispatch — a constraining assume present, `semantics="ieee"`, no solver
    installed — or an obligation whose slice declines does not carry one; on
    those runs the propagator's own coarse `assume constraint DROPPED` note
    is still emitted, so no assume goes unmentioned, but it names no
    per-obligation reason.
  * **An assume inside a `jit` / `custom_jvp` body is now forwarded
    CORRECTLY rather than skipped**, which decides obligations that
    previously returned UNKNOWN. Measured on a **288-harness** generated
    sweep (`sweep_assume_scope.py`, the instrument's full product:
    4 carriers × 2 ndecls × 3 tails × 3 assume-sets × 2 exprs × 2 orders):
    **96 UNKNOWN→VERIFIED and 36 UNKNOWN→REFUTED**, no harness moving away
    from a decided verdict, and zero verdict changes on the **72**
    top-level-assume harnesses. Of the 96 new VERIFIEDs, **48 are vacuous**
    — an `unsat` assume set now reaches the solver from a `jit` body as it
    already did from top level; see the SOUNDNESS.md entry.
  * **A relational assume inside a `lax.cond` branch is no longer forwarded
    at all.** It is a branch-scoped precondition, not a fact about the
    query; the drop says so and keeps violations withheld.
  * `smt.emit` no longer raises `IndexError` on a shape-mismatched assume,
    and no longer emits a partial axiom over element 0 of an unrelated
    array (both arms of the same missing check).

- **SOUNDNESS FIX — a withheld violation is released only when every
  `assume` is accounted for, and that is now decided by a per-assume
  LEDGER rather than by two counts.** See the SOUNDNESS.md entry. The rule
  compared `len(propagation.relational_assumes)` against a script's emitted
  count, and that shape produced a false REFUTED twice: once because the
  denominator counted only the *relational* assumes while the flag gating
  the rule is set by any drop reason at all (audit 0.2.0 S6), and once
  because no longer forwarding branch-scoped assumes silently moved the
  denominator, so `1 == 1` released a witness whose branch precondition the
  solver had never been told. Development-only; no released version is
  affected.

  What changed, user-visible:

  * `Propagation.assume_ledger` — one
    `stelling.propagate.AssumeDisposition` per assumed conjunct the
    propagator classified, with kind `applied`, `no-op`, `forwarded` or
    `dropped`. It is written where the classification happens and is TOTAL
    over the assumes the walk sees, including inert mode.
  * `stelling.propagate.unaccounted_assumes(ledger, emitted_origins)` is
    the release test: a definite violation is released only when it returns
    empty. It joins on identity, counts nothing, and **whitelists** the
    accounted-for dispositions — a kind it has not been taught is
    unaccounted, so a drop reason added later refuses rather than defaults
    open.
  * The withholding note now NAMES the conjunct that caused it, with its
    disposition, reason and source line, instead of restating the rule.
  * `Propagation.assume_dropped` is unchanged and still gates the rule.

- **SOUNDNESS FIX — a discharge is no longer accepted when an EMPTY assumed
  region alone explains it.** See the SOUNDNESS.md entry. A relational
  `assume` is inert in the interval domain, so the empty-declared-set oracle
  (`UnsatisfiableAssumptionError`) never saw it — that oracle meets a box
  with a half-space. Since 0.2.0 the same assume is emitted to the solver as
  a positive axiom, and an unsatisfiable axiom set makes
  `boxes ∧ axioms ∧ ¬P` unsat for every `P`: every obligation discharged and
  the verdict was VERIFIED. Measured: `dt ∈ [5, 10]`, `dt_max ∈ [0, 1]`,
  `assume(dt < dt_max)`, `assert_(dt + dt_max <= 1.0)` — VERIFIED, and
  REFUTED with the assume deleted (audit 0.2.0 S7). The non-relational form
  of the identical mistake has always been refused; this closes the route
  around that refusal. Development-only; no released version is affected —
  at `v0.1.0` no assume reaches the solver at all.

  What changed, user-visible:

  * **`check()` and `check_inductive_step()` now raise
    `stelling.propagate.UnsatisfiableAssumptionError` when a forwarded
    relational assume set admits no point of the declared set** *and one
    obligation's script states the whole contradiction*. Same class, same
    closing sentence ("harness defect; nothing was verified"), as the
    non-relational refusal. `check()` already documents that class among the
    two it does not convert to a status. A contradiction spread across
    obligation cones — `assume(x<y); assume(y<z); assume(z<x)` with an
    assert depending on two of the three — cannot be refused, because no
    script ever holds more than one link of it; it is DISCLOSED instead (see
    two bullets down, and audit B3 in SOUNDNESS.md).
  * Before crediting an `unsat`, the backend that produced it is asked one
    more question — the same script with the negated obligation removed
    (`stelling.smt.emit(..., states_obligation=False)`) — and only on an
    obligation that discharged with at least one forwarded axiom on its
    script. **Zero extra solver calls on a query with no relational
    assume**, and none when the propagation's own non-emptiness certificate
    (`Propagation.region_inhabited`) already settled the question. Measured
    on the 288-harness sweep, where every harness carries a relational
    assume: 324 admitted-region invocations out of 1044 total, +11% wall.
  * An admitted-region check that does not settle the question does not
    withdraw the discharge; it stamps it. The obligation detail gains
    `[MAY BE VACUOUS: …]` and the stamp gains an `assumes:` line beginning
    `precondition satisfiability uncertified` — the may-be-vacuous line
    SOUNDNESS.md's constraining-assume policy already required and this path
    did not emit. **Two ways not to settle it, both stamped, each naming its
    mechanism on the obligation**: nobody answered, or the answer was `sat`
    over an axiom set that is not the whole query's (audit B3 — a model of a
    relaxation of your precondition is not a point of your precondition).
  * **A forwarded relational axiom now stamps its conditionality.** New
    `assumes:` line `forwarded relational assume(s) on obligation(s) …`,
    carrying the same `the verdict holds where the precondition holds`
    phrase an interval narrowing has always carried. It names the
    obligations it reaches, and the two readers of that phrase —
    `Verdict.render`'s conditional REFUTED wording and the inductive-step
    note — read the SCOPE (audit B3): a whole-query narrowing line qualifies
    every obligation, a forwarded line only the ones it names. Before that,
    a forwarded axiom on one obligation made an unrelated interval
    refutation render as "conditional … judged over the propagated superset
    of the precondition-narrowed set" and an unconditional inductive step
    render as "CONDITIONAL — NOT the inductive step".
  * The `vacuity checked …` line appends `WHAT THIS MEASUREMENT DOES NOT
    SAY: …` whenever the stamp carries any `precondition satisfiability
    uncertified` line: widening a bound can make an unsatisfiable
    precondition satisfiable again, so a re-check that fails to re-derive an
    obligation is not, there, evidence that the VERIFIED is substantive.
  * `stelling.solvers.Escalation` gained `region_uncertified` and
    `conditional_on_assumes` (obligation indices). Neither decides a
    verdict; both feed the stamp.
  * **`check_inductive_step`: an `assume` in the body no longer gets the
    unconditional note.** An assume is a precondition on the whole query, so
    a VERIFIED means "every state in the ASSUMED SUB-REGION stays in bounds
    after one step" — not the inductive step, because the successor need not
    re-enter that sub-region. The note now begins `inductive step
    CONDITIONAL — NOT the inductive step` and names the fix (put the
    restriction in `state_bounds`); the module docstring and
    `docs/inductive-step.md` say the same (audit 0.2.0 M5). Measured:
    `x -> 1.5x` on `[-1, 1]` under `|x| <= 0.5` is VERIFIED and iterating
    from the admitted `x = 0.4` leaves the invariant at step 3.
  * **`check_inductive_step`'s REFUTED note no longer names the wrong
    variable** when `body` declares its own `assert_` (audit 0.2.0 M4). The
    obligation-to-state-variable map was positional against an index that
    every body obligation shifts; the offset is now derived. A REFUTED whose
    violated obligations are all the body's own says so instead of blaming
    the invariant.

- **z3 tactic workaround for high-degree polynomials**: when a solver
  obligation contains a rational-pow auxiliary variable (`y^q = x^p`
  encoding), z3 uses a custom tactic chain (`simplify`, `solve-eqs`,
  `factor`, `purify-arith`, `tseitin-cnf`, `nlsat`) instead of the
  default `Solver()`. This restores the z3 cross-check on high-degree
  polynomials (measured: d=80 from 10s+ timeout to 0.35-0.6s). The tactic
  is activated automatically; cvc5 handles these natively.

- **Per-obligation withholding refinement**: when relational assumes are
  only partially emitted for a given obligation slice (some operands fall
  outside the backward cone), the solver ran over a wider domain than
  intended. A definite violation is un-withheld ONLY when every assume the
  user wrote is accounted for on **that** obligation's query — see the
  ledger entry above for the rule that decides it.

- **An assume that excludes nothing no longer withholds forever.** An
  assume whose entire content is a conjunct definitely TRUE over the boxes
  in force (`x ∈ [0,10]`, `assume(x >= -1. | x >= -2.)`) took the whole-drop
  path, which sets the withholding flag unconditionally, and the old release
  test could never fire on it. The ledger records that conjunct as `no-op`
  and the violation is released — the rule the mixed-conjunction path
  already applied to the same class of conjunct. Measured: UNKNOWN → REFUTED
  at `x = 6`, which is in the declared box, satisfies the assume, and
  falsifies the assert.

- **Emission guards resolve through inlined aliases**: guards (div, is_finite)
  now follow the slicer's alias chain to find propagated intervals for
  variables defined inside transparent calls (jit, custom_jvp_call).

- **An `assume` inside a `scan` or `while_loop` body is recorded instead of
  ignored** (audit 0.2.0 S13; see [SOUNDNESS.md](SOUNDNESS.md) — **this one
  reaches the released 0.1.0**). The propagation descends the transparent
  wrappers and `cond`; it does not enter a loop body, so a `stelling_assume`
  written in one was never classified — and, the part that made it a
  soundness defect rather than a precision limit, left no record that
  anything had been ignored. Nothing withheld, and a REFUTED came back
  naming a point the user's own precondition excludes. Measured on the
  `v0.1.0` tag and on `main`: `assume(x <= y)` inside a `lax.scan` body with
  `assert_(x - y <= 0.0)` returned REFUTED at `x = 0, y = -1`.

  `propagate._record_undescended_assumes` now reconciles the assume ledger
  against the STATIC set of assume equations the query contains, before
  anything reads the run's assume state, and writes a `dropped` disposition,
  a note naming the construct and the source line, and a stamped
  `precondition satisfiability uncertified` assumption. The same missing
  record reached three rules and all three now see it: the withholding rule
  (**REFUTED → UNKNOWN**, the violation withheld and the reason quoted), the
  admitted-region gate, and `REGION_NOT_ASKED` — which used to skip the
  region question outright whenever no relational axiom was forwarded, on a
  ground that is untrue for an assume that never narrowed anything.

  **The loop is still not descended**, deliberately: a loop body's `assume`
  is a per-iteration statement about a carry this analysis does not model.
  Write the precondition at the top level of the harness to have it
  honoured — see
  [docs/harness-api.md](docs/harness-api.md#an-assume-inside-a-scan-or-while_loop-body-is-not-descended).

  Verdicts move **REFUTED → UNKNOWN** on harnesses of that shape, and a
  discharge there gains a may-be-vacuous line. **This costs correct
  refutations, and the number is not zero.** Measured over a 240-harness
  loop-carrier corpus (`scan`/`while_loop`/`fori_loop`/nested `scan`/
  `scan`-in-`cond`/top control, comparison set `lt`/`le`/`gt`/`ge`, four
  asserts in both directions), scoring every moved row against the pre-fix
  run's own witness in exact `Fraction`: **200 rows move, and 80 of them —
  40 % — were correct refutations carrying correct witnesses**, spread
  evenly over all five loop carriers; 40 more were vacuous, 40 had no
  correct refutation at all, 40 had one with a different witness. The 40
  top-level control rows move 0. (A narrower 144-harness corpus scored 96
  moved rows all false; that partition is a property of ITS `lt`/`le`,
  one-assert-direction pairing — see
  [SOUNDNESS.md](SOUNDNESS.md).) Over the 288-harness
  `jit`/`cond`/`custom_jvp` corpus: 0 verdicts and 0 caveat states move —
  though the tightening is not gated on loops, and two non-loop shapes
  outside that corpus do gain a correct caveat (an assume inside a
  `lax.cond` branch, and `assume(jnp.all(...))` with no control flow).

- **ATTRIBUTION FOR THIS BATCH, PUBLISHED — with the census method, so the
  numbers can be re-derived rather than trusted** (audit 0.2.0 B6 audit 3,
  F5). The batch's commit message said *"every code change was reverted
  ALONE and the claiming tests go red"* and shipped **no table**, so the
  claim rested on the author. Re-deriving it moved two of the numbers.

  **CENSUS METHOD.** A raw hunk count is a property of the DIFF, not of the
  change: adjacent edits merge at wider context. So the width is stated.

  1. `git diff -U<W> 96ab47a d6b6d0b -- src/`, counting `@@` markers.
  2. Split that diff into one patch per hunk, each applicable alone with
     `git apply -R`.
  3. Classify each hunk **SEMANTIC** or **PROSE**: revert it alone, parse
     the file with `ast`, strip every docstring, compare `ast.dump`. A
     hunk whose lone revert leaves the docstring-stripped AST identical
     cannot change behaviour — nothing can red on it except a test that
     reads source line numbers.
  4. Run the whole suite once per SEMANTIC hunk, reverted alone.

  ```
  raw hunk census         -U0   -U3 (git's default)
    obligation.py          18    10
    interval.py             4     2
    ir.py                   5     2
    solvers.py              1     1
    TOTAL                  28    15

  at -U3:  SEMANTIC 12   PROSE 3
    PROSE: interval.h1 (the R4 comment), obligation.h8 (the preamble
    docstring), solvers.h1 (the Escalation docstring)
  ```

  So the batch is **15 hunks, 12 of them semantic** — not the 8 an earlier
  summary gave, and not the 10 a later one did.

  **TWO CONFOUNDS, AND BOTH ARE ELIMINABLE BY CONSTRUCTION rather than
  subtractable.** A revert experiment needs a clean tree, and two obvious
  ways to make one are not clean. `cp -a` preserves mtimes, so the copied
  `__pycache__` validates and its `co_filename` still names the ORIGINAL
  tree — `test_undescended_assume.py` compares a traced frame's filename
  against the test module's `__file__`, and reds in the UNREVERTED base.
  `git archive` carries no `.git`, so `test_reuse_pins.py`'s scratchpad
  floor skips ("not a git repository") and
  `test_skip_inventory.py::test_no_session_skip_is_undisclosed` reds on
  the undisclosed skip — again in the unreverted base. `git clone` has
  neither, and is the method.

  One confound genuinely does have to be subtracted:
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`
  reds on ANY line-count change in `src/stelling/obligation.py`, because
  `docs/supported-primitives.md` embeds source line numbers, and
  regenerating the page per revert would make the experiment circular.

  **RESULT** — full suite per revert, `JAX_ENABLE_X64=1`, jax 0.11.0,
  `pytest -q -p no:randomly`; NET = raw failures minus that row's base
  confounds:

  ```
  revert (hunks, -U3)                      raw  conf  NET  the tests that red
  R1  _declared_shape family h1+h2+h3+h4     8     1    7  the four below, plus
                                                            ..._DECLINES[bytes]
  R1a _binding_shape dispatch     h2         5     1    4  aval_lie: NO_shape_param
                                                            _binds_at_the_scalar;
                                                            slicer_closes_..._ON_ITS_OWN;
                                                            lie_no_longer_reaches_a_discharge;
                                                            BINDING_witness_alone_closes
  R1b the element-budget reader   h3         3     1    2  ..._DECLINES[str],
                                                            ..._DECLINES[not-iterable]
  R1c the slice-input reader      h4         1     1    0  NOTHING  <-- see below
  R2  TranscriptionError decline  h5         2     1    1  lie_is_refused_when_the_
                                                            descent_re_transcribes_it
  R3  the handler's _safely   h6+h10         3     1    2  net_around_the_association_
                                                            cannot_itself_raise;
                                                            ..._DECLINES[not-iterable] (*)
  R4  _frames list arm            h7         2     1    1  frames_is_total_on_a_list_
                                                            that_will_not_iterate
  R5  claimants read once         h9         2     1    1  the_claimants_count_is_read_ONCE
  R6  ir list recursion       ir.h1          1     0    1  load_walk_recurses_into_LIST_params
  R7  ir _validate_decl_eqn   ir.h2          1     0    1  declaration_check_reads_the_
                                                            EXTENTS_not_the_param_type
  R8  interval _indices    interval.h2       4     0    4  oracle_NORMALISES_its_dims (+3
                                                            0-d-array rows of ..._AND_NOTHING_ELSE)

  PROSE controls (the anti-vacuity half: a prose revert must red nothing)
  P1  interval R4 comment  interval.h1       0     0    0
  P2  preamble docstring          h8         1     1    0
  P3  Escalation docstring solvers.h1        0     0    0

  the unreverted base (a clone at d6b6d0b):  3557 passed, 10 skipped, 0 failed
  ```

  *`R7`'s test was RENAMED at audit 0.2.0 B6 audit 4 —
  `..._reads_the_EXTENTS_not_the_param_type` is now
  `test_the_declaration_check_compares_BOTH_holders_and_refuses_the_rest`,
  because the old name stated a rule the code stopped implementing at
  `30d4b04`. This row quotes the old name because it is driven against a
  clone at `d6b6d0b`, where the old name IS the name in the tree and the
  old rule IS the code (`ir.py:761`). That is the row
  `tests/test_array_emission.py` points a reader at, and it had no
  annotation until audit 0.2.0 B6 audit 5, F4 — the note had been attached
  to `OPT` instead, which is driven on a different tree.*

  **(*) R3's GROUP IS ONE HUNK SHORT, and the row says so rather than
  banking the extra red.** `_safely` has a THIRD call site, installed by
  `obligation.h1` inside `_declared_shape`, so reverting h6+h10 leaves it
  live: `..._DECLINES[not-iterable]` reds with a bare `NameError: name
  '_safely' is not defined` leaking into the decline reason, which
  measures an inconsistent tree and not the handler's degraded
  composition. R3's one genuine behavioural red is
  `test_the_net_around_the_association_cannot_itself_raise`, where the
  handler raises `RuntimeError` out of `getattr(o, "index", -1)`;
  reverting h6 alone reds that test and
  `test_slice_unknown_obligations_CANNOT_RAISE_from_its_OWN_body`. A
  revert group defined by "which hunks mention this symbol" is not the
  same as "which hunks the symbol needs", and this is what the difference
  costs.

  *This row said `obligation.h2` inside `_binding_shape` until audit 0.2.0
  B6 audit 4, F4. The substance was right and the LABEL was not, in a
  table whose whole purpose is re-derivability — so here is how to check
  it in one command, which is what the row should have carried in the
  first place:*

  ```
  git diff -U3 96ab47a d6b6d0b -- src/stelling/obligation.py \
    | awk '/^@@/{n++} {print n": "$0}' | grep '^[0-9]*: +' | grep _safely
  ```

  *prints the hunk number beside every added `_safely` line: **h1** (one
  call site, in `_declared_shape`'s "cannot be read" refusal — which is
  exactly the `..._DECLINES[not-iterable]` red above), **h6** (the
  definition), **h10** (four call sites, in the association net). `h2` is
  the `_binding_shape` dispatch and adds two lines, neither of them a
  `_safely`.*

  **And `obligation.h1`'s only OWN attributable red is
  `..._DECLINES[bytes]`** — it appears in the family row and in none of
  h2/h3/h4 individually. Since h1 cannot be reverted alone without
  breaking the tree, that single test is the whole behavioural evidence
  for the `_declared_shape` extraction, visible only through the group.
  Recorded because "seven tests red on the family" reads as seven tests
  red on the extraction, and it is one.

  Two hunks cannot be reverted alone at all and are reported as such
  rather than as measurements: `obligation.h1` removes `_declared_shape`
  while h2/h3/h4 still call it (**607 failed, 2949 passed** — an
  inconsistent tree, not a difference), and `obligation.h6` removes
  `_safely` while h1 and h10 still call it. Both are grouped above for
  that reason and neither is a row.

  **`obligation.h4` — the slice-input reader — REDS NOTHING, and is
  recorded as UNREACHABLE AS A GUARD rather than claimed.** The element
  budget calls `_declared_shape` over the same vids first, so no document
  can reach this call in a state the budget did not already decline: it is
  unreachable *as a difference*.

  **THE PRE-EMPTION IS NOW EXHIBITED, because `docs/norms.md` clause 1
  requires a document and this row had an argument** (audit 0.2.0 B6
  audit 5, F6). The document is a declaration whose `shape` param is
  `b"\x02\x02"`, installed past `__post_init__` — the hand-built route
  `SOUNDNESS.md` scopes in. Run through `slice_obligation`, with
  `_Slicer._declared_shape` instrumented to record its caller:

  ```
  outcome: DeclinedObligation
  reason : input declaration of variable 1 has a shape param b'\x02\x02'
           of type bytes: a declaration records its extents in a tuple or
           a list, ...
  _declared_shape call sites reached, in order:
      obligation.py:3360 in slice     <- the element budget
                                      <- obligation.py:3526, the
                                         slice-input reader, is never
                                         reached
  ```

  *(Both line numbers are the MERGED tree's. B6 published them as 3293 and
  3459, read on a tree without B7; B7's own `obligation.py` edits sit above
  both call sites and shifted each by 65; B11's two-line import addition
  shifted each by a further 2, to 3360 and 3526. The exhibit's own driver
  `test_the_R1c_disclosure_EXHIBITS_its_pre_emption` reads the budget's line
  off the source and requires this block to quote it, so it RED-ed on each
  of those merges until these digits were repinned — which is the exhibit
  working, and
  the reason it quotes a line rather than describing one.)*

  One refusal, produced by the earlier reader, on the same input, before
  `h4`'s site runs — which is the whole of what clause 1 asks. Driven by
  `tests/test_ir_screen.py::test_the_R1c_disclosure_EXHIBITS_its_pre_emption`
  so the exhibit is a run and not a transcript. `docs/norms.md` forbids
  exactly the move
  of asserting coverage by construction, and the batch's blanket "each
  change has a test that reds when reverted alone" was false here. It is
  KEPT and not deleted, because it is not a guard: it is a VALUE read, and
  the value it must produce is the one the budget counted and the one
  `_binding_shape` compared every reference against. An independent read
  there is UNSOUND-1 itself. That no test can tell the two apart today is
  a fact about today's readers agreeing, not a licence to let them
  diverge.

  Note also what P2 shows: a PROSE revert of `obligation.py` reds the
  supported-primitives page and nothing else, which is what makes that
  subtraction a line-count effect rather than a behavioural one.

  **AND AUDIT 3'S OWN FIXES, ATTRIBUTED THE SAME WAY** — by MUTATION,
  which is what `docs/norms.md` prescribes for a one-line guard and what a
  hunk revert degenerates into at this size. Each mutation asserts its own
  anchor before running, so a mutation that lands on nothing is an error
  rather than a green run; each is driven over the whole suite in its own
  clone. The base confound here is
  `test_sdist_contents.py::test_no_untracked_file_anywhere_would_ship`,
  which reds in every row including the control because `git apply` leaves
  a new test file untracked in a clone; with the file `git add`ed it is
  green.

  ```
  mutation                                  raw  conf  NET  the tests that red
  (control: no mutation)                      1     1    0  --
  F1 return the SECOND read in
     _declared_shape                          2     1    1  declared_shape_RETURNS_
                                                              the_extents_it_validated
  F1 compare RAW objects in the ir door,
     and re-read with int(d) for the
     byte-length product                      4     1    3  door_compares_the_extents_
                                                              it_VALIDATED_not_a_second_read;
                                                              byte_length_product_uses_the_
                                                              extents_the_guard_validated;
                                                              a_hostile___repr___cannot_raise_
                                                              out_of_the_public_constructor
  F2 narrow all four handlers back to
     `except TypeError`                      13     1   12  door_refuses_whatever___index__
                                                              _raises x3;
                                                              declaration_reader_... x3;
                                                              check_shape_refuses_... x3;
                                                              oracle_refuses_... x2;
                                                              declaration_refusal_cannot_be_
                                                              stopped_by_a_hostile___repr__
  F3 unguard every quoted repr             4     1    3  the three ..._hostile___repr__
                                                              tests, one per module
  F4 restore "THE ONE READER" and drop
     the named second reader                  2     1    1  declared_shape_is_NOT_the_
                                                              librarys_only_reader_of_an_
                                                              element_count
  F6 remove clause 4's convention from
     SOUNDNESS.md                             2     1    1  the_entrys_clause_4_states_
                                                              the_convention
  F7 remove the blindness paragraph           2     1    1  the_entry_names_the_screens_
                                                              blind_classes
  OPT restore the (str, bytes, bytearray)
     enumeration on both faces                4     1    3  declaration_check_reads_the_
                                                              EXTENTS_not_the_param_type;
                                                              ..._DECLINES[memoryview];
                                                              ..._DECLINES[array.array]
  ```

  F5 is the table above and is pinned by
  `test_ir_screen.py::test_the_batch_ships_an_attribution_table_that_adds_up`,
  driven three ways rather than by a whole-suite mutation: with no table at
  all (the state the finding reports), with one row's arithmetic broken,
  and with the `R1c` row claiming a red it does not have. All three red.

  *The `OPT` row's first test was renamed at audit 0.2.0 B6 audit 4 —
  `..._reads_the_EXTENTS_not_the_param_type` is now
  `test_the_declaration_check_compares_BOTH_holders_and_refuses_the_rest`,
  because the old name stated the rule the code had stopped implementing.*

  *AND THE ANNOTATION ABOVE NAMED THE WRONG COMMIT (audit 0.2.0 B6 audit
  5, F4). It said "the row measures `d6b6d0b`". It does not: `OPT` is
  driven on the audit-3 fix tree, **`30d4b04`**, where the old name is the
  one in the tree — and where the code is already the positive
  `isinstance(shape, (tuple, list))` test, so there is an enumeration to
  restore. `d6b6d0b` is what the mutation REPRODUCES, not what it runs on;
  at `d6b6d0b` the enumeration IS the code
  (`ir.py:761`, `obligation.py:2031`), so "restore the enumeration" would
  land on nothing there and this table's own anchor rule makes that an
  error rather than a green run. The row whose base really is a clone at
  `d6b6d0b` is `R7` in the revert table above, and it now carries its own
  annotation — which is the row `tests/test_array_emission.py` points a
  reader at. This is the same label-versus-substance slip the fixup two
  paragraphs above corrected for `obligation.h2` → `h1`.*

- **A SHAPE IS JUDGED BY THE SAME RULE WHEREVER IT APPEARS, AND THE ONE
  PLACE IT WAS NOT WAS REACHABLE FROM A JSON FILE** (audit 0.2.0 B6
  audit 8; reaches `main` at `198a2b5` and `dff95fc` identically).
  `ir._load_extents` guarded the per-extent `operator.index` and not
  `for d in shape`, and `ir._decode` read `tuple(obj["shape"])` in FRONT
  of it — a second reader of the document's shape standing before the one
  that owns the question. From pure JSON, through `ClosedJaxpr.from_dict`,
  with no Python object in the document:

  ```
  aval shape 2      raw TypeError: 'int' object is not iterable
  aval shape null   raw TypeError: 'NoneType' object is not iterable
  aval shape 1.5    raw TypeError: 'float' object is not iterable
  aval shape true   raw TypeError: 'bool' object is not iterable
  aval shape {}     ACCEPTED, and the aval records shape ()
  aval shape ""     ACCEPTED, and the aval records shape ()
  ```

  The last two are the ones that matter: they are not malformed extents a
  caller can be told about, they are a document silently given a
  different array than it wrote, after which the slicer, the propagator's
  element counts and the emission all model a scalar the document never
  described. The raw `TypeError`s are the catchability finding this batch
  already carries — `TranscriptionError` SUBCLASSES `TypeError`, so
  `except TranscriptionError` catches none of them — and unlike the three
  routes disclosed in [SOUNDNESS.md](SOUNDNESS.md) and driven by
  `tests/test_canonicalization_routes.py`, this one needs **no attacker
  Python at all**.

  **The rule applied is the one this module already states**, not a new
  one: `_SHAPE_PARAM_CONTAINERS`, which has judged a declaration's `shape`
  PARAM since audit 4 for a reason that is about the aval's shape word for
  word — *reading any other container as a shape models an array the
  document never described* (`tuple(b"34")` is `(51, 52)`; `tuple({})` is
  `()`). Both are asked by `_held_in_a_shape_param_container` now, so the
  two cannot be hardened apart. The per-extent rule stays deliberately
  wide: an extent may still be any object with a working `__index__`.
  Nothing that arrives through a document or a trace changes hands —
  `_decode` builds a `list` from JSON, `_jax_compat.any_array` normalises
  with `tuple(int(d) for d in shape)`, and a jax or numpy `.shape` is a
  `tuple`.

  **Measured before and after**, pure-JSON `from_dict` partition over the
  nine shape positions of a real document x 12 JSON values = **108 cells**
  (2026-08-17, python 3.12.3, `git clone --shared` trees):

  ```
                        ACCEPTED   TranscriptionError   raw TypeError
  main (198a2b5)            14             58                 36
  dff95fc                   14             58                 36
  ac2dcb1                   14             58                 36
  this commit                8            100                  0
  ```

  The 6 accepts that went away are `{}` and `""` at the three aval-shape
  positions that reach a live aval. The 8 that remain are `[]` and
  `[1, 2]` at those same three, plus `true` at the two shape-ELEMENT
  positions — `operator.index(True)` is `1`, so a `true` extent is stored
  as the `int` `1`, which is pre-existing and is not this finding.
  Pinned by `tests/test_shape_param_rule.py::test_an_AVAL_shape_is_held_
  to_the_same_container_rule_as_the_param` and its two siblings, all three
  red at `ac2dcb1`.

- **THE RECORD'S OWN CLAIMS, RE-READ AGAINST THE CODE** (audit 0.2.0 B6
  audit 8). No behaviour change; the entries above and below are corrected
  where measurement contradicted them, and every figure that moved is now
  either computed by a control or labelled as a dated off-tree
  measurement. In one place, so that none of them is a silent edit:

  * the per-face bypass counts (7/9 and 1/9, not "every face" and "the
    other eight") — above;
  * `9 x 3 x 3 = 81`, three bypass SPELLINGS and not "each of the two
    bypasses" — above;
  * "every row refused is refused" has an exception, and the `dtype: null`
    document is the one exception to "byte-identical `content_hash()`" —
    above;
  * MRO forgery is possible for 16 of the 21 stored types, not for `bool`
    alone, and buys nothing because every base the door dispatches
    `issubclass` against is one CPython refuses to forge — `ir.py`,
    computed by
    `tests/test_canonicalization_routes.py::test_MRO_FORGERY_is_possible_
    for_most_stored_types_and_buys_NOTHING`;
  * `_register_stored_type`'s frozen check establishes no-rebinding, not
    single-valuedness — `ir.py` and `interval.py`, pinned by
    `tests/test_canonicalization_routes.py::test_the_FROZEN_check_does_
    not_establish_SINGLE_VALUEDNESS`;
  * *"only an `object.__setattr__` past the frozen dataclass reaches it"*
    named one route where there are three — `SOUNDNESS.md`, and the
    enumeration is now held to the tests that drive it;
  * the `487 ns` scan cost is a distribution over scan position
    (170–489 ns across the 13 `ir` dataclasses in one process), and the
    "141-equation traced query" matched no query in this repo — the
    corpus's largest is `mime_fvm.py::h_f3` at 112 equations and 1204 `ir`
    objects.

  And `tests/test_ir_message_totality.py`'s MODULE docstring carried a
  second, unparsed copy of the figure table that the controls beside it
  had already outgrown. It carries no figures at all now, and
  `test_this_MODULE_docstring_states_no_figure_a_control_does_not_parse`
  is what keeps it that way — red against the docstring as it stood at
  `ac2dcb1`.

- **UNSOUND — THE DOOR'S OWN DISPATCH WAS BUILT FROM THE TWO MOST
  OVERRIDABLE TESTS IN PYTHON, SO IT COULD BE WALKED PAST; IT DECIDES BY
  IDENTITY NOW** (audit 0.2.0 B6 audit 7, **S14**; reaches `main` and the
  **released `v0.1.0`**). The entry below replaces every document-supplied
  value with an exact built-in "or refuses it". Its MECHANISM is sound —
  each read is the base type's own accessor, called exactly once, and the
  result is bound — and every sentence about it is about what happens
  once the door has decided to read. The decision itself was:

  ```
  if type(obj) in _CANONICAL_EXACT:      # a frozenset -> the METACLASS's
      return obj                         #   __hash__ / __eq__
  ...
  for base, read in _CANONICAL_READS:    # first entry: (bool, lambda v: v)
      if isinstance(obj, base):          # falls back to obj.__class__
          return read(obj)
  ```

  Three lines of metaclass answering as `float`, or a two-line
  `__class__` property returning `bool` into an arm whose read is the
  IDENTITY, and an arbitrary two-faced object is stored in an `ir` field
  untouched. **Each bypass sufficed alone**, and they are recorded
  separately because a repair that closed only their conjunction would
  leave both open. Per face, re-measured at `dff95fc` (2026-08-17, python
  3.12.3, `git clone --shared`) after this paragraph said "every one of
  the nine faces" and "the other eight": the metaclass alone stores the
  liar on **7 of the 9** faces — `tuple` and `list` refuse cleanly,
  because at `dff95fc` their exact arms are `t is tuple` / `t is list`
  (identity, which no metaclass moves), the frozenset a metaclass *can*
  answer held only the seven scalar faces, and the remaining
  `isinstance(obj, tuple)` arm reads the object's `__class__`, which this
  spelling does not override — and the
  `__class__` property alone stores it on **1 of the 9** (`bool`, the
  single face whose arm is an identity read), raw-crashes on **7**, and
  refuses cleanly on `NoneType`, which has no read arm to crash in. "The
  other eight raw-crashed" is **seven**. Driven together:

  ```
  query   x = any(shape=(2,), lo=1, hi=2)
          assert sum(x) <= C   and   assert C <= 79/20
  truth   max over [1,2] x [1,2] of (x0 + x1) = 4 > 79/20
  dff95fc obligation 0 = 'discharged'   obligation 1 = 'discharged'
  dee8bc2 obligation 0 = 'discharged'   obligation 1 = 'discharged'
  v0.1.0  obligation 0 = 'discharged'   obligation 1 = 'discharged'
  ```

  The document is SELF-CONTRADICTING — the two obligations together say
  the maximum is at most 3.95, and it is 4 — so refuting it needs no
  reference to the liar's identity, and `jnp.sum` at `x=[2,2]` returns
  `4.0`. Every object in it is built through a public `stelling.ir`
  dataclass; there is no `object.__setattr__` anywhere. The `main` and
  `v0.1.0` rows were re-measured here, on `git clone --shared` trees at
  `dee8bc2` and at the tag. The class extends past leaf values: an
  `ir.Var` or `ir.Aval` SUBCLASS with the same metaclass was CARRIED by
  the arm whose whole justification is that the object canonicalized its
  own fields, so a stored `Var`'s `id` answered `1, 99, 99, 99` across
  four reads — the exact hazard `Var.__post_init__`'s own comment names.

  **THE REPAIR IS THAT THE DOOR ASKS THE OBJECT NOTHING.** It reads
  `type(obj)` (the object header; no `__class__` property can move it),
  tests membership by IDENTITY through an `id()`-keyed index of the types
  it stores — `id()` has no override hook and its result is an exact
  `int` — and asks `issubclass(type(obj), base)`, which dispatches
  `type.__subclasscheck__` on the **BASE**, whose metaclass is exactly
  `type` (no `ABCMeta`, checked), so the derived class gets no say. And
  **no arm's read is the identity**: `bool` needs none, because CPython
  refuses it as a base class — a premise now measured rather than
  asserted. A metaclass can still forge the C-level MRO; CPython's own
  layout check refuses that for every base with its own layout, and the
  single case it permits is `bool` from a real `int` subclass, which
  shares `int`'s layout and is therefore not a lie about the payload —
  measured, and stored as the exact `int` it carries. Where a read is
  nonetheless handed an object without its type's payload it RAISES, and
  the door turns that into a `TranscriptionError` instead of letting a raw
  `TypeError` — which `TranscriptionError` SUBCLASSES, so `except
  TranscriptionError` does **not** catch it — out of a public constructor.
  Driving a liar of each of the nine faces, in each of the **three
  spellings** of the two bypasses — metaclass alone, `__class__` alone,
  both together — into a plain param value and into both declaration
  params (`9 x 3 x 3 = 81` combinations) gave **19 raw `TypeError`s at six
  distinct statements** on `dff95fc` and **none** on `main`, which has no
  door to raise them: that half is a regression of this batch and not a
  defect of the release. The arithmetic is written out because *"each of
  the two bypasses"* over 81 rows is a sentence that does not multiply —
  the mechanisms are two and the spellings driven are three (audit 0.2.0
  B6 audit 8). The message-totality sweep reaches none of the six, because
  every leaf it injects is a real SUBCLASS and none of these objects is a
  subclass of anything. All 81 are `TranscriptionError` now — re-measured
  2026-08-17: `dff95fc` 81 driven / 29 stored / 19 raw at 6 statements,
  this commit 81 / 0 / 0.

  **THE SAME MEMBERSHIP WAS THE HEADLINE TEST'S ORACLE**, which is the
  campaign's signature pattern arriving inside the test written to prove
  it closed. `test_every_value_a_document_stores_is_of_an_EXACT_type`
  asserted `type(v) not in allowed` — the door's own primitive — so:

  ```
  the TEST oracle: type(v) not in allowed      -> False  (reports NO defect)
  the DOOR:        type(v) in _CANONICAL_EXACT -> True   (store as is)
  ```

  The oracle asks identity now, the door asks `id()`, and the test's
  population gained a liar per face per bypass SPELLING — `9 x 3 = 27`
  rows, computed from the door's own declaration of what it stores.

  **THE `dtype` PARAM'S TYPE WAS UNCHECKED, a seventh member of the
  read-pair class inside the guard that closed the fourth.**
  `_validate_decl_eqn` gated the agreement check with
  `isinstance(raw_dtype, str)`, so a `dtype` param that is any other exact
  built-in did not fail the comparison, it SKIPPED it: measured,
  `b'float64'`, `0`, `64.0` and `('float64',)` were all ACCEPTED under a
  `float64` aval while a `str` `'int64'` was correctly refused. So *"two
  self-descriptions of one declared set"* held only when the param
  happened to be a `str`. The param is canonicalized and required to BE a
  `str` now, in its own sentence. No verdict moved on the three spellings
  driven — `propagate._ieee_any` re-derives most of the haze from the aval
  — which is a fact about how much of the model the param reaches, not a
  reason to leave a self-description unchecked; what it does reach is the
  subnormal band, and `str(b'float64')` is `"b'float64'"`, naming no ieee
  format at all.

  **`ir._register_stored_type` is gated.** It was module-level and checked
  nothing, so any code that can `import stelling.ir` could opt a type out
  of the door entirely. What registration delegates is single-valuedness,
  and the property that gives `interval.IntervalArray` it is that the
  class is a FROZEN dataclass — the same property `_CANONICAL_IR_TYPES` is
  computed with — so that is checked, along with "is a class" and "is not
  already a type this door stores". It is **still not a security
  boundary** and `ir.py` says so where the function is: code that can call
  it can equally rebind `_canonical`. The boundary the door defends is a
  DOCUMENT, and no document reaches this arm — `_decode` has no tag for a
  registered type and `_encode` refuses to encode one.
  *(THE `_encode` HALF OF THAT REASON IS FALSE, and the wording is left
  standing because a log that edits itself is not one. `_encode` refuses a
  registered value only in the arms where it RECURSES; at eighteen
  measured slots it writes the object straight through and `to_dict()`
  raises nothing. The conclusion — no document reaches this arm — holds on
  `_decode` alone. Corrected at the code in the B12 entry above, where it
  is measured.)*

  **ONE RULE, ONE READING.** `ir._SHAPE_PARAM_CONTAINERS` was shared by
  the load door and `obligation._Slicer._declared_shape` so the two faces
  could not hold different LISTS — and each face still spelled the test
  itself, both as `isinstance`. Hardening one alone would have re-opened
  by the reading the gap that was closed by the list, so the reading is
  `ir._held_in_a_shape_param_container` and both faces call it.

  **Compatibility, re-measured over all three routes.** A traced query's
  `content_hash()` is **byte-identical** on `dee8bc2`, `dff95fc` and this
  commit, over four documents (traced simple, traced rich, and each
  reloaded through `from_dict`); `to_dict()` round-trips stably on all
  three. The accept/refuse partition over the hand-built population is
  **unchanged from `dff95fc`** except in the ways named here, and the
  exceptions are named because the sentence that stood here — *"every row
  that was accepted is accepted and every row refused is refused"* — has
  three (audit 0.2.0 B6 audit 8; all re-measured 2026-08-17 on
  `git clone --shared` trees):

  * **narrowed:** a `stelling_any` `dtype` param that is not a `str`, when
    the outvar aval has a dtype.
  * **narrowed:** a value that lies about its type, previously stored and
    now refused.
  * **and one row went the other way.** A `bytes` SUBCLASS with a
    `__class__` property returning `str` was a raw `TypeError` at
    `dff95fc` (`descriptor '__str__' requires a 'str' object`) and is
    **accepted here and stored as an exact `bytes`** — which is right: it
    really is a `bytes` carrying `b'abc'`, the door reads
    `bytes.__getitem__` and stores the payload it finds. Accepted on
    `main` too, but there it stays the subclass. "Every row refused is
    refused" was the false half of the sentence, not the fix.

  **The `dtype: null` document is the one exception to "byte-identical
  `content_hash()`", and it is an exception because this commit produces
  no hash for it at all.** `_validate_decl_eqn`'s gate was
  `params.get("dtype") is not None`, so `.get` used `None` as its own
  sentinel and could not tell an ABSENT `dtype` param from one present and
  `null`; it is `"dtype" in params` now, which is a branch-SELECTION
  change and not only a type check. A hand-built document carrying
  `["dtype", null]` under a `float64` outvar aval was ACCEPTED at
  `dff95fc` and on `main` at `198a2b5` (both hashing to `64a0ce8d…`) and
  is a `TranscriptionError` here. The refusal is right —
  `propagate._ieee_any` would have read it as `str(None) == "None"`, a
  string naming no ieee format, so the declaration would have taken the
  no-float-format arm and got no subnormal band, the same silent misread
  `b'float64'` produces — but it is a compatibility change and the
  enumeration beside the code (`b'float64'`, `0`, `('float64',)`) did not
  contain it.

  **Cost, measured on this tree** (jax 0.11.0, python 3.12.3,
  `/home/nick/venvs/stelling-jax`, best of 9, three `git clone --shared`
  trees). The entry below was reported with microbenchmark figures that
  appear nowhere in the diff or its commit message; they are re-derived
  here rather than transcribed, so the absolute numbers are this run's:

  ```
                              dee8bc2   dff95fc   this commit
  250k ir.Var                  0.153 s   0.227 s   0.178 s
  250k ir.JaxprEqn             0.430 s   1.249 s   1.014 s
  traced query, 868 ir objects:
    from_dict(to_dict())       1.300 ms  2.043 ms  2.344 ms
    propagate()                2.196 ms  2.120 ms  2.291 ms
  ```

  250,000 `ir` objects is not a query; the traced query above builds 868,
  and `propagate()` on it is within noise of `main`.

  **THE QUERY THAT ROW NAMES DOES NOT EXIST — audit 0.2.0 B6 audit 8.**
  It was described in `ir.py` as *"a 141-equation traced query builds 868
  `ir` objects"*, and nothing in this repo traces to 141 equations or to
  868 objects. Swept 2026-08-17 over every zero-argument harness in
  `corpus/supply`: the largest is `mime_fvm.py::h_f3` at **112 equations
  and 1204 `ir` objects**, and the objects-per-equation ratio across the
  corpus runs **5.36 to 10.83** — a band the invented 6.16 sits inside,
  which is how it survived being read. The `ir.py` comment now names the
  measured query; the timing row above is left as the dated off-tree
  measurement it is, labelled by the object count it was actually taken
  on rather than by an equation count nobody can reproduce.

  **The membership microbenchmark is OFF-TREE and no control computes
  it**, because a microbenchmark asserted in a suite is a promise about
  someone else's machine. Re-measured 2026-08-17 (python 3.12.3,
  `/home/nick/venvs/stelling-jax`, 200k iterations per cell, three fresh
  processes, 20-type set):

  ```
  t in <frozenset>              23-39 ns    FORGEABLE (runs the metaclass)
  id(t) + one dict lookup       51-61 ns    unforgeable, constant
  any(t is k for k in ...)     168-508 ns   unforgeable, by SCAN POSITION
  ```

  The choice is between the two unforgeable spellings, and among those the
  index is the constant-time one; the `frozenset` is the cheapest and is
  the one that had to go, for S14 and not for cost. Two things this entry
  said about those figures do not survive re-measurement (audit 0.2.0 B6
  audit 8):

  * **`487 ns` is a DISTRIBUTION quoted as a constant.** It is the cost of
    the scan reaching that type, so it is linear in where the hit lands in
    the set's iteration order, and that order comes from the type objects'
    addresses: `ir.Jaxpr` sat at scan position 19 of 20 in one process
    (508 ns) and at position 6 in the next (297 ns). `487` was a
    last-position reading quoted as a typical one, inside an argument
    whose whole subject is that a scan cost depends on position.
  * **The exact-leaf / `ir`-dataclass split does not reproduce.** This
    entry gave them different `frozenset` costs (13.5 ns against
    33.1 ns); both are one hash lookup on a type object and measure the
    same here within noise. The absolute numbers above are roughly twice
    this entry's on every row, so they are this machine's on this date and
    the RATIOS are what the argument rests on.

  What bounds the real cost is not any of those per-call figures but the
  traced-query rows of the table above, which are a whole document through
  a whole pass. Those rows are themselves a dated off-tree measurement and
  are left as measured; the object count in their label was not
  re-derived here.

  The reproducer is `tests/test_aval_lie_both_faces.py::test_a_value_that_
  LIES_about_its_TYPE_can_no_longer_mint_a_FALSE_VERIFIED`, held beside
  the two documents below because all three are the same lie at three
  depths — a shape param's contents, a param key, and a value's type — and
  each of the last three rounds has closed one and been followed by the
  next.

  **ATTRIBUTION.** Same method as the table above — `git clone --shared`
  per row, the revert applied ALONE by exact-string replacement (a miss is
  a hard error, so a row that no longer applies fails loudly rather than
  measuring an unreverted tree), the whole suite run once per row,
  `JAX_ENABLE_X64=1`, jax 0.11.0, python 3.12.3,
  `pytest -q -p no:randomly`. No confound to subtract on any row:
  `docs/supported-primitives.md` cites line numbers in `obligation.py`,
  `propagate.py` and `coverage.py`, and the only row that touches one of
  those (`R7`) is line-count neutral. NET = raw.

  ```
  revert (this commit, applied alone)       raw  conf  NET  the tests that red
  R1  the door's membership test, back       42     0   42  every_value_a_document_
      to `type(obj) in <a frozenset>`                          stores_is_of_an_EXACT_type
                                                              (14 LIAR rows);
                                                              a_liar_is_refused_as_a_
                                                              TranscriptionError_and_never_
                                                              a_raw_TypeError (14);
                                                              an_ir_DATACLASS_subclass_is_
                                                              refused_even_with_a_LYING_
                                                              METACLASS (13);
                                                              a_value_that_LIES_about_its_
                                                              TYPE_can_no_longer_mint_a_
                                                              FALSE_VERIFIED
  R2  the door's arm test, back to            0     0    0  NOTHING alone — see R2+R4
      `isinstance(obj, base)`
  R3  the `(bool, identity)` read,             5     0    5  a_value_that_LIES_about_its_
      restored to the table                                   TYPE_can_no_longer_mint_a_
                                                              FALSE_VERIFIED;
                                                              the_doors_index_is_the_three_
                                                              sets_it_MERGES;
                                                              the_types_with_NO_read_are_
                                                              the_ones_that_cannot_be_
                                                              subclassed;
                                                              NO_read_in_the_table_is_the_
                                                              IDENTITY;
                                                              a_metaclass_that_FORGES_the_
                                                              mro_cannot_forge_a_payload
  R4  `_read_or_refuse`, back to the           0     0    0  NOTHING alone — see R2+R4
      bare `read(obj)`
  R2+R4  both of the above together           33     0   33  every_value_a_document_
                                                              stores_is_of_an_EXACT_type
                                                              (16);
                                                              a_liar_is_refused_as_a_
                                                              TranscriptionError_and_never_
                                                              a_raw_TypeError (16);
                                                              a_value_that_LIES_about_its_
                                                              TYPE_can_no_longer_mint_a_
                                                              FALSE_VERIFIED
  R5  the `dtype` param's TYPE                 6     0    6  a_value_that_LIES_about_its_
      constraint                                              TYPE_can_no_longer_mint_a_
                                                              FALSE_VERIFIED;
                                                              a_liar_is_refused_as_a_
                                                              TranscriptionError_and_never_
                                                              a_raw_TypeError (2 str rows);
                                                              the_dtype_param_must_BE_a_
                                                              str_and_not_merely_pass_an_
                                                              isinstance;
                                                              the_recorded_FIGURES_are_the_
                                                              ones_the_sweep_MEASURES;
                                                              the_QUOTE_SITE_COUNT_the_
                                                              record_quotes_is_the_union_
                                                              it_measures
  R6  the `_register_stored_type` gate         1     0    1  registering_a_stored_type_is_
                                                              gated_on_the_property_it_
                                                              delegates
  R7  the slicer's READING of the              1     0    1  the_measured_partition_IS_the_
      shape-param container rule                              documented_rule
  R8  the canonicalization test's              2     0    2  the_ORACLE_this_file_uses_is_
      ORACLE                                                  NOT_the_doors_own_primitive

  PROSE control (the anti-vacuity half: a prose revert must red nothing)
  P1  `ir._safe_repr`'s residue                0     0    0
      paragraph

  the unreverted base (a clone at this commit): 3733 passed, 10 skipped, 0 failed
  ```

  **R2 AND R4 EACH RED NOTHING ALONE, AND THE ROW SAYS SO RATHER THAN
  BANKING THE PAIR'S REDS TWICE.** They are two defences against the same
  object and each masks the other: with `issubclass` in place a liar never
  reaches a read, so removing `_read_or_refuse` changes nothing; with
  `_read_or_refuse` in place a liar that reaches a read is refused by the
  accessor raising, so restoring `isinstance` changes nothing either.
  Reverted together they red 33. Keeping both is deliberate — the
  `issubclass` half is what makes the refusal a TYPE decision rather than
  an accident of what the accessor happens to do, and the
  `_read_or_refuse` half is what makes the door TOTAL by construction
  instead of by an argument about CPython's layout check.

  **R8's ROW EXISTS BECAUSE THE ORACLE IS OTHERWISE UNATTRIBUTABLE.** With
  the door fixed, a liar document is refused before any oracle is
  consulted, so reverting the oracle alone would red nothing and the
  repair would rest on the author. It is measured directly instead —
  `test_the_ORACLE_this_file_uses_is_NOT_the_doors_own_primitive` hands
  the oracle a liar and requires it to say no — and the two spellings were
  driven against each other on the `R1` tree, where the liar document
  BUILDS:

  ```
  door reverted (R1); the stored ceiling is of type Liar_float
    OLD oracle  (type(v) in allowed)   -> reports NO DEFECT
    NEW oracle  (identity)             -> query.jaxpr.eqns[2].invars[1].val
  ```

- **UNSOUND — THE DOOR NOW STORES EVERY DOCUMENT-SUPPLIED VALUE AS AN
  EXACT BUILT-IN, OR REFUSES IT; CLOSING THE PAIRS ONE AT A TIME IS WHAT
  KEPT THIS OPEN** (audit 0.2.0 B6 audit 6). The entry below made the
  declaration door INSTALL the extents it validated, with

  ```
  (k, dims) if k == "shape" else (k, v)
  ```

  and `k` is document-supplied too. A `str` SUBCLASS answering that
  comparison True for `_validate_decl_eqn`'s two reads and **False** for
  the install's own third read let the door validate the param and report
  `dims` while the comprehension matched **nothing** — so the equation
  kept the raw lying object, and every later reader found the key again
  (True from the fourth call on) and read the lie. Same query, same
  oracle, same four read sites, same verdict:

  ```
  query    x = any(shape=<lies>, lo=1, hi=2);  assert sum(x) <= 3.9
  aval     x : f64[2]   — the shape the door validated the param at
  truth    max over [1,2] x [1,2] of (x0 + x1) = 4 > 39/10
  f729d70  obligation status = 'discharged'      <- VERIFIED, and false
  ```

  No `object.__setattr__` anywhere: every object was built through a
  public `stelling.ir` dataclass, exactly as the document below was. The
  sentence that stood over that install — *"exactly one `shape` key: the
  duplicate-key refusal above ran"* — was the defect: that refusal
  guarantees no key appears twice and says nothing about whether the
  comparison matches one. Here it matched zero.

  **FIVE MEMBERS IN FOUR ROUNDS.** Two more were measured beside the key
  and are closed by the same repair. The duplicate-key refusal asks
  `hash` (through `set`) and `eq` (through `list.count`) of the same keys,
  so two `str` subclasses with equal text and different `__hash__` were
  two set elements **and** two count hits — no duplicate seen — and a
  document carrying both `("update_jaxpr", None)` and
  `("update_jaxpr", <the add jaxpr>)` was ACCEPTED with `params_dict()`
  picking one **by hash placement**: precisely the `scatter-add`
  replace-vs-accumulate hazard that refusal's own comment says it exists
  to close. And the `dtype` param was compared with `==` at the door (a
  `str` subclass satisfies both the `isinstance` and `str.__eq__`) and
  consumed with `str()` by `propagate._ieee_any`, which picks the
  subnormal band from it — measured accepting a param whose `repr` is
  `'float64'` and whose `str()` is `'int64'`, the arm taken for a
  declaration with no float format at all. That one was **not** driven to
  a moved verdict: every direction tried was caught by the comparison
  transfers, which haze their operands from the value's own aval as well.
  A fifth family — `axes`, `new_sizes`, `slice_sizes`,
  `dimension_numbers` — had no door at all; driven, a two-faced `axes` was
  read twice with two different answers and `_one_shape_per_value` caught
  the divergence and DECLINED, so that member cost liveness rather than
  soundness, and only because an invariant elsewhere was standing there.

  **THE REPAIR IS NOT A SIXTH PAIR.** Python's protocols are overridable,
  so any document-supplied value participating in a decision can answer
  `==`, `hash`, `iter`, `str`, `index`, `len` or `getitem` differently on
  two reads. `ir` now carries a **canonicalization door**: at
  construction, every field of every `stelling.ir` dataclass — read from
  `dataclasses.fields`, not listed — is replaced by an EXACT instance of a
  type the module is closed over. A subclass is read ONCE through its base
  type's own accessor (`str.__str__`, `int.__index__`, `float.__float__`,
  `bytes.__getitem__`, `tuple.__getitem__`), which reaches the payload the
  instance carries and cannot be redirected by an override; a `list` is
  stored as a `tuple` (an exact `list` is mutable, so it is still two
  answers waiting to happen, and `_encode` has no `list` arm at all); and
  a type with no exact form to store is REFUSED naming its type. After
  that no later read *can* differ, because there is nothing left to
  override — and that is a property of the STORED OBJECT rather than of
  any reader, so it holds for readers nobody has written yet, which is
  what the per-member repairs could not do. **That last sentence was
  false as shipped, because the door's own DISPATCH was overridable and
  the value never reached the mechanism it describes; see the audit-7
  entry above, which is where it becomes true.**

  **A SUBCLASS IS READ AND NOT REFUSED, AND THE TRACE PATH IS WHY**:
  `np.float64` IS a `float` subclass and `_jax_compat.Transcriber.param`
  returns it unchanged from its `isinstance(v, (int, float, complex))`
  arm; `np.str_` IS a `str` subclass leaving the
  `isinstance(v, (bool, str))` arm the same way. Both measured. A door
  that refused every subclass — the auditor's stated minimum for the key —
  would refuse traced queries.

  **WHERE THE BOUNDARY IS between a value that decides and one that is
  merely carried: there is not one, and looking for it is how this stayed
  open.** `dtype` was a carried param until `_ieee_any` began selecting a
  band from it; `update_jaxpr`'s mere PRESENCE became semantic when the
  scatter-add row learned to read it. The rule is therefore uniform over
  every field. Two keep their own, stronger rule, and the generic door is
  held back from them: aval and array extents (`_load_extents` reads any
  object with a working `__index__` once and installs a plain `int`,
  which is *wider* than a type test and already single-valued), and a
  declaration's `shape` param, so that it is still refused by the
  container rule with the sentence that names it.

  **WHAT IT DOES NOT DO.** It makes a param SINGLE-VALUED, not CORRECT.
  `axes` still has no schema — that would be per-primitive shape
  inference, which `ir.py` scopes out in writing — so the transfer and the
  emission now read the same extents, and whether those are the right
  extents is a claim nothing here makes. Carried as a disclosed follow-up.

  **Compatibility, measured over all three routes.** `from_dict` keys come
  from JSON object entries and its values from `_decode`; trace keys come
  from a jax `dict` and its values from `Transcriber.param`. Both are
  exact already, so both are unchanged: the canonical document round-trips
  to the same `to_dict` and the same `content_hash`, and a traced query
  carries no value of an inexact type. Three narrowings, all of hand-built
  IR: a param key that is not a `str`, a `params` entry that is not a
  `(key, value)` pair, and a param value of a type the module cannot store
  (`memoryview`, `array.array`, `range`, `dict`, `set`, `bytearray`, a
  bare `object`, or a SUBCLASS of an `ir` dataclass — a dataclass subclass
  can make any field a property, so there is no single read of it to
  take). One widening: a `list` param under any key now stores as a
  `tuple` and therefore serializes, where before it produced IR
  `content_hash()` could not encode. `interval.IntervalArray` is declared
  to `ir` from `interval.py` rather than refused, because
  `ClosedJaxpr.consts` may hold one in place of a value and `ir` may not
  import it.

  The reproducer is `tests/test_aval_lie_both_faces.py::test_a_lying_
  param_KEY_can_no_longer_mint_a_FALSE_VERIFIED`, held beside the document
  below because they are the same lie at two depths and closing one has
  reopened the other twice. The class is
  `tests/test_ir_canonicalization.py`.

- **UNSOUND — A GUARD MUST INSTALL THE VALUE IT VALIDATED, NOT MERELY
  RETURN IT** (audit 0.2.0 B6 audit 5, F1). A `tuple` SUBCLASS whose
  `__iter__` yields `(2,)` on the first read and `(1,)` on every read
  after it. No `object.__setattr__` and no smuggling: every object is
  built through a public `stelling.ir` dataclass, and
  `JaxprEqn.__post_init__` **accepts** the document, because the door
  reads the param once — audit 3's F1 repair, and correct — and the read
  it gets agrees with the outvar aval.

  ```
  query    x = any(shape=(2,), lo=1, hi=2);  assert sum(x) <= 3.9
  truth    max over [1,2] x [1,2] of (x0 + x1) = 4 > 39/10
  321209d  obligation status = 'discharged'      <- VERIFIED, and false
  ```

  Four readers reached that param and the door was the only one that
  validated it: `ir.py`'s door, `ir._encode`, `propagate`'s `stelling_any`
  transfer (which built the ONE-element box the discharge rests on), and
  `coverage.sub_jaxprs`. `content_hash()` succeeded and hashed `(1,)`
  while the door had validated `(2,)`. **`main` (`dee8bc2`) and `96ab47a`
  refuse this document by ACCIDENT** — there the door read the param
  TWICE and the second read caught the lie; `d6b6d0b`, `30d4b04` and
  `321209d` all discharge it. Read-once-and-bind removed the accidental
  catch and nothing replaced it.

  **The repair is this batch's own theme one level up.** A guard that
  hands its value back to its own caller has fixed one read; a guard that
  INSTALLS its value into the object has fixed every read there will ever
  be. `JaxprEqn.__post_init__` now writes the extents
  `ir._validate_decl_eqn` returned back into `params` — the same
  `object.__setattr__` idiom that method already uses to sort them — and
  `Aval.__post_init__` and `Array.__post_init__` do the same for their own
  `shape`. Every later reader is handed a plain `tuple` of plain `int`.

  **Why not a shared reader.** Routing every reader through one function —
  the `_size`/`_extents` repair one module over — makes every read use one
  PROTOCOL; it does not make two reads return one VALUE, which is exactly
  what this document exploits. And two of the four readers could not be
  routed at all: `ir._encode` is generic over tuples and cannot know which
  one is a shape, and `coverage.sub_jaxprs` walks params looking for
  sub-jaxprs without ever asking what a param means. Normalisation covers
  both without touching either file, and it is why `propagate`'s transfer
  — held to no container rule of its own — now needs none: for every
  equation built through `ir.JaxprEqn`, which is every equation
  `propagate` is given, there is only one value left for it to read. (The
  transfer FUNCTION still has no rule, and
  `test_the_TRANSFER_face_is_NOT_held_to_the_container_rule` still drives
  it with a raw params dict to say so.)

  **Compatibility, and what the argument for it is worth.** The fix adds
  no refusal: every `_load_check` and every condition it tests is
  unchanged, and the only new statements STORE values the guards had
  already computed. For any document whose extents and containers answer
  CONSISTENTLY — which is every document a trace, a `from_dict` or an
  ordinary hand-build produces — the installed value equals every read and
  behaviour is identical by construction. For a document that answers
  inconsistently, later checks now see the FIRST read, the one the guard
  passed, instead of a second one; that IS the fix, and it is a change in
  what those checks judge, so it is measured rather than argued away — the
  suite in both environments, the 686-document exact-`Fraction` fuzz (686
  built, 0 false discharges, 0 raw escapes), and the doc-example corpus.
  The canonical `shape=(2,)` declaration's `content_hash()` is
  byte-identical on both trees.

  *(That sentence used to pin a 16-character hash LITERAL, and the
  literal matched neither of the two documents it could have named —
  audit 0.2.0 B6 audit 6, F5. Both were built and hashed on both trees:
  the full `sum(x) <= 3.9` query and the declaration-only document each
  hash identically across the two trees, so the PROPERTY the sentence
  claims verifies and only the number was wrong. The number is dropped
  rather than corrected. A hash literal in prose is a figure no reader
  can check and no test holds — it goes stale the first time the
  serialization changes for an unrelated reason, and this one was already
  wrong when it shipped. What replaces it is the property, computed:
  `tests/test_ir_canonicalization.py::test_every_canonicalization_the_
  record_names_is_DEMONSTRATED` builds both spellings of the canonical
  declaration and asserts their hashes agree, and
  `test_the_record_does_not_pin_a_hash_LITERAL_in_prose` reds if one is
  written back into this file. A hash literal inside an EXECUTED doc
  example is a different thing and is fine: `tests/test_doc_examples.py`
  recomputes those.)*

  Two documents are now
  ACCEPTED further than before, both toward canonicalisation: a `list`
  `shape` param — a form `ir._SHAPE_PARAM_CONTAINERS` explicitly blesses —
  used to raise `TypeError: stelling.ir cannot encode list` out of
  `content_hash()`/`to_dict()` and now hashes identically to the `tuple`
  spelling of the same declaration; and an `Aval` whose extents are numpy
  integers now encodes (it used to raise *"Object of type int64 is not
  JSON serializable"*). One value changes: an extent written as `True`
  stores and encodes as `1`, so a document with a boolean extent hashes
  differently. No jax trace produces one, and `operator.index(True)` is 1,
  so the two documents denote the same shape.

  **A THIRD LEAK SITE GOES WITH IT** (audit 0.2.0 B6 audit 5, F5).
  `obligation._size`'s residue paragraph named two callers that can let a
  `_Decline` out unnetted; there are three. `_index_box`
  (`range(_size(shape))`, reached from `_pair_elementwise` /
  `_route_structural`, which `smt.emit` drives AFTER `slice_obligation`
  has returned, and `smt.py` nets no `_Decline` at all) is the third, and
  it is new at `321209d` — at `30d4b04` `_size` was a raw product, so the
  same shape left the helper as a bare `TypeError`. Swept over the read at
  which a hostile extent starts refusing, on one declaration query:

  ```
  321209d   k=32, k=35  escalation attempted; internal error: _Decline
            k=37, k=40  EmissionInfidelityError / ReplayError
  with F1   no decline at any k
  ```

  because `Aval.__post_init__` now freezes the extent at the read it
  validated. The paragraph's ARGUMENT is corrected too: *"the shapes come
  from an `ObligationSlice` whose extents ... already normalised"* is true
  of `SliceInput.shape` and was never true of `sl.root.aval.shape` or
  `_shape_of(eqn.outvars[0])`, which were fresh reads of a raw `ir.Aval`
  field. What makes those safe is the constructor, and it is named as
  such.

  The blocking document is `tests/test_aval_lie_both_faces.py::
  test_a_lying_shape_param_can_no_longer_mint_a_FALSE_VERIFIED` (it now
  reaches REFUTED with a two-element witness the exact-rational replay
  confirms), and the mechanism is swept over both accepted container types
  and five flip points in `..._test_the_DOOR_INSTALLS_the_shape_param_it_
  VALIDATED` and over the computed container population in
  `tests/test_shape_param_rule.py::test_the_door_INSTALLS_what_it_
  VALIDATED_so_a_second_read_cannot_differ`.

  **Two sentences this batch shipped are falsified by that document and
  are corrected with it.** `tests/test_shape_param_rule.py` said *"a
  document whose param and aval disagree is refused at the `ir` door on
  every deserialized route"* — this one's param and aval disagree on every
  read after the first and the door accepted it. And
  `obligation._shape_problem` / `_Slicer._declared_shape` both said the
  containment was `ClosedJaxpr.content_hash()`, *"which cannot encode such
  a param"*. It can: `ir._encode` iterates a shape param ONCE and encodes
  what that read returned. The `list` the claim was measured on raised for
  an unrelated reason — `_encode` has no `list` arm at all, so an honest
  `shape=[4]` raises the identical `TypeError` — and the `tuple` half of
  the rule was never contained by anything. Measured at `321209d`, a
  `tuple` subclass whose `__iter__` answered `(4,)` once and `()`
  afterwards hashed cleanly at every flip point.

- **A CLAIM ABOUT CONTAINER TYPES IS COMPUTABLE, SO IT IS COMPUTED** (audit
  0.2.0 B6 audit 4, F1). Four audits of this batch have found the same
  defect — a principle stated in prose beside code that does something
  else — and the last three found it in the fix meant to close the previous
  one. The fifth instance was `ir._validate_decl_eqn`'s own docstring: it
  described *"any sequence of extents is now compared, and a `shape` param
  that is not a sequence of extents at all is REFUSED"* for two commits
  after the code had become `isinstance(shape, (tuple, list))` — a check on
  the param's PYTHON TYPE, under which `range`, `array.array`,
  `memoryview`, a numpy array and every custom iterable ARE sequences of
  extents and are all refused. And the stale sentence reached users:
  `np.array([4])` was refused with *"shape param array([4]) is not a
  sequence of extents"*, which is not why it was refused.

  So the sentence is not the deliverable. **The rule now lives in one
  object**, `ir._SHAPE_PARAM_CONTAINERS`; `ir._validate_decl_eqn` and
  `obligation._Slicer._declared_shape` both ask THAT rather than each
  keeping a copy of the answer, both refusal messages are composed from
  `ir._SHAPE_PARAM_RULE` (derived from it), and both docstrings name it
  instead of restating it. The refusal now names the type it refused and
  splits the two failures the single sentence had merged: the wrong
  CONTAINER TYPE, and the right type whose ITERATION RAISES.

  **And the partition is measured, not asserted.**
  `tests/test_shape_param_rule.py` builds a population of container types
  by SCANNING `builtins`, `collections`, `collections.abc`, `array`,
  `queue` and `types` with a battery of generic constructor arguments,
  derives a subclass of every type that allows one, derives a
  refusing-`__iter__` subclass of every ACCEPTED type, and adds numpy's
  containers by hand. Measured on this tree: **51 objects, 8 accepted, 43
  refused, and the two faces partition identically** — so the door and the
  emission cannot come to hold different rules, in either direction, and
  the soundness direction (emission ⊆ door) is the one that would be
  UNSOUND-1 again.

  **What that pin does not cover is stated in the test rather than
  discovered later.** It does not prove the population complete — a
  container type in a namespace the scan does not visit is measured only
  because someone added it, which is the class narrowed and not
  eliminated. It does not stop the rule being WIDENED: everything derives
  from one tuple deliberately, so a change moves the rule, the messages
  and the expected partition together, and what reds on that is one
  deliberate line
  (`test_the_rule_itself_is_pinned_to_tuple_and_list`). And it says
  nothing about the THIRD reader — `propagate`'s `stelling_any` transfer
  reads the same param with a bare `tuple(shape)` and is held to no
  container rule at all. That asymmetry is now DRIVEN and recorded rather
  than described (`test_the_TRANSFER_face_is_NOT_held_to_the_container_
  rule`); it is the standing disclosure this transfer already carried.

- **"EVERY QUOTE HERE IS GUARDED" WAS FALSE 44 LINES BELOW ITSELF, and
  the four sites were six short** (audit 0.2.0 B6 audit 4, F2). A
  `_load_check` message is an ARGUMENT, so it is composed on the passing
  path too; `ir`'s validation runs inside the public dataclasses'
  `__post_init__`, so a message that raises is a raw exception out of
  `ir.Array(...)`, `ir.Literal(...)`, `ir.JaxprEqn(...)` or
  `ir.ClosedJaxpr(...)` — the class the validation exists to close.
  Driving the class rather than reading it, over one canonical
  well-formed document: **10 DISTINCT quote sites**, of which the audit
  had named four. (A quote site here is one INTERPOLATION of an object
  into a message. The figures further down count MESSAGE EXPRESSIONS,
  which is a different unit — the duplicate-key refusal is two quotes in
  one message — and the two agree at ten by different routes rather than
  one being derived from the other. Said, because writing an identity
  across two units is this entry's own defect one step subtler.) Two of
  the six the audit had not named fire on the PASSING path, on documents
  with nothing wrong with them:

  ```
  ir.Array(dtype=<str subclass whose __repr__ raises>, ...)      PASSING path
  ir.Literal(val=<int subclass whose __repr__ raises>, aval=())  PASSING path
  ```

  Three further sites the canonical document MASKS rather than clears —
  `_validate_jaxpr` composes its own `where` string before
  `_validate_required_params` runs — are driven one row each:
  `ir.JaxprEqn`'s duplicate-key refusal, its `{dups}` list, and
  `_validate_required_params`' primitive quote. And the
  `NamedTupleParam` field name in a `where` path is one the sweep found
  and no reading had. All of them now go through `ir._safe_repr` /
  `_safe_type_name` / `_safe_str`; four more quotes outside the
  `_load_check` pass (`_encode`/`_decode`/`from_dict` type names and
  `_decode`'s unknown-tag) are guarded for uniformity rather than because
  a document was built for them. **All are pre-existing and none is a
  regression** — `d6b6d0b` and `30d4b04` escape on these too.

  The sweep is `tests/test_ir_message_totality.py`, and it is a property
  rather than a list: one canonical well-formed `ClosedJaxpr`, its leaves
  found by walking `dataclasses.fields()` (so a field added to any IR
  dataclass joins the sweep with no edit), each leaf replaced in turn by a
  SUBCLASS OF ITS OWN TYPE that refuses `__repr__`, `__str__` and
  `__format__`, the document rebuilt from the root so every
  `__post_init__` re-runs, and `ir._validate_loaded` driven over the
  result.

  **THE FIGURES, AND WHICH MEASUREMENT EACH BELONGS TO** (audit 0.2.0 B6
  audit 5, F2). Three sentences in this entry and its test carried four
  numbers for two measurements — "26" and "27" escapes, "9" and "eight"
  quote sites — because each was typed where it was needed. The sweep's
  own unit is the MESSAGE EXPRESSION, which is neither of the other two: a
  `_load_check` message spans several source lines and can interpolate on
  more than one, so a per-LINE count is larger, and a per-QUOTE count (the
  ten above) is larger again. Measured on `jax 0.11.0` / `python 3.12.3`,
  x64 on:

  ```
  at dff95fc, as shipped             95 swept / 0 escapes / 20 skipped
  guards neutered                     1 escape  /  1 line  / 1 message
  guards neutered, and the door's
    LEAF READS neutered too          26 escapes /  8 lines / 8 messages
  30d4b04, guards absent             28 escapes / 10 lines / 8 messages
  message-expression union           10 = those 8 + the 2 the canonical
                                          doc masks
  ```

  That last 10 is the same number as the ten quotes above and is not the
  same quantity; it is stated so a reader can recompute it, not so the two
  can be treated as one.

  **THOSE ARE `dff95fc`'S FIGURES AND THE TREE HAS MOVED** — audit 0.2.0
  B6 audit 7 gave `_validate_decl_eqn`'s `dtype` param a refusal for its
  TYPE, which is a message expression the sweep reaches, so at the end of
  B6 the
  door-removed row was `27 escapes / 9 lines / 9 messages` and the union
  was `11 = those 9 + the 2`. **AND IT MOVED AGAIN AT B12**, which is why
  those two are now written in the past tense: the field-annotation rule
  is a THIRD defence standing in front of six of these sites, the
  door-removed row reads `87 / 5 / 5` with it shipped and `29 / 10 / 10`
  with it neutered as well, and the union — taken over CONFIGURATIONS now,
  precisely so a defence in front of a guard cannot make the guard look
  unnecessary — is `13 = 11 + the 2`. See the B12 block at the head of
  this section. The shipped row and the guards-neutered row are
  unchanged at `95/0/20` and `1/1/1`. Both are COMPUTED by the two tests
  named below, which now also read the table in their own docstring back
  out and compare it — a table beside a dict was an honour-system copy of
  it, and at `dff95fc` the two had already parted (that docstring stated
  `27/9/8`, which is no control this file runs).

  **THE POSITIVE CONTROL NOW REMOVES TWO DEFENCES, NOT ONE** (audit 0.2.0
  B6 audit 6). Every hostile leaf this sweep injects is a SUBCLASS of a
  stored type, and the canonicalization door added in audit 6 replaces one
  with an exact twin before any message quotes it — so 25 of the 26
  ESCAPES do not happen, across 7 of the 8 message expressions, and they
  are not GUARDED but unreachable *for the leaf this sweep injects*.
  **That qualifier was missing and its absence was read as a claim about
  `_safe_repr` being load-bearing at ONE site** (audit 0.2.0 B6 audit 7).
  It is not: a leaf that BYPASSES the door is stored unchanged — which is
  exactly what neutering the leaf reads simulates — so it reaches every
  one of those sites, and there the guards are the only thing between it
  and a raw crash out of a public constructor. `_safe_repr` is
  load-bearing at all of them. A control that only neutered
  `_safe_repr` measured 1 escape and would have gone on passing with every
  guarded read in the module deleted. The guard figures are therefore
  measured with the door's LEAF READS neutered as well — not the whole
  door, which also collapses `list` to `tuple` and canonicalizes the
  declaration's own `dtype` — and the door's effect on this sweep is the
  difference between the two rows. The one escape that survives it is the
  hostile `int` inside a declaration's `shape` PARAM, which
  `_validate_decl_eqn` owns and reads as handed in. The guards-neutered
  figures also moved by one escape and one LINE against `f729d70` (27/9 ->
  26/8): `_validate_decl_eqn`'s `dtype` message interpolated the param on
  one line and the aval's dtype on the next, and the param half is
  canonicalized inside that function now. The MESSAGE count, which is the
  unit this entry quotes, is unchanged at 8.

  The zero is the claim; the 26 is the positive control that makes it mean
  something. What the sweep does not reach is counted rather than silent:
  20 leaves whose type cannot be subclassed (`bool`, `None`). Every one of
  these numbers is now COMPUTED by
  `test_the_recorded_FIGURES_are_the_ones_the_sweep_MEASURES` and
  `test_the_QUOTE_SITE_COUNT_the_record_quotes_is_the_union_it_measures`,
  which red naming this file when they move — that is what stopped the
  same class recurring for the container rule one entry above.

- **AN ELEMENT COUNT COMES FROM `__index__`, NOT FROM `__mul__`** (audit
  0.2.0 B6 audit 4, F3). `obligation._size` was `n = 1; for d in shape: n
  *= d` over the RAW objects — a third protocol beside the `__index__`
  every guard validates with and the `__eq__` the shape comparisons use.
  Measured: an extent with `operator.index(d) == 2` and
  `_shape_problem((d,)) is None` gave `_size((d,)) == 1`.

  The audit named four count-readers that took the predicate face and then
  counted with an unvalidated second read, and said plainly that it could
  not drive any of them to a false verdict — the constvar route is closed
  earlier by the `ir` door, the `_decode_elements` route by the
  byte-length check. **The containment was accidental, and the four were
  not the count either.** A census of `_size`'s call sites at `30d4b04`
  finds **14 whose argument is a shape read straight off an `ir.Aval` or
  an `ir.Array` at the call site** — the divisor probe, the term-count
  pass, four in the element budget, two on the replay path, and more at
  one remove through a local; some with no validation in front of them at
  all. Enumerating them would have been the same defect one level up. So
  the repair is `_size` itself, which now reads its extents
  through `_extents` and declines a shape it cannot count rather than
  returning a product of whatever `__mul__` said. **No caller OF `_size`
  can obtain a count from a third protocol, including one written
  tomorrow.**

  **THE SCOPE OF THAT SENTENCE IS `obligation`, AND IT WAS WRITTEN
  WITHOUT ONE** (audit 0.2.0 B6 audit 5, F3). It stood here as *"no
  caller anywhere"*, which is false one module over: `propagate` does not
  call `_size` and carries **six** raw `n = 1; for d in shape: n *= d`
  products of its own. Measured on this tree, three of them loop over a
  shape read straight off an `ir.Aval` or an `ir.Array` at the site —

  ```
  _refused_value_problem    for d in value.shape       propagate.py:1180
  _atom_element_count       for d in atom.aval.shape   propagate.py:6584
  _declared_element_count   for d in out.aval.shape    propagate.py:10657
  ```

  (line numbers as measured at this commit; the census itself is computed
  from `propagate`'s own AST by `tests/test_aval_lie_both_faces.py::
  test_the_element_count_census_covers_propagate_TOO`, which reds naming
  this entry if a seventh appears or one of these moves.)

  — one more (`propagate._elements`, `propagate.py:814`) is reached at one
  remove and only from `ir.Array.shape` at both its call sites, and the
  remaining two take a caller-supplied `shape` argument. `_declared_
  element_count` is the library's SECOND element-count reader and was
  already named by an earlier audit in `_Slicer._declared_shape`'s
  docstring; the other five appear in this entry for the first time. The
  version of this sentence in `_shape_problem`'s docstring opens *"It is
  `_size`:"* and is correctly scoped; this one is the one a reader quotes,
  and it is now scoped too.

  What makes those six safe is not `_size` and is stated where it is:
  after audit 5's F1, `ir.Aval` and `ir.Array` CARRY the extents their own
  `__post_init__` validated, as plain `int` in a plain `tuple`, so a
  product taken over one of their shapes has no second protocol to reach.

  The four named readers additionally BIND what `_extents` returned, so
  they read each shape once; every other `_size` caller still reads a
  second time, which is safe against a third protocol and is not safe
  against an object that answers `__index__` differently between calls.
  **What contains THAT is the constructor and not the hash** (audit 0.2.0
  B6 audit 5, F1): this entry said *"contained by
  `ClosedJaxpr.content_hash()`, which cannot encode such a param"*, and
  `ir._encode` passes an `int` SUBCLASS through untouched for `json.dumps`
  to read its stored value — so a two-faced `__index__` on an `int`
  subclass hashes perfectly well, and a `tuple` subclass with a drifting
  `__iter__` hashed cleanly at `321209d` while minting a false VERIFIED.
  The containment is that the shapes reaching these readers come off IR
  objects that installed what they validated. Recorded in
  `_shape_problem`'s docstring rather than claimed away.

- **`docs/norms.md`: "unreachable as a guard" now has a QUALIFYING TEST.**
  The paragraph distinguishing a guard from a value read supplied no test
  for when a site qualifies, so it could become an all-purpose excuse for
  a mutation nothing caught. Four clauses, every one required: the site
  has no refusal of its own; the other reader is NAMED at a file and a
  symbol; the divergence is EXHIBITED as a concrete object, not merely
  conceivable; and the site appears as a row with a zero in the same table
  as the reds.

- **ATTRIBUTION FOR AUDIT 4, by MUTATION** — same census method as the
  table above: each mutation asserts its own ANCHOR before it runs (a
  mutation that lands on nothing is an error, not a green run), each is
  driven over the WHOLE suite in its own `git clone`, and the base is the
  same clone unmutated. `JAX_ENABLE_X64=1`, jax 0.11.0, python 3.12,
  `pytest -q -p no:randomly`. **The control is clean at 0 failures** —
  the previous table's `test_sdist_contents` confound is gone because
  both new test files are committed rather than `git apply`ed.

  ```
  mutation                                  raw  conf  NET  the tests that red
  (control: no mutation)                      0     0    0  --  3613 passed, 10 skipped (*)
  F1a widen ir._SHAPE_PARAM_CONTAINERS
      to (tuple, list, memoryview)            5     0    5  rule_itself_is_pinned_to_tuple_
                                                              and_list;
                                                              TRANSFER_face_is_NOT_held_to_
                                                              the_container_rule;
                                                              named_rows...[memoryview];
                                                              ..._DECLINES[memoryview];
                                                              declaration_check_compares_
                                                              BOTH_holders_and_refuses_the_rest
  F1b obligation keeps its OWN copy of
      the rule, one type wider                3     0    3  measured_partition_IS_the_
                                                              documented_rule;
                                                              named_rows...[memoryview];
                                                              ..._DECLINES[memoryview]
  F1c the docstring restates the rule
      instead of naming the object            1     0    1  rule_is_stated_once_and_the_
                                                              prose_points_at_it
  F1d one sentence for both refusals,
      as before the split                    10     0   10  rule_is_stated_once...;
                                                              measured_partition...;
                                                              named_rows... x6;
                                                              declaration_check_compares_
                                                              BOTH_holders...;
                                                              a_hostile___repr___cannot_
                                                              raise_out_of_the_public_
                                                              constructor
  F2a unguard the ir.Array dtype quote        2     0    2  no_message_in_the_ir_validation_
                                                              pass_can_raise;
                                                              named_sites[Array dtype,
                                                              PASSING path]
  F2b unguard the scalar `val` quote          2     0    2  no_message...;
                                                              named_sites[Literal scalar val,
                                                              PASSING path]
  F2c unguard the NamedTupleParam
      where-segment                           1     0    1  no_message_in_the_ir_validation_
                                                              pass_can_raise
  F3a _size back to the raw product           2     1    1  an_element_COUNT_comes_from___
                                                              index___and_not_from___mul__
  F3b _decode_elements counts a SECOND
      read                                    1     0    1  the_named_count_readers_bind_
                                                              what_the_guard_VALIDATED
  ```

  **(*)** the mutation runs were driven at **3613/10**, two tests before
  the tree's final **3615/10**: the two extra rows are
  `test_ir_message_totality`'s `[the duplicate-key LIST itself]` and
  `[_validate_required_params' own primitive quote]`, added afterwards to
  make good on that file's claim to drive the three sites its canonical
  document masks. Neither is a claiming test of any mutated hunk.

  **F3a's one confound is the line-count one this table already names:**
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`
  reds on ANY change to `src/stelling/obligation.py`'s line count, and
  F3a is the only row that changes it. F1b, F1c and F3b edit the same
  file at constant length and do not red it, which is what makes the
  subtraction a line-count effect rather than a behavioural one.

  **AND F1a IS THE ROW THAT SAYS WHAT THE PIN DOES NOT COVER.** Widening
  the rule moves the door and the emission TOGETHER — they read one
  object — so `test_the_measured_partition_IS_the_documented_rule` stays
  **GREEN** on it, correctly: the behaviour still equals the rule. What
  reds is the one deliberate line and the named rows. That is the
  intended division and it is stated here rather than left to be
  discovered: *the computed partition catches DRIFT between the faces
  (F1b) and catches a rule the messages no longer state (F1d); it does
  not and cannot catch a rule someone widened on purpose.*

### Inductive step verification

- **`stelling.inductive.check_inductive_step`**: verify that a loop body
  preserves declared bounds in one step. VERIFIED means the invariant
  holds for all iterations by induction. Constructs the harness
  automatically from the body function and declared state bounds.
  Supports scalar and array-shaped state variables (shape specified per
  variable in the bounds declaration).

### Known limitations (0.2.0)

- **A query's content hash is a function of the jax that traced it, and jax
  0.11.1 moved it for max/min reductions.** jax 0.11.1 added an
  `out_sharding` param to the `reduce_max` and `reduce_min` primitives, so a
  harness containing `jnp.max`, `jnp.min`, `.max()`, `.min()`, `jnp.amax` or
  `jnp.amin` traces to a **different** `query <hash>` on 0.11.1 than on
  0.11.0. **No verdict changes** — both primitives are unmodelled on both
  releases and fall to ⊤ on both — and a stored document keeps its stored
  hash and still loads. What breaks is re-derivation: trace the same source
  on the new jax, compare against a hash stored under the old one, and the
  two differ with nothing raising. If you key anything on that equality (a
  verdict cache, an "already checked?" lookup), re-derive the keys after a
  jax upgrade. Elementwise `jnp.maximum`/`jnp.minimum` are unaffected;
  `jnp.sum` is unaffected. The cause is upstream's `out_sharding` rollout;
  see the 2026-08-18 entry in [SOUNDNESS.md](SOUNDNESS.md).

- **An `assert_` inside a sub-jaxpr does not reach the solver.** Solver
  escalation slices top-level `stelling_assert` equations; an `assert_`
  written inside a `jax.jit` helper, a `cond` branch or a `scan`/
  `while_loop` body is judged by interval propagation and then declines
  escalation, with the reason quoted per obligation. Since the M17 fix it
  costs only ITS OWN escalation — its siblings are decided normally — but
  it is still undecided, so a query containing one cannot reach VERIFIED on
  the strength of the solver. Write the `assert_` at the top level of the
  harness. Lifting this is a capability change rather than a repair, and
  the `cond` case is not merely mechanical: a branch assert is
  CONDITIONAL, so slicing it as an unconditional obligation would be
  unsound.
- **An `assume` inside a `scan` or `while_loop` body is not honoured.** The
  propagation does not enter those bodies, so such an assume narrows
  nothing and is not forwarded to the solver. It is now RECORDED as a
  dropped assumption rather than ignored — the note names the construct and
  the source line, the stamp carries `precondition satisfiability
  uncertified`, and every definite violation is withheld to UNKNOWN — but
  the precondition still does not constrain the analysis. Write it at the
  top level of the harness. Descending the loop is a separate feature: a
  loop body's assume is a per-iteration statement about a carry that
  changes, and this release models neither.
- **The libm accuracy budget is DECLARED, never verified.** stelling
  widens the `exp`/`pow` bracket by the ulps you declare and stamps the
  declaration; it has no way to measure the function your backend
  executes, so a budget smaller than that function's real error mints a
  VERIFIED nothing here can catch. The shipped profile
  `"xla-cpu-2026-08"` is a measurement of **one** jaxlib on **one** device
  class on **one** day, and its name says so; on any other target it is a
  guess with a date on it. There is also no *residual* budget: an
  `(op, format)` pair a budget does not name declines, and stelling never
  extrapolates from one format to another (measured, the same backend
  ranges over 0.50 to 5.51 ulps across the four formats for the same op).
- **`sqrt` under `ieee` still brackets binary64 with a POINT** — no
  outward bump at all — which is sound only because IEEE-754 *requires*
  `sqrt` to be correctly rounded, so `math.sqrt` and the compiled `sqrt`
  must agree bit for bit. That is a standard's guarantee rather than a
  measurement, and it is a genuinely different footing from `exp`/`pow`,
  which IEEE-754 does not constrain at all. A backend that violates it
  (a fast-math build, an approximate reciprocal-sqrt path) is outside
  what this mode can catch, and `sqrt` carries no budget dial to say so.
- `assume(x > 0)` in real mode still narrows to `[0, hi]` (closed
  intervals cannot represent open bounds in exact reals). The IEEE bump
  is exact; the real-mode overapproximation is sound. In real mode, the
  strict-sign certificate — not the box — is what lets boundary-aware
  division use the resulting `[0, hi]`.
- **The strict-sign certificate is dropped by every primitive without an
  explicit rule**, and by every `sub`. So `assume(x > 0); 1/(Σxᵢ² − c)`
  declines even where `c` makes the divisor genuinely nonzero, and
  `assume(x > 0); y = jnp.sqrt(x); 1/jnp.sum(y*y)` declines because
  `sqrt` has no rule (both measured). Sound in that direction (a dropped
  fact can only turn a
  VERIFIED into an UNKNOWN) and extending it is a rule-per-primitive job,
  each rule a soundness claim of its own. It is also whole-array
  granularity — "every element of this value is certainly positive" —
  rather than per-element, so a mixed-sign array carries nothing even
  where some elements are certified.
  A nonzero finite CONSTANT does **not** drop it, whether it reaches the
  rules as a literal (a scalar) or as a constvar (an array): `0.5*Σxᵢ²`,
  `2.0*x`, `x/2.0`, the `/n` inside `jnp.mean`, and
  `jnp.sum(jnp.array([1.,2.,3.,4.]) * x*x)` all keep the chain (measured
  VERIFIED). A constant array must be strictly one-signed THROUGHOUT — a
  mixed-sign weight vector really can sum a positive quadratic to zero —
  and a zero element, a non-finite element, or a dtype with no decoder
  still drops it.
- **The certificate does not cross a sub-jaxpr boundary — `jit`
  included.** Any transparent call wrapper, or a `cond` branch, runs with
  a fresh table, so a division inside one of them sees no certificate
  from its caller and the cond's outputs carry none back. The wrappers
  are `stelling.coverage.DEFAULT_TRANSPARENT` = `jit`, `remat2`,
  `custom_jvp_call`, `custom_vjp_call` — and **`jit` is the one that
  matters in practice**: `assume(x > 0); 1/jax.jit(lambda v: jnp.sum(v*v))(x)`
  is UNKNOWN, and so is the same query with the `assume` moved inside the
  `jit` (both measured, 0.2.0). Earlier text here named only `remat` and
  `custom_jvp`, which understated the cost: almost no jax user writes
  those, and almost every jax user writes `jit`. Conservative in the
  sound direction, and it is what keeps a branch-local assume from
  licensing anything outside its branch.
- **The interval domain cannot represent the sign of an IEEE zero**, so
  under `semantics="ieee"` every divisor box that reaches zero divides to
  ⊤ — including the one-sided shapes real mode tightens, and including
  the ones the subnormal haze creates by hulling a strictly-signed
  interval with `0.0`. Closing this needs a signed-zero lattice threaded
  through every kernel that can produce or consume one, which is a larger
  feature and was deliberately not built here: a half-done version would
  put a trustworthy sign bit on values only some producers set, which is
  the defect S10 already was. Declining to tighten is the sound posture in
  the meantime.
- The dependency problem (A ∧ ¬A = unknown in intervals) is inherent to
  the non-relational domain. Solver escalation is the designed remedy.
- Rational pow requires non-negative base (JAX returns NaN for
  `pow(negative, fractional)`). One cap (128) bounds the degree of the
  emitted `aux^q = x^p` on both sides.
- **A non-integer `pow` exponent escalates only when it is a small dyadic
  rational**, because that is the only case where the emitted rational IS
  the traced binary64 literal. `x**(1.0/3.0)` and `x**0.1` decline to
  UNKNOWN. Admitting them soundly is a larger feature and was deliberately
  not built in this round: it needs the substitution *stamped as an
  assumption*, its amplified error `|x^a − x^(p/q)| ≤ x^a·(e^{|δ|·ln hi} − 1)`
  bounded against the obligation's slack over the declared box, and the
  discharge direction barred until that bound exists. Declining is the
  sound posture in the meantime.
- **A REFUTED through a non-integer `pow` needs a witness whose exact
  value is rational.** The replay extracts exact `q`-th roots; where the
  true value is irrational it reports "witness not independently
  replayable" and the obligation stays UNKNOWN rather than resting on a
  rounded float. Deciding those points needs exact algebraic (not
  rational) arithmetic in the replay, which this release does not have.
- A relational `assume` inside a `lax.cond` branch is **not** forwarded to
  the solver, and is not emitted as an implication either — the drop says
  so. Branch-scoped preconditions therefore buy no solver precision.
- An **unsatisfiable** set of relational assumes makes the emitted script
  `unsat` for a reason unrelated to the obligation, and the discharge that
  follows is vacuous. The unsatisfiable-precondition refusal consults the
  interval domain, which by construction cannot decide a relational
  assume, so it does not see this. Correct forwarding widens the reach of
  this pre-existing limitation from top-level assumes to `jit`-carried
  ones; see the SOUNDNESS.md entry of 2026-08-14.
- An obligation discharged with a **forwarded relational axiom cannot
  narrow the VERIFIED bar**: the bar's re-derivation re-slices without the
  propagation, so its script does not carry the axiom and the two do not
  match. In a query containing a barred primitive the bar therefore falls
  back to the whole query. Conservative (a wider bar, never a narrower
  one), pre-existing, and made more frequently reachable by this release;
  see the SOUNDNESS.md entry of 2026-08-14.

---

## 0.1.0 — 2026-08-12

Initial release.

### Static verification

- Forward interval propagation over the jax-free IR, outward-rounded (one
  deliberate ulp per operation), with three-valued verdicts: VERIFIED,
  REFUTED, UNKNOWN.
- SMT escalation via an optional portfolio (cvc5 for nonlinear, Z3 for
  linear, cross-checked when both are installed). REFUTED verdicts carry a
  concrete witness confirmed by exact-rational replay.
- Every verdict carries a full stamp: versions, query content hash,
  arithmetic mode and semantics, precision configuration, solver
  invocations (or their recorded absence), transfer tiers and provenance,
  assumptions, and coverage.
- Precondition obligation templates (`field_positive`, `scalar_nonzero`)
  with a one-call entry point (`check()`).
- Vacuity checking (two modes: `inputs-only`, `all`) built into the
  pipeline — a VERIFIED that does not depend on its declared envelope says
  so in itself.
- Affine (zonotope) refinement layer for interval-undecided obligations,
  opt-in via `refine="affine"`.
- IEEE-semantics mode (opt-in): judges censused binary64 behaviours and
  stamps itself separately from real-mode verdicts.

### Overflow tripwire

- `pytest -p stelling.overflow` — hooks the constant-fold site where JAX
  silently narrows out-of-range integer literals during tracing.
- Reports each narrowing with source location, arithmetic, independent
  recomputation, and a one-line reproducer.
- **Gates the verifier**: when the tripwire is armed and a narrowing fires
  during a harness trace, the verdict is UNKNOWN — the pipeline refuses to
  certify a jaxpr that does not represent the program as written.
- xdist support: workers serialise findings back; the controller reports
  the true total and flags lost workers.
- Fail-closed on every JAX version change: probes in both directions at
  arm time, disables itself cleanly if the hook site moved.

### Architecture

- Zero required dependencies. JAX and SMT solvers are opt-in extras,
  imported lazily.
- `import stelling` never imports JAX. Only `stelling/_jax_compat.py` may
  import jax; enforced by pre-commit hook and test.
- REUSE-compliant (SPDX headers on every file), DCO-signed commits, PyPI
  Trusted Publishing with PEP 740 attestations.

### Known limitations

- Control flow (`cond`, `scan`, `while`) falls to top and is counted in
  coverage — not handled.
- Default semantics is real arithmetic (ℝ); a predicate can hold in ℝ and
  fail in floats. The stamp names this.
- The tripwire does not see `jnp.full`, `jnp.where`, `jnp.clip`, eager
  execution, or anything traced before the plugin armed. Each is documented
  and printed on every run.

Tested on JAX 0.10.2 and 0.11.0, Python 3.10–3.12, Linux x86_64.
