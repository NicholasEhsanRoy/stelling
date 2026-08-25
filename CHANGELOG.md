<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## 0.2.0 — 2026-08-24

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
  **Versions: `v0.1.0` and 0.2.0 development builds.** The two missing
  decoder entries are in the released tag — `v0.1.0`'s `_STRUCT_FMT` has
  neither `<f2` nor a `<V2` route, and an ordinary real-mode `check()`
  there answers UNKNOWN on both formats with that note, measured at the
  tag under `jax_enable_x64=True` with no solver. `v0.1.0` has no
  `semantics` parameter at all, so this is not an ieee-only gap. No
  definite released verdict is affected: the failure is an UNKNOWN, and
  an UNKNOWN claims nothing. The reproduction is in
  [`SOUNDNESS.md`](SOUNDNESS.md)'s `## Log`, under the 2026-08-15 B4
  part 1 entry.

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

Strict one-liners: ID, one-sentence statement, affected versions, link.
The predicate, the measurement and the derivation behind each one are in
[`SOUNDNESS.md`](SOUNDNESS.md#020-mode-2-detail-routed-from-changelogmd),
under the same rule that governs the **Soundness** section
(`DOCUMENTATION_ARCHITECTURE.md` §8.3).

**Nothing was summarised on the way**, and that is checked rather than
claimed: every block that left this section is pinned by the sha256 of the
text it left as at `de80ad8`, and has to be found under its own ID in
`SOUNDNESS.md` with the sha256 of the text it arrived as. **One block was
edited in transit** — `M2-0.2.0-01`, whose route census this commit
corrected — and the eleven source lines it did not carry are quoted in
`tests/_soundness_routing_manifest.py` one by one and measured against the
pre-routing file. The check is a partition in both directions, so an entry
cannot be brought into compliance by deleting it.

*Versions.* Nothing in this section reaches a release. Mode 2 is 0.2.0
development work throughout, so `v0.1.0` predates all of it and every M2
finding below was fixed before `stelling.__version__` became `0.2.0`.
(This read *"`v0.1.0` is the only release"* while that was so.)

- **M2-0.2.0-01** — An out-of-range integer constant narrowed at array
  construction RAISES at the line that wrote it, under the opt-in
  `--stelling-eager-truncation=error`. Versions: 0.2.0 development builds
  only. [Detail](SOUNDNESS.md#m2-020-01)

- **M2-0.2.0-02** — `stelling.intentional_wrap(value, dtype)` and
  `stelling.EagerTruncationError` are public, and a declared wrap produces a
  program byte-identical to an undeclared one. Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#m2-020-02)

- **M2-0.2.0-03** — The eager dial reaches the exit code and the session
  report on its own, without `--stelling-overflow` being on. Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#m2-020-03)

- **M2-0.2.0-04** — `expected_truncation` is dynamically scoped and says so,
  after being described as lexically bounded in four places. Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#m2-020-04)

- **M2-0.2.0-05** — `_tripwire.arm()` on an already-armed process returns the
  recorder that is actually recording, not a fresh disconnected one. Versions:
  0.2.0 development builds only. [Detail](SOUNDNESS.md#m2-020-05)

- **M2-0.2.0-06** — Arming fails CLOSED on drift: it verifies the private jax
  function it patches and drives every construction route it claims, in both
  directions. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#m2-020-06)

### The narrowing perimeter (Mode 3), DEFAULT-OFF

A THIRD instrument, opt-in and switched on by nothing else. The detail is
inline here rather than routed to `SOUNDNESS.md`: routing is for text that
LEFT this file and is pinned by the sha256 of what it left as
(`tests/_soundness_routing_manifest.py`), and none of this was ever here.

*Versions.* Nothing here reaches a release. Mode 3 is 0.2.0 development work
throughout, so `v0.1.0` predates all of it and every finding here was fixed
before `stelling.__version__` became `0.2.0`. (This read *"`v0.1.0` is the
only release"* while that was so.)

- **The defect it closes.** `x <= 2**31 - 1` on a `float32` array is a
  program about `2147483648.0` — the literal has no `float32` and jax
  converts it to the next one up, so the comparison that runs is one greater
  than the one written. `stelling.preconditions.check` returned **VERIFIED**
  about that harness, correctly, with every layer below it doing its job: the
  trace is faithful and the moved constant really is in the jaxpr. The
  written number simply never existed. Neither instrument that shipped before
  this can see it — the const-fold tripwire watches integer RANGE, and
  `2147483647` is out of range for nothing; the eager detector watches array
  CONSTRUCTION, and no array is being constructed. Under
  `--stelling-narrowing-perimeter=error` it raises `stelling.NarrowingError`
  at the line that wrote the literal. Driven both ways in
  `tests/test_narrowing_perimeter.py`: VERIFIED disarmed, refused armed, in
  the same process, for `2**31 - 1` and for `16777219`, with `1000`
  unaffected in both directions.

- **The second face, and the door it closes.** The arithmetic and comparison
  slots on the concrete array type — the EAGER spelling, before any trace
  exists. Measured on this tree with **everything stelling ships armed** and
  `jit` warm: `jnp.full((3,), 40000, int16)` is refused by the eager detector
  in the same window, and `x_int16 + 40000` is `-25536` with no fire from any
  instrument, cold or warm, as is `40000 + x_int16` and
  `x_float32 <= 2**31 - 1`. The eager detector watches CONSTRUCTION and
  nothing is being constructed here, so the two are not substitutes. Driven
  both ways with all three instruments armed.

- **The array face's slot list is NOT uniform, and the hole is measured.**
  `__pow__` is not installed: jax lowers `x ** k` to `integer_pow[y=k]` and
  keeps `k` a Python int in the program's own structure, so the written
  integer survives exactly and a guard there is a pure false-positive
  generator (4,647 corpus checks, zero fires under two independent guards).
  `__rpow__` IS installed — `40000 ** x_int16` runs as `(-25536) ** x`. Both
  halves are driven through the installed slots and not only at the
  predicate, because the exclusion is enforced in two independent places.
  `__matmul__`/`__rmatmul__` are installed although jax refuses a scalar
  matmul today; "jax still refuses" is a canary row, not an assumption. No
  in-place slot exists on either face, so `y += 40000` falls back to
  `__add__` and is covered — also a canary row.

- **What it attaches to.** The six comparison slots on jax's tracer type,
  which is the face verification runs through: inside a harness the operand
  is a `DynamicJaxprTracer` and **not** a concrete array, so an array-only
  perimeter never fires during `check()` at all. Six slots and not two,
  because Python maps `N >= x` onto `x.__le__(N)` and `N > x` onto
  `x.__lt__(N)` — a spelling and its reflection share a slot, while a
  spelling and its opposite do not, and users write both. The type is FOUND
  rather than named: the adapter asks jax for a traced value and walks its
  `__mro__` for the class that owns the slots, so a rename is a refusal at
  arm time rather than a hard-coded private import.

- **The predicate is vendored, not written.** `stelling/_tripwire/prop_guard.py`
  is the artefact of the 0.2.0 dunder-perimeter fuzz round, copied in with
  six edits and no others, its self-test and its provenance comment intact.
  **The note in its header says what KIND each edit is, and not only how many
  there are** — one provenance edit (its SPDX lines and the note itself),
  three import-route edits, one behaviour edit and one message-text edit —
  because an import reroute, a change to what the predicate answers after an
  internal fault, and a change to the English in an error message are three
  different claims about how much of the scoring survives the copy, and a
  bare count makes none of them. (The last two land in this batch's fixups
  below; neither moves an answer the corpus was scored on, and the second
  moves no answer at all.) Its 24 cases run in this repository's suite. **The evidence
  behind it is two kinds and they are not interchangeable**: a 482,691-check
  real-corpus census with zero false positives, which establishes that the
  predicate is not trigger-happy on code people write — and a
  204,300-evaluation property census, which is what the thirteen individual
  mitigations are answerable to. The corpus is not evidence about the
  individual rules: `__pow__` carried 4,647 checks with no fires, truediv
  421, bfloat16 214, float16 130, complex64 5, complex128 6, bool 3, there
  was no int4/uint4/float8 or extended-dtype traffic at all, and 14 of the 34
  slots carried none. `docs/overflow-tripwire.md` states that distinction
  where a reader meets the zero.

- **Arm and disarm are SESSION-scoped.** `arm()` takes an owner and `disarm()`
  restores only when the last owner releases. Without that, an idempotent arm
  beside an unconditional disarm means a nested in-process pytest session
  unhooks its parent — the inner session installs nothing (correctly), its
  teardown restores everything (catastrophically), and every remaining outer
  test runs unprotected with **nothing red**. The plugin's owner is the
  session's `Config`, so a nested session is a different owner by
  construction. All four lifecycles are driven — double arm, arm/disarm/arm,
  a raise between the two, and a real nested session — with the original slot
  object's identity asserted at the end of each, and with the counterfactual
  (an anonymous release DOES unhook) driven beside the nested one.

- **It perturbs `source_info`, and that is disclosed rather than denied.**
  Every equation built through an armed slot carries one extra traceback
  frame. `content_hash`, `consts`, `to_dict(include_metadata=False)`, that
  document's `eqns`, the StableHLO text and `check()`'s status and notes are
  byte-identical armed and disarmed; `str(jaxpr)`,
  `to_dict(include_metadata=True)` and any raw `repr()` carrying
  `source_info` are **not**. Do not byte-compare a persisted document across
  an armed boundary. `source_info_util.register_exclusion` does not fix this
  — driven — and has no un-register API, so it is not used.

- **`stelling.NarrowingError` is public and inherits `BaseException`**, so an
  ordinary `except Exception:` cannot swallow a soundness alarm — and, one
  reason further than Mode 2's, because an `Exception` raised inside a
  binary-operator slot can be caught by the operator protocol and turned into
  a silent retry of the reflected operation. The vendored predicate's own
  `OverflowError` subclass is untouched and unused by this module.

- **The escape is the one Mode 2 already has.** `intentional_wrap(value,
  dtype)` needs no support from the perimeter — it returns a value already in
  range, so nothing is left to detect — and `expected_truncation(reason)`
  covers this instrument too, counting, siting and printing every narrowing
  it permits with the reason its author gave. The refusal message also names
  the value the program actually uses, which is the first answer.

- **The report carries the predicate's two decline counters.** A run
  reporting zero refusals with a non-zero decline count is not a clean run,
  it is an unmeasured one, and the terminal section says so in those words.
  The armed slot list is printed with them, as the whole perimeter rather
  than a summary of one.

- **Fail-closed, and the nightly canary keys on the facts it rests on.**
  Arming drives the reference defect through the live slots in **both**
  directions and refuses on `not-invoked` or `cries-wolf`; a missing slot is
  `no-slot`, a moved type is `no-type`. The canary asserts positively that
  the type is still a heap type, still owns the slots, still rebinds and
  restores by identity, that a **warm** traced operation still enters Python,
  and that no in-place slot has appeared — plus the predicate's promotion
  identity against the dtype jax actually converts a literal into. Three
  injected faults are driven against those rows. A displaced perimeter now
  also reaches `_tripwire.displaced()`, which the trace gate consults.

- **The dial can be turned on over this repository's own suite, and the seven
  tests it refused are now declared.** With
  `--stelling-narrowing-perimeter=error` the whole suite passes with the dial
  armed at the end, exit 0, every permitted narrowing printed with the reason
  its author gave. The figures are a measurement of one tree and move when it
  does. Re-measured at `49d1ff4` (this batch on `6c40ddc`) with
  `JAX_ENABLE_X64=0 pytest -q -p no:randomly
  --stelling-narrowing-perimeter=error`, they read **4575 passed, 10
  skipped**, `1473 integer literal(s)` checked, `15 narrowing(s) PERMITTED at 9
  site(s)`; when this
  entry was written they read `4404 passed, 10 skipped`, `1447` and `11 at 9`.
  This entry and `docs/overflow-tripwire.md` carried **4568** and **4565** for
  the same claim at the same tip, so a test now holds the two copies equal.
  With the seven declarations taken back out — measured at the
  time, not re-driven since — it is **7 failed, 4397 passed**, and the report
  reads `1455 ... checked; 15 do not exist`, `7 of those NOT inside an
  expected_truncation region`, `8 PERMITTED at 6 site(s)` — which is the
  measurement they answer.

  Three of the seven are new `expected_truncation` regions
  (`test_tripwire_arm.py`, and two in `test_tripwire_gate.py` where an
  incidental `int8` bound of `200` runs as `-56`). **The other four could not
  use one, and that is measured rather than argued.** `expected_truncation`
  is a single declaration covering BOTH runtime instruments by design — which
  is right for code whose subject is a narrowing and exactly wrong for a test
  whose subject is *Mode 2 firing on* a narrowing. Driven with a region
  there: `assert fired["a + 200"] == (200, -56)` reads
  `'SILENT' == (200, -56)`, and `test_tripwire_gate_coverage.py`'s eager
  inventory reports every `raises` route as `silent` — a detector reported
  blind by the declaration added to keep it running. Those four take Mode 3
  DOWN for the block instead, through `conftest.lowered_perimeter`, exactly
  as that file already detaches Mode 2 when the subject is the unpatched
  program. The helper hands the hold back through the shipped `arm()` with
  its self-check and **raises on a PARTIAL hand-back, not only a total one** —
  a lowering that comes half-way back reads exactly like one that came all the
  way back. Four things are checked and `status.armed` sees only the first:
  `arm()` agreed, every face is installed again, `live_check()` says every
  slot IS the live binding, and each slot's SAVED ORIGINAL is the object that
  was lowered. The last has no other witness. Driven with only `status.armed`
  checked: something binding over a slot while the perimeter is down becomes
  the "original" the next `arm()` captures, the self-check still passes,
  `armed_faces()`, `live_check()` and `owners()` all read `armed` — and after
  the next `disarm()` the live slot is the interloper, permanently, with
  nothing red.

  **Every declaration is narrow, and that is measured rather than asserted.**
  With each declaration replaced by a recorder, the three regions cover
  exactly **one check and one narrowing each**, at the line they are written
  on. The four lowerings cover 1, 1, 8 and 8 narrowings — and each of the
  eight is one of `GATE_COVERAGE`'s own enumerated routes (`x + N`, `N + x`,
  `x - N`, `x * N`, `x < N`, `x & N`, `x // N`, `x % N` at `OVER` on
  `DTYPE`), which is the table those two tests exist to walk. Nothing outside
  a declaration's own subject is covered by it; the counts are recorded beside
  each site.

- **The shared helper is inert where it is not used.** `tests/conftest.py` is
  imported by every test file in this tree, the zero-dep lane included, so
  `test_import_hygiene.py` now pins that importing it pulls in **no jax and no
  numpy**, that it arms nothing (it is a plain function, not a fixture), and
  that entering the block with nothing armed is a no-op yielding `()`. Driven
  in a genuinely jax-less interpreter as well as under the probe.

- **`_isolate` used to unhook the session's own hold, and the dial was
  therefore unmeasurable on the shipped tree.**
  `tests/test_narrowing_perimeter.py`'s autouse fixture restored
  unconditionally — the exact asymmetry `arm(owner=...)` exists to prevent,
  aimed at the plugin's hold. That file is collected EARLY — and before
  `tests/test_tripwire_record.py`, the one file that binds a stand-in over
  the predicate memos this instrument classifies through, which is what
  keeps its verdict a reading of the perimeter rather than of another file's
  teardown. So its first test took the perimeter out and **every test
  collected after it ran unprotected**, while the `reset_counters()` in the
  same fixture wiped the tally the tests before it had already earned. Both
  halves read as one number: the documented dial-on command over the whole
  suite reported `NOT ARMED [detached] ... 0 integer literal(s) ... were
  checked` — **zero**, over the 4,393 tests that passed in that run
  (`e6968fe`). It now records what it found, lowers the hold for its own
  window only, and hands it back by identity through `arm()`, raising if it
  cannot.

  **AND THAT RUN WAS NOT GREEN, WHICH THIS ENTRY USED TO IMPLY IT WAS.** It
  said the later tests ran unprotected *with nothing red*, and a reader can
  take that for "the run came back clean". It did not: re-driven at
  `e6968fe` with the dial on, this file on its own ends `56 passed, 1
  error`, and the error is `ERROR at teardown of
  tests/test_narrowing_perimeter.py::test_the_vendored_predicates_own_selftest_passes_in_this_cell`
  — that version of the fixture ended `assert perimeter.armed_faces() ==
  before == ()`, and it failed on this file's FIRST test, with `assert () ==
  ('tracer', 'array')`, because the session's hold was still standing on the
  way in. The incident reported its own shape, on the first test it touched,
  in the run that produced the zero. What was missing was not a red mark; it
  was a reader.

  **WHERE THAT FILE SITS IN THE COLLECTION IS LOAD-BEARING, AND IT IS
  CHECKED RATHER THAN WRITTEN DOWN HERE.** A rank is a property of a
  checkout's file set: it moves whenever anyone adds a test file, in any
  lane, for any reason — so an ordinal on this page would be a number a
  stranger's tree falsifies, and a page edit would be the price of writing a
  new test. This entry names no ordinal, and no figure that is not a reading
  of a NAMED commit — the pass count above is what one run at `e6968fe`
  reported, which no later checkout falsifies and any reader can go and take
  again. The file derives its own position instead, from `pytest
  --collect-only -q -p no:randomly`, and asserts the two relations the
  incident above actually rests on; a second check reads this page and
  refuses that coordinate in seven spellings rather than the one it was last
  written in, and refuses the *unanchored* form of each, because a figure
  with the commit it was read at against it is a measurement and not a
  coordinate of somebody's tree; and a third holds this entry to carrying
  the phrases the disclosure turns on, because taking the numeral off took
  with it the only thing that had been keeping them on the page.

  **AND THAT THIRD CHECK HOLDS PHRASES, NOT MEANING — WHICH BELONGS ON THE
  PAGE IT GUARDS RATHER THAN ONLY IN ITS OWN DOCSTRING.** It is a
  presence-of-tokens check: it has no notion of polarity, quotation or
  retraction, and an entry that keeps every phrase it looks for while
  denying every claim those phrases belong to passes it — driven, with the
  whole suite green, as does leaving this entry untouched and appending one
  sentence withdrawing it. No arrangement of patterns fixes that, and one
  that looked as though it had would be worse, because its green would read
  as a warrant. So what it holds is that these ten anchors cannot be
  deleted, nor reworded far enough to lose them: the disclosure stays
  LOCATED. Whether it still says this is something only reading it settles.

- **`prop_guard._target_dtype` no longer memoises a fault.** It cached the
  `None` produced inside its own `except` branch, so one transient failure
  blinded the guard for that dtype for the rest of the process while
  `INTERNAL_DECLINES` recorded exactly **1**. Reachable through public API:
  the `truediv` branch asks jax for the promotion by allocating
  `jnp.zeros((0,), dt) / 1`, and `jax.transfer_guard("disallow")` makes that
  raise. Driven — after the window closed the reference defect fired **0 times
  in 20**, over a run of 21 checks whose report named exactly one decline
  while twenty more were answered out of the memo. The branch now returns
  instead of caching; the same drive fires **20 of 20**. This is the
  **fifth** edit to the vendored predicate, and the only one that can change
  an answer it gives; it is
  answer-preserving on the scored corpus, whose `INTERNAL_DECLINES` was empty
  in all nine configurations, so the branch was never taken during scoring.

- **The perimeter's refusal was ungrammatical on two of its four reasons.**
  `prop_guard.Finding.message` interpolates a phrase and then the target
  dtype, and two of the four phrases were not phrases a dtype can follow — so
  the text of a `NarrowingError`, which `perimeter.py` quotes verbatim into
  the sentence it raises, read *"the literal 100000 written in `__le__` is
  overflows float16"* and *"the literal -3 written in `__ge__` is a negative
  literal cannot exist in the unsigned type it is compared against uint8"*.
  Driven end to end through the armed perimeter on all four reasons, before
  and after. `overflows-float` is the reason this release promotes to a
  headline on two pages, so the release documented a user-facing string it
  had left broken. The phrases are a mapping keyed by reason now, rather than
  an `elif` chain whose `else` handed the overflow phrase to anything it did
  not recognise, and `tests/test_narrowing_perimeter.py` holds the keys equal
  to `REASONS` and pins the sentence each reason renders. This is the
  **sixth** edit to the vendored predicate, and the only one that changes no
  answer at all: `classify` is untouched, its 24-case self-test is
  byte-identical across the repair, and a differential over 286,824
  `(dtype, slot, literal, shape)` comparisons — jax 0.10.2 and 0.11.0, both
  x64 cells — reports **0** differences in `reason`, `narrowed_to`,
  `target_dtype`, `literal` or `slot`.

- **And the vendoring ledger moved in the same commit, because it had to.**
  `prop_guard.py`'s note is a claim about provenance that three documents
  restate, with nothing holding any of them equal — which is why the batch
  that found the defect above declined to fix it: a further edit falsified
  every restatement at once, and quietly incrementing a stated count is the
  class this campaign keeps closing. **A hand-grep for one spelling did not
  find them all** — the `**fifth** edit` sentence in the bullet above was
  outside the list one produced, and so were the markers in the file's own
  body. The note now enumerates its edits with a KIND on each and a `TALLY`
  line beside the list; every edit but the note itself is marked at the line
  it changed; and `tests/test_prop_guard_ledger.py` derives the figures from
  that list and holds every restatement in the tree to them — found by
  SCANNING the tree, so a site nobody declared is a failure rather than a
  silence. **The markers are pinned to the list in both directions and are
  deliberately NOT what the figures are derived from**: a marker is a comment
  beside a line of code and is forgotten at least as easily as a sentence, so
  deriving from it would make the gate green on exactly the omission it
  exists to catch. What no in-tree check can see is an edit made with neither
  marker nor entry; the witness for that is the artefact the note names, and
  the sentence telling a reviewer to `diff` against it is itself held here.
  Every fence above and here was driven on its own mutation — a site changed
  alone, a marker added without an entry, an entry added without a marker, a
  count planted in a fourth file, a phrase reverted to the broken one.

- **`arm()`'s exception handler no longer restores faces another owner
  holds.** It restored everything installed rather than what that call
  installed — B8b's shape through the exception door. Driven: OWNER-1 armed
  both faces, OWNER-2's `arm()` faulted, and OWNER-1's perimeter was gone
  while OWNER-1 was still registered and still believed armed.

- **The state guard now watches the perimeter.**
  `tests/_state_guard.py::ENTRIES` had four entries and none covered
  `perimeter._installed`, `_owners`, the 39 live slot bindings or the
  predicate's lazy module caches — while `ci.yml`'s `random-order` lane reads
  a shuffled failure the guard did not name as "state outside that
  inventory". A fifth entry, `perimeter:installed`, reads the saved original
  by identity and the live binding as *which* of the three known objects it
  is, which is stable under a restore-to-equivalent and fires on an
  installation, a release or a foreign patch that outlives a test.

- **Two more `does NOT see` bullets, and Mode 3 now has a "what it does not
  cover" section like Modes 1 and 2.** Its limits lived only in
  `report.PERIMETER_UNCOVERED`, which prints at the END of a session — after
  the decision to arm. Driven with all three instruments armed, the same
  moved `2**31 - 1` threshold in three spellings of one harness:
  `assert_(x <= 2**31 - 1)` is **REFUSED**, `assert_(jnp.less_equal(x, 2**31
  - 1))` is **VERIFIED** and `assert_((x - (2**31 - 1)) <= 0.0)` is
  **VERIFIED** — identical in both x64 cells. Two causes, neither previously
  named: a `jnp.*` FUNCTION form carries no Python operator at all, so no
  slot is entered (`jnp.add(x_int16, 40000)` is `-25536` in silence); and the
  tracer face carries the six comparisons only, so of the 39 armed slots just
  those six can fire on a traced operand — `x - N` inside a harness goes
  through `Tracer.__sub__`, which is not installed. The second is a **scope
  decision, not a jax limitation**: measured, jax's tracer type owns all 27
  arithmetic slots. The printed list's general bullet — *"any narrowing that
  happens on a route with no Python operator on it at all"* — did cover the
  function form, and its three examples were all about tracing and
  caching, which is how a true disclosure gets read as the examples.

- **The float answer was written with its silence axis INVERTED, and the test
  that certified it silenced the alarm.** `docs/overflow-tripwire.md` said a
  float value that overflows *"is seen by nothing"* and *"raises no alarm
  anywhere in this release"*. None of the three instruments sees one — that
  part holds — but the narrowing is numpy's, and some of it numpy reports as
  `RuntimeWarning: overflow encountered in cast`, so
  **`pytest -W error::RuntimeWarning` fails on that part today**. Driven under
  `warnings.simplefilter("error")` over the six cases the page named, in all
  four cells (0.10.2 / 0.11.0 × x64 off/on): **five warn eagerly with x64
  off, three with x64 on** — the eager half is exactly what the x64 flag
  moves — and **five of the six warn inside `jit`**, because a trace embeds
  its constants through that same cast. The sixth is
  `jit(lambda a: a.astype(jnp.float32))` on `[1e300, 1e300]` with the operand
  built outside the measurement window, which is **silent in all four cells**:
  the operand really is an array by then and the conversion really is XLA's.
  The integer cases the page contrasted all six against are genuinely silent,
  eager and traced alike. The page defines
  silence as *"no `RuntimeWarning` you could turn into one"* and names
  `-W error` as common in scientific repos, so the conclusion drawn — *"a
  scope boundary and not a hole … closing it would be a different
  instrument"* — withheld a remedy the reader already had.

  **The gate could not see it**: it wrapped its drive in
  `warnings.simplefilter("ignore", RuntimeWarning)` and passed under
  `pytest -W error::RuntimeWarning`. It asserts the warning now, per case,
  in both directions.

  **The genuinely silent residue is smaller and sharper, and it is the case a
  numerical program actually produces.** The split is HOST versus DEVICE:
  once a value is inside a `jax.Array` no host cast runs and there is nothing
  to warn. Measured silent under the same filter in all four cells: `a * a`
  and `a ** 2` on a `float32` array of `1e30`, `jnp.exp` of a `float32`
  `1000.0`, `a.astype(jnp.float16)` / `lax.convert_element_type(a,
  jnp.float16)` on that array, and `x_f16 + 70000.0` run EAGERLY — each
  `inf`, 0 fires from all three instruments, and `-W error::RuntimeWarning`
  does not reach any of them. The overflow there is *computed*, so there is no
  literal for a literal-watcher to read even in principle.

  **Whether a given line is on the host at all can change with the x64 flag**,
  and two lines change sides between cells to prove it: `x_f32 + 1e300` and
  `jnp.asarray([1e300, 1e300]).astype(jnp.float32)`, run eagerly, warn at
  `JAX_ENABLE_X64=0` and are silent at `JAX_ENABLE_X64=1` — numpy
  canonicalises to `float32` on the way in with x64 off, and XLA does the
  narrowing with it on. They are a declared group in the gate, asserted in the
  direction the running cell is in — and driving all four cells is what caught
  a case this fixup had first filed under WARNS for the wrong reason
  (`jit(a.astype(jnp.float32))` on `[1e300, 1e300]`, whose x64-off warning came
  from the `asarray` that built the operand, not from the `astype` it was
  named for).

- **…and HOST versus DEVICE does not partition it either, which is the second
  correction to the same paragraph.** The fixup's replacement claimed a
  universal — *"where the narrowing is done ON THE HOST, by numpy, the cast
  emits `RuntimeWarning`"* — and drew a remedy from it: `-W
  error::RuntimeWarning` *"covers whatever numpy touched and nothing else."*
  Measured, **it does not cover `bfloat16`**: `jnp.full((2,), 1e300,
  jnp.bfloat16)`, `jnp.array([1e300], jnp.bfloat16)` and `jnp.bfloat16(1e300)`
  are each `inf` with nothing raised, in all four cells, at the same three
  construction doors that raise for `float32` — and `jax.make_jaxpr` already
  holds `inf:bf16[]`, so it is a host narrowing and not the device residue.
  **The page contained its own counter-example** three paragraphs from the
  sentence: its float8 table said `float16` is the only format whose host cast
  warns when traced, which is seven silent host casts beside a claim that host
  casts warn.

  **What decides is the TARGET FORMAT.** The warning comes from numpy's own
  floating-point machinery, which knows only the formats numpy implements
  itself — `float16`, `float32`, `float64`. The **other twelve** float
  formats `jax.numpy` exposes come from `ml_dtypes`, which converts by integer
  bit arithmetic and raises no floating-point flag. The control that settles
  it has no jax in it: one numpy cast loop on one `float32` array of `1e30`
  **warns** into `float16` and is **silent** into `float8_e5m2` (`inf`) and
  `float8_e4m3fn` (`nan`) — the silent ones lost far more. So the page now
  states the remedy exactly — it catches a host narrowing into `float16`,
  `float32` or `float64` and nothing else — and a new gate derives both lists
  from `jax.numpy`, drives all three doors on all fifteen formats, and
  compares the page's table, its enumeration and its remedy sentence against
  what was driven. **Measured before it existed:** inverting the axis
  sentence, rewriting the device-silent case list, and changing the page to
  claim `-W error::RuntimeWarning` *does* reach the device residue were all
  three green.

- **`overflows-float` fires on EIGHT of `jax.numpy`'s float formats, not the
  one the page claimed — and four of them run as `nan`.** The sentence said
  *"among the four catalogued formats only `float16` has a finite range an
  integer literal can leave quietly"*, justified as *"arithmetic rather than
  policy"*, and the justification is what made it wrong: the four were
  `propagate._FLOAT_FORMATS`, the **verifier's** IEEE-mode catalogue, which
  this perimeter's page never introduces and its predicate never consults.
  `prop_guard` has no catalogue — it asks `ml_dtypes.finfo`, which is why F1
  exists and why its own self-test drives `float8_e5m2`. Measured disarmed
  under `simplefilter("error")`, identical in all four cells: `float16` plus
  seven of the eight `float8_*` names it exposes lose an integer literal
  quietly (`float8_e8m0fnu`'s range covers every `int64`), and
  `float8_e4m3fn`, `float8_e4m3b11fnuz`, `float8_e4m3fnuz` and
  `float8_e5m2fnuz` — which encode no infinity — run as **`nan`**, so
  `x <= N` **inverts to `False` everywhere** rather than comparing against
  `inf`. `float16` is the only one of the eight whose host cast warns when
  traced, so on the other seven this perimeter is the only instrument that
  speaks — the same axis the corpus census records as carrying no traffic at
  all. The set is now derived from `jax.numpy` in the test rather than
  borrowed, and compared against the page's table in both directions.

- **Five gates that could not fail.** (1) The page's opening `x + 256` demo
  was compared against a re-implementation in the test, so the page's CODE
  could drift from the answers printed under it — measured: changing it to
  `x + 300` left 325 tests green while the real jaxpr becomes `add a 44:i8[]`
  and the result `[-112, 94, 34]`. The block is executed **out of the page**
  now. (2) Nothing in the tree read the artefact table, so a row moved
  between its halves silently; both directions are pinned against `IDENTICAL`
  and `PERTURBED`. (3) The *what runs* table's cells were matched by
  substring, so `-25536` was satisfied by `-255360`, and the comparison row's
  driven value was discarded and replaced by a re-derivation down numpy's
  cast path; cells are compared exactly and the comparison row is read out of
  the **jaxpr**. (4) `x_f16 <= 100000` appears twice on the page and the
  assertion was `... in page`, so either occurrence kept it green; every
  occurrence is checked. (5) The StableHLO control was blind to what it
  exists to detect — `jit(...).lower().as_text()`, the spelling the page
  quotes, emits **no `loc(` at all**, so the `IDENTICAL` entry could not
  redden on a `source_info` perturbation. `as_text(debug_info=True)` emits
  144 and **DIFFERS**; both are in the table, the `loc(` counts are pinned,
  and the three lowerings are taken from **one source line**, because that
  text records the caller's line number and comparing two call sites
  "differs" with nothing armed. (6) A sixth, found in this fixup's own work
  and the same class as the other five: the silent half's failure message was
  **referenced and not defined**, and every test stayed green, because an
  `assert` message and an unreached `pytest.fail` argument are evaluated only
  on the failing path — so the sentence a maintainer needs at the moment the
  disclosure has become wrong was a `NameError`. It is rendered on the
  PASSING path now, with both call shapes.

- **A test that arms the tripwire no longer detaches the session's hold.**
  `_tripwire.arm()`/`disarm()` carry no owner, so an unconditional `disarm()`
  in a test takes out a session-armed instrument. Under
  `pytest -p stelling.overflow --stelling-overflow=require` —
  the spelling the page tells readers to use — `tests/test_narrowing_perimeter.py`
  ended `NOT ARMED [detached]`, PARTIAL, **exit 1**, with `tests/_state_guard.py`
  erroring at the test that did it. `conftest.borrowed_tripwire` /
  `borrowed_eager` apply the one rule that fixes it — **do not put back what
  you did not take** — and all four sites in that file use them; the same
  command now reports `armed`, exit 0.

- **The runtime message shipped the sentence the page had corrected.**
  `report._suggestions` still read *"it rejects a Python int at none of the
  eleven, which is the spelling in front of you"* — contradicted by the
  bullet directly above it in the same list, which tells the reader that
  `jnp.array(N, dt)` raises `OverflowError` for a Python int. It now says
  which rejection a bare Python int does not get (`TypePromotionError`, at
  none of the eleven) and which three doors raise on the **value** instead,
  and one measurement holds both the page's row and the message.

### Verification pipeline

- **`check(..., falsify="sample")` — the falsification probe, DEFAULT-OFF
  and UNAUDITED.** It SHIPS in 0.2.0 — this entry read *"and UNRELEASED"*
  while 0.2.0 was a development line, and the version bump is what made
  that false. Nothing about the keyword changed: it is off by default, no
  audit has been run against it, and it is not a surface to build on. A new
  keyword on `stelling.preconditions.check`,
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
  yet. Three of the six fixtures `tests/test_falsify_probe.py`'s live corpus
  USED TO HOLD are `scatter` and now decline; they are listed there, in
  `DECLINED_FOR_WANT_OF_AN_EXACT_READING`, with the primitive that costs each
  one, so `LIVE` holds the three that remain.

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

- **REMOVED — the reachability conjunct, and `src/stelling/reachability.py`
  with it** (audit 0.2.0 B8a, item 4 / **M1**; `f82b87b`). This section
  described it as shipping: *"a backward walk from the jaxpr's outputs
  identifies variables that flow to an output. Violated obligations on
  'dead' variables … are downgraded from REFUTED to UNKNOWN with a
  note."* **The downgrade could not fire on a dead variable at all.**
  `reaches_output` seeds every assert's outvars — correctly — so the
  reverse walk makes every assert's invars, which is exactly the set the
  conjunct tests, live by construction. Its one reachable input was a
  var-id COLLISION, and there the downgrade is WRONG: driven on hand-built
  IR, a `jit` body asserting `x < 0.5` over `x ∈ [1, 2]` is REFUTED, and
  adding one unrelated, never-read `exp` equation whose outvar id equals
  the inner predicate's id turns the same query UNKNOWN. So its only
  reachable act was to silence a true REFUTED. Repairing it would mean
  un-seeding the asserts, which is reverting a correct decision; the
  reachability question that IS real here is answered by certificate in
  `propagate._withhold_uncertified_branch_refutations`, which is
  untouched. `docs/harness-api.md` promised the downgrade and was
  corrected in the same commit. No verdict that was correct becomes
  incorrect: what moves is UNKNOWN → REFUTED on the collision shape.
  `tests/test_reachability_removed.py` is the removal's regression test.

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

  **Versions: `v0.1.0` and 0.2.0 development builds.** The count check is
  in the released tag: `v0.1.0:src/stelling/obligation.py`'s
  `slice_unknown_obligations` carries it verbatim, and driven there under
  jax 0.11.0 with `JAX_ENABLE_X64=1` and both solver wheels, a query of
  one top-level `assert_` plus one `@jax.jit`-nested one comes back
  UNKNOWN with **both** obligations quoting *"escalation declined: 2
  obligation(s) but 1 top-level stelling_assert equation(s)"*, where the
  top-level one alone is REFUTED. **If you are holding a stored verdict
  whose per-obligation detail carries that sentence, it came from a build
  with this defect — `v0.1.0` included — and 0.2.0 may now decide it.** No
  verdict was ever wrong: the old behaviour declined, it never asserted.
  The drive at the tag is in [`SOUNDNESS.md`](SOUNDNESS.md)'s `## Log`,
  under the 2026-08-15 B6 entry for M17.

  **A nested `assert_` is still not sliceable**, and its own obligation
  still declines — with a reason that now names the actual cause instead of
  an arithmetic mismatch. Every obligation still undecided after this fix,
  on the corpus above, is a nested one.

- **A message about a declared input now uses the name the witness uses**
  (audit 0.2.0 B8a, item 5 / **M3**, and its fixup; `f82b87b`). Two 0-based
  namespaces met in one sentence and nothing related them: a
  two-declaration query numbers declaration #0 as IR var 1 and #1 as IR
  var 2, so an unsatisfiable-assume message read *"var 2"* about the input
  the witness calls `x1`. `coverage.declaration_name` is the single minter
  of the published name now, `propagate._declaration_names` derives the
  same index by the same walk `obligation._Slicer._flatten` uses, and the
  messages read `x1 (IR var 2)` — the witness's name with the internal id
  beside it, so a reader tracing a note back into the IR still can.
  **Where the two walks could diverge the numbering is ABANDONED** and the
  message falls back to `IR var N`, which cannot be read as a declaration
  index: a declaration under a `cond`, a wrapper the slicer will not
  inline, or any id collision in the scope — keyed on every id the scope
  binds, not only declaration outvars, because the first spelling missed a
  declaration's `(scope, id)` reused by an ordinary equation and named
  `-x0` as `x0`, turning an uninformative id into a confident wrong name.
  **MESSAGE TEXT ONLY**: no transfer, judgment, counter or hash reads it,
  and no verdict moves. Reachable only through `from_dict` or hand-built
  IR, since jax tracing is SSA.

- **A discharge over a ZERO-ELEMENT predicate says which kind of true it
  is** (audit 0.2.0 B8a, item 6 / **M18**; `f82b87b`). `jnp.all` of an
  empty array is true because there is no element, and the verdict said
  *"definitely true for all 0 element(s)"* — true, and read by every
  consumer as an ordinary discharge. The **status is unchanged**, because
  a universal over the empty set IS true and downgrading it would be a
  false UNKNOWN; the obligation detail now names the case and a note
  carries it into `Verdict.notes` and the render, on the assert face and
  the membership face alike: *"obligation #0 at …: is VACUOUSLY
  discharged: its predicate has ZERO elements, so 'true for every element'
  holds because there is no element"*.

- **The stamp's `nonvacuity` field takes SIX values, two of them new**
  (audit 0.2.0 B8a fixup, item 2; `f82b87b`, docs at `83b3f1d`). The
  zero-element case above was still being counted into the plain
  `checked — N membership condition(s) definitely true (the declared set
  contains the stated point)` line, which `Verdict.render` prints ABOVE
  the notes: no point had been tested and the parenthetical was false.
  Added: **`VACUOUS — N membership condition(s) hold over ZERO elements: no
  point was tested, so nothing ties the declared set to the incident's
  data`** when every condition is vacuous, and **`checked in part — K of N
  …; the other N−K hold VACUOUSLY over ZERO elements and tested no
  point`** when only some are. The all-vacuous prefix is deliberately not
  `checked`, so `make_verdict`'s may-be-vacuous VERIFIED caveat — gated on
  that prefix — now fires for it; a mixed run HAS tied the declared set to
  a stated point and stays `checked`. A run with no vacuous condition is
  byte-identical to before. The sentence had two hand-kept copies with a
  comment requiring them to stay identical — a requirement is not a
  mechanism — and there is **one minter** now,
  `propagate.nonvacuity_summary`, which both verdict paths call.
  `docs/harness-api.md` and `docs/reading-a-verdict.md` were corrected
  from "five values" to six, with `VACUOUS` added to the latter's
  failing-looks-like list.

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

Strict one-liners: ID, one-sentence statement, affected versions, link.
The predicate, the measurement and the derivation behind each one are in
[`SOUNDNESS.md`](SOUNDNESS.md) — this project's soundness ledger and the
single source of truth for them (`DOCUMENTATION_ARCHITECTURE.md` §8.3).

**Nothing was summarised on the way.** Each entry's detail was ROUTED
there whole, block for block, and `tests/test_soundness_routing.py`
checks the move rather than asserting it: every block that left this
section is pinned by the sha256 of the text it left as, and has to be
found under its own ID in `SOUNDNESS.md` with that hash. **Eight blocks
were edited in transit** — `SF-0.2.0-07`'s route census, which said 32
construction routes and 7 `unwatched` against a dict holding 33 and 8;
`SF-0.2.0-51`'s solver-workaround justification, which named a case the
emission cannot produce; `SF-0.2.0-59`'s hash literal, which nobody
could reproduce; and three stamped measurements that had gone stale
underneath their own entries — `SF-0.2.0-46`'s account of what an
infinite endpoint costs `mul`, `SF-0.2.0-62`'s container-population
census, and `SF-0.2.0-64`'s element-count census together with the four
`propagate.py` line numbers it quoted, all four of which now point at
unrelated code. **Two more, and a SECOND edit to `SF-0.2.0-59`, are that
same kind and were made on 2026-08-23**: `SF-0.2.0-56`, `SF-0.2.0-66`
and `SF-0.2.0-59` each record a revert or mutation experiment that
subtracted a line-count confound the generated
`docs/supported-primitives.md` stopped creating when it stopped carrying
source-line coordinates. All eight are declared in the manifest with
the reason, with a count of the source lines the edit did not carry,
and with those lines QUOTED — and the count and the quotation are
both measured against
the pre-routing file rather than taken on trust, which they were not
until the B8c fixup. The digit in this sentence is derived from the
manifest too, so a ninth edit cannot leave it reading eight. The check
is a partition in both directions: an ID missing from either file, an ID
in one and not the other, and a detail section nothing links to are each
a failure, so **an entry cannot be made to comply by deleting it.**

*Versions.* `v0.1.0` was the only release when these IDs were minted and
`0.2.0` is the second. An entry marked *0.2.0 development builds only*
reaches NEITHER: it arrived after the `v0.1.0` tag and was fixed before
`stelling.__version__` became `0.2.0`. An entry marked *`v0.1.0` and 0.2.0
development builds* reaches `v0.1.0`, was reproduced at the tag, and
`SOUNDNESS.md` carries its retroactive-invalidation scope and what to
re-run — which for `SF-0.2.0-46` is *nothing to re-run*: it reaches
`v0.1.0` as a precision loss that never produced a wrong verdict there,
and reaching a release is about the defect, not about how bad it was.

*The IDs are positional.* They were minted once, at 0.2.0, in the order
the entries stood in this file at `8f0adf2`; they are stable labels and
not a ranking.

- **SF-0.2.0-01** — The trace gate now consults the tripwire's displacement
  check. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-01)

**Batch B15 — the trace gate observed part of a program and claimed all of
it** (`fix/B15-trace-gate-observation`) —
[SF-0.2.0-02](SOUNDNESS.md#sf-020-02)

- **SF-0.2.0-03** — A warm jit trace cache no longer hides a narrowing from
  the gate. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-03)

- **SF-0.2.0-04** — The gate has three states where it had two, and the
  third has its own sentence. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-04)

- **SF-0.2.0-05** — Arming the trace gate has a real, measured user-visible
  cost: every `check()` calls the process-global `jax.clear_caches()`, which
  drops the caller's own compiled functions. Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#sf-020-05)

- **SF-0.2.0-06** — The gate's cache eviction cost NOTHING in verdicts — 1475
  armed gated traces at `a759809`, 88 not fully observed, 0 of them narrowing,
  VERIFIED unchanged at 312. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-06)

- **SF-0.2.0-07** — The tripwire's coverage claim is now an asserted
  inventory. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-07)

- **SF-0.2.0-08** — Three disclosures the trace-gate batch narrowed are
  corrected in the pages that made them: `design/d4-wrap-disclosure.md`,
  `docs/quickstart.md` and the README's retracted faithfulness claim.
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-08)

- **SF-0.2.0-09** — `design/d4-wrap-disclosure.md`'s flagship worked example
  did not run. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-09)

**Batch B13 — the instruments that read as enforcing something**
(`fix/B13-instrument-reach`) — [SF-0.2.0-10](SOUNDNESS.md#sf-020-10)

- **SF-0.2.0-11** — A verdict replayed on a different jax now says so.
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-11)

- **SF-0.2.0-12** — The const-fold tripwire's "this release has never been
  read" carve-out no longer waves through real jax releases. Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-12)

**Batch B12 — the from_dict document-schema batch**
(`fix/B12-from-dict-structure`; audit 0.2.0 S15, S16) —
[SF-0.2.0-13](SOUNDNESS.md#sf-020-13)

- **SF-0.2.0-14** — `ClosedJaxpr.from_dict` now judges the TYPE the code
  declares at every position it stores a document value at (audit 0.2.0
  **S15** and **S16**). Versions: `v0.1.0` and 0.2.0 development builds.
  [Detail](SOUNDNESS.md#sf-020-14)

*B7 and B6 branched from one commit, were developed in parallel, and merged
on 2026-08-16 — how their figures relate* —
[SF-0.2.0-15](SOUNDNESS.md#sf-020-15)

*The merged tree's suite counts, and the two batches' additivity* —
[SF-0.2.0-16](SOUNDNESS.md#sf-020-16)

**Batch B6 — the plan-structure batch** (`fix/B6-plan-structure`; audit
0.2.0 S12, S12&prime;, S12&Prime;, S14, M17&prime;) —
[SF-0.2.0-17](SOUNDNESS.md#sf-020-17)

- **SF-0.2.0-18** — `dot_general` shape well-formedness is now ONE
  definition, shared by the interval transfer and the SMT emission (audit
  0.2.0 **S12**). Versions: `v0.1.0` and 0.2.0 development builds.
  [Detail](SOUNDNESS.md#sf-020-18)

- **SF-0.2.0-19** — The emission may not model a DIFFERENT ARRAY than the
  propagation did: one shape per value, checked for every primitive at once
  (audit 0.2.0 **S12′**). Versions: `v0.1.0` and 0.2.0 development builds.
  [Detail](SOUNDNESS.md#sf-020-19)

- **SF-0.2.0-20** — A non-integer `dimension_numbers` entry no longer
  raises a raw `TypeError` out of the public `propagate()`, and
  `interval.dot_general_geometry` keeps its documented contract (audit
  0.2.0 **S12″**). Versions: `v0.1.0` and 0.2.0 development builds.
  [Detail](SOUNDNESS.md#sf-020-20)

- **SF-0.2.0-21** — A guard that refuses a malformed extent can no longer be
  stopped by the extent (audit 0.2.0 B6 **audit 3**, F1/F2/F3). Versions:
  0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-21)

- **SF-0.2.0-22** — A declaration's `shape` param is accepted by a POSITIVE
  rule (audit 0.2.0 B6 audit 3). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-22)

- **SF-0.2.0-23** — `slice_unknown_obligations` can no longer raise (audit
  0.2.0 **M17′**). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-23)

**Batch B11 — the propagation identity** (`fix/B11-propagation-identity`) —
[SF-0.2.0-24](SOUNDNESS.md#sf-020-24)

- **SF-0.2.0-25** — SOUNDNESS FIX — a `Propagation` from one query could be
  stamped as a verdict about another, minting a false VERIFIED (and a false
  REFUTED) on the released `v0.1.0` and on every 0.2.0 revision up to
  `207faca`. Versions: `v0.1.0` and 0.2.0 development builds.
  [Detail](SOUNDNESS.md#sf-020-25)

**Batch B7 — the `pow`-row bar and gauge batch** (`fix/B7-bar-gauge`, landed
on `main` at `198a2b5`; audit 0.2.0 M10, S4) —
[SF-0.2.0-26](SOUNDNESS.md#sf-020-26)

- **SF-0.2.0-27** — The VERIFIED bar's re-derivation is given the query's
  forwarded relational assumes (audit 0.2.0 **M10**). Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-27)

- **SF-0.2.0-28** — The `pow` emission row has a fidelity gauge, and stays
  out of the VERIFIED bar (audit 0.2.0 **S4**). Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#sf-020-28)

- **SF-0.2.0-29** — The `pow` fidelity gauge was overfit to the exponents its
  own battery drove, and states its exponent reach as a MEASUREMENT now rather
  than as a branch count. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-29)

- **SF-0.2.0-30** — `docs/gauge-coverage.md`'s coverage figures are
  recomputed from the battery instead of typed. Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#sf-020-30)

- **SF-0.2.0-31** — `pow`'s rational branch no longer has an unreachable arm
  that reads as covered. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-31)

- **SF-0.2.0-32** — The `pow` fidelity gauge measured two of its row's three
  seams: `smt._pow_aux_name`, the only one handed the array shape, was left as
  prose. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-32)

- **SF-0.2.0-33** — The `pow` gauge's blind spot then moved to
  `_pow_integer_body`'s OTHER argument, the BASE TERM, which every fixture in
  the battery drove at one value. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-33)

- **SF-0.2.0-34** — Enumerating the `pow` gauge's parameters closed the LIST
  and left every parameter's RANGE as open as before, which is the defect the
  enumeration itself shipped. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-34)

- **SF-0.2.0-35** — Two documented claims about the `pow` gauge were out of
  date in `docs/gauge-coverage.md`, and the test reading them was missing its
  sibling's inconsistency guard. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-35)

- **SF-0.2.0-36** —
  `test_the_documented_coverage_figures_are_the_MEASURED_ones` accepted
  incomplete gauge-coverage tables and rejected a correct one. Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-36)

- **SF-0.2.0-37** — `_rat_denominator_false_harness`'s docstring described a
  rational-`pow` emission that the same round had just outlawed. Versions:
  0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-37)

- **SF-0.2.0-38** — `exp` and `pow` under `semantics="ieee"` now require a
  DECLARED libm accuracy budget (audit 0.2.0 **S9** and **S11**). Versions:
  `v0.1.0` and 0.2.0 development builds. [Detail](SOUNDNESS.md#sf-020-38)

- **SF-0.2.0-39** — A rational-`pow` exponent was admitted on a binary64
  distance test that reads exactly `0.0` for `0.1`, so
  `limit_denominator(128)` decided identity (audit 0.2.0 **S1**). Versions:
  0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-39)

- **SF-0.2.0-40** — No emitted term is a unary `(* t)` (audit 0.2.0 S2).
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-40)

- **SF-0.2.0-41** — The rational-`pow` replay is exact (audit 0.2.0 S3, M8).
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-41)

- **SF-0.2.0-42** — A non-integer `pow` over a declaration-independent base
  was stamped `QF_LRA` while its emission wrote `(* aux aux)`; the fragment
  stamp follows the aux encoding now (audit 0.2.0 **M9**). Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-42)

- **SF-0.2.0-43** — An IEEE divisor box that reaches zero divides to ⊤
  (audit 0.2.0 S10). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-43)

- **SF-0.2.0-44** — A real-mode divisor box that reaches zero declines
  unless a strict `assume` excludes the zero (audit 0.2.0 B5-1). Versions:
  0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-44)

- **SF-0.2.0-45** — `boundary_div` answers `inf/inf` instead of raising
  (audit 0.2.0 B5-3). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-45)

- **SF-0.2.0-46** — `mul` is exact when its corner products are
  representable (audit 0.2.0 M16). Versions: `v0.1.0` and 0.2.0
  development builds. [Detail](SOUNDNESS.md#sf-020-46)

- **SF-0.2.0-47** — A relational `assume` over two variable operands is
  forwarded to the solver as a positive axiom instead of being dropped by the
  interval domain. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-47)

- **SF-0.2.0-48** — SOUNDNESS FIX — a forwarded assume is now resolved by a
  scope-correct identity; it could previously be emitted about the wrong
  values. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-48)

- **SF-0.2.0-49** — SOUNDNESS FIX — a withheld violation is released only
  when every `assume` is accounted for, and that is now decided by a
  per-assume LEDGER rather than by two counts. Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#sf-020-49)

- **SF-0.2.0-50** — SOUNDNESS FIX — a discharge is no longer accepted when
  an EMPTY assumed region alone explains it. Versions: 0.2.0 development
  builds only. [Detail](SOUNDNESS.md#sf-020-50)

- **SF-0.2.0-51** — A z3 tactic chain replaces the default `Solver()` on
  obligations carrying a rational-`pow` auxiliary variable, restoring the
  cross-check on high-degree polynomials. Versions: 0.2.0 development builds
  only. [Detail](SOUNDNESS.md#sf-020-51)

- **SF-0.2.0-52** — A definite violation is un-withheld only when every
  `assume` the user wrote is accounted for on THAT obligation's query, rather
  than on the query as a whole. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-52)

- **SF-0.2.0-53** — An assume that excludes nothing no longer withholds
  forever. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-53)

- **SF-0.2.0-54** — Emission guards resolve through inlined aliases.
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-54)

- **SF-0.2.0-55** — An `assume` inside a `scan` or `while_loop` body is
  recorded instead of ignored (audit 0.2.0 S13). Versions: `v0.1.0` and
  0.2.0 development builds. [Detail](SOUNDNESS.md#sf-020-55)

- **SF-0.2.0-56** — The B6 batch's per-change attribution table is published
  WITH its census method, so every row can be re-derived rather than trusted
  (audit 0.2.0 B6 audit 3, F5). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-56)

- **SF-0.2.0-57** — A SHAPE IS JUDGED BY THE SAME RULE WHEREVER IT APPEARS,
  AND THE ONE PLACE IT WAS NOT WAS REACHABLE FROM A JSON FILE (audit 0.2.0
  B6 audit 8). Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-57)

- **SF-0.2.0-58** — The record's own claims were re-read against the code and
  corrected in one place where measurement contradicted them, rather than
  edited silently (audit 0.2.0 B6 audit 8). Versions: 0.2.0 development builds
  only. [Detail](SOUNDNESS.md#sf-020-58)

- **SF-0.2.0-59** — UNSOUND — THE DOOR'S OWN DISPATCH WAS BUILT FROM THE TWO
  MOST OVERRIDABLE TESTS IN PYTHON, SO IT COULD BE WALKED PAST; IT DECIDES
  BY IDENTITY NOW (audit 0.2.0 B6 audit 7, **S14**). Versions: `v0.1.0` and
  0.2.0 development builds. [Detail](SOUNDNESS.md#sf-020-59)

- **SF-0.2.0-60** — UNSOUND — THE DOOR NOW STORES EVERY DOCUMENT-SUPPLIED
  VALUE AS AN EXACT BUILT-IN, OR REFUSES IT; CLOSING THE PAIRS ONE AT A TIME
  IS WHAT KEPT THIS OPEN (audit 0.2.0 B6 audit 6). Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-60)

- **SF-0.2.0-61** — UNSOUND — A GUARD MUST INSTALL THE VALUE IT VALIDATED,
  NOT MERELY RETURN IT (audit 0.2.0 B6 audit 5, F1). Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-61)

- **SF-0.2.0-62** — A CLAIM ABOUT CONTAINER TYPES IS COMPUTABLE, SO IT IS
  COMPUTED (audit 0.2.0 B6 audit 4, F1). Versions: 0.2.0 development builds
  only. [Detail](SOUNDNESS.md#sf-020-62)

- **SF-0.2.0-63** — "EVERY QUOTE HERE IS GUARDED" WAS FALSE 44 LINES BELOW
  ITSELF, and the four sites were six short (audit 0.2.0 B6 audit 4, F2).
  Versions: 0.2.0 development builds only. [Detail](SOUNDNESS.md#sf-020-63)

- **SF-0.2.0-64** — AN ELEMENT COUNT COMES FROM `__index__`, NOT FROM
  `__mul__` (audit 0.2.0 B6 audit 4, F3). Versions: 0.2.0 development builds
  only. [Detail](SOUNDNESS.md#sf-020-64)

- **SF-0.2.0-65** — `docs/norms.md`: "unreachable as a guard" now has a
  QUALIFYING TEST. Versions: 0.2.0 development builds only.
  [Detail](SOUNDNESS.md#sf-020-65)

- **SF-0.2.0-66** — Audit 4's attribution is established BY MUTATION: each
  mutation asserts its own anchor, runs over the whole suite in its own clone,
  against an unmutated control (audit 0.2.0 B6 audit 4). Versions: 0.2.0
  development builds only. [Detail](SOUNDNESS.md#sf-020-66)

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
- An **unsatisfiable** set of relational assumes is now REFUSED, not
  discharged — but only when a backend can decide the admitted region.
  Before an `unsat` is credited on an obligation whose script carries a
  forwarded relational axiom, the same backend is asked the same script
  with the negated obligation removed: `unsat` there raises
  `UnsatisfiableAssumptionError`, `sat` with every assume accounted for
  leaves the discharge clean, and an UNDECIDED region leaves the discharge
  standing with a `[MAY BE VACUOUS: …]` obligation detail and a
  `precondition satisfiability uncertified` stamp line. That last case is
  the limitation that remains: a region the backend cannot decide is
  DISCLOSED, not refused. *This bullet said the refusal "does not see
  this" from `f54990c` (2026-08-14 17:53) until the 0.2.0 release; audit
  S7 closed it in `1dc1b52` five and a half hours later the same day, and
  the bullet did not move* — see the SOUNDNESS.md entry of 2026-08-14 and
  `tests/test_vacuous_precondition.py`.
- *WITHDRAWN, and kept rather than deleted because it stood here as a
  limitation and stopped being one. **No release ever carried it**: it
  existed between 2026-08-14 and 2026-08-15, after the `v0.1.0` tag and
  nine days before `0.2.0`, and this line said "shipped" for one day.*
  This bullet said that an obligation
  discharged with a forwarded relational axiom **cannot narrow the
  VERIFIED bar**, because the bar's re-derivation re-sliced without the
  query's relational assumes and so re-emitted the recorded script minus
  its `(assert …)` axiom lines. That was true from `f54990c` until audit
  0.2.0 **M10** (B7, `48e836f`, 2026-08-15), which hands the re-derivation
  the same assumes; the bar narrows on such a query now, and
  `tests/test_verified_bar.py::test_a_FORWARDED_AXIOM_does_not_cost_the_verdict_its_scope`
  is the anti-vacuity-guarded measurement of it. The bullet stood
  unamended for nine days and was found by re-reading this section against
  the release at the 0.2.0 bump; see the SOUNDNESS.md entry of
  2026-08-15 (B7).

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
