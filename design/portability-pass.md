# The portability pass — making the precondition class portable

**Status:** PASS RECORD, 2026-07-19. Consolidation, not capability: two
structuralizations, a user-facing front door, documentation, and a
portability self-audit. Standing gates: **no recorded verdict may flip;
the `widen()` extraction must be byte-identical at every call site.**
Both held — evidence below.

## The goal-driven rerank, recorded

The next step had been the LA contract, justified by qMRI needing
conditioning bounds. **That justification is withdrawn** (qMRI
deprioritized), and the stated goal is now *"stelling broadly useful for
MADDENING, MIME, and hopefully other people's frameworks."* Under that
goal: LA contract and affine are **deepening** work on Nick's own
solvers — roadmapped, not next — and the **precondition class** is the
most portable, most broadly-useful capability the tool has (pre-solve,
interval-shaped, no walls, ported to a second codebase with a real
finding). The §6 blockers from the magnetics pass therefore stopped
being polish and became blocking: a capability that needs an expert per
codebase is not broadly useful. This pass is that fix. Recorded here as
an **owner decision** (a rerank is a decision, not a lesson — the ledger
records the pass's L12 instance; decisions live in design records).

## §1b — `widen()` extracted: one implementation, imported everywhere

Measured before touching anything: **three copies** in the corpus —
`tautology_test.py` and `maddening_cfl.py` carrying byte-identical
"all-declarations" implementations, `mime_fvm.py` carrying the
"inputs-only" variant differing by exactly one condition
(`params["lo"] != params["hi"]`). Precisely L12's drift shape, on the
instrument every vacuity-gated count relies on.

`stelling/vacuity.py` now holds the single implementation:
`widen(closed, *, mode)` with the two **registered** procedures as modes
(`"all"` — `design/obligation-vacuity.md`; `"inputs-only"` — the fixed
successor, `design/mime-fvm-job.md`). `mode` is **required** — the two
procedures answer different questions and a silent default would let a
harness run the wrong one without saying so. The finite-⊤ /
named-range-theorem criterion is deliberately **not** encoded (it is a
registration-level counting rule, not an IR transform; code would
misrepresent its status). Zero-dep; the nested-declaration loud guard
kept. The three harnesses now delegate to it.

**The byte-identical gate, measured:** all three harnesses captured
before and after. Outputs identical modulo (a) solver wall-times and
(b) **rendered source line numbers** — the delegations are shorter than
the local defs, so later addresses shift, which is the
"cache the proof, not the report" rule behaving as designed (file/line
re-derived from current source, never stored). The decisive invariant:
**every query content hash identical before and after** (3 + 3 + 4
hashes, all matching) — the queries are bit-identical; only rendering
addresses moved.

## §1a — the template gap closed, tested on the shape that exposed it

`field_positive`'s transform may now return a **tuple/list**; one
obligation is stated per produced value, the return mirroring the
transform's shape. The §1a acceptance is a permanent test
(`test_field_positive_poses_the_magnetics_face_shape_without_hand_adaptation`):
the magnetics SPD check's real shape — one cell field, the code's
`a = θ` path, conservative face averaging via `roll` producing two face
arrays — poses **through the template directly**, VERIFIED over the
supported envelope and honestly UNKNOWN over the sign-spanning one.
The shape that needed a hand-written harness last pass no longer does.

## §2 — the front door, the docs, and the self-audit

- **`stelling.preconditions.check(harness, *, solver_timeout_ms=None)`**
  — trace → propagate → stamped verdict, escalating **only** when a
  timeout is passed explicitly (never-on-defaults applied to the
  convenience layer; no implicit solver budget exists). Stamps filled
  from the live environment; the precision entry records the *actual*
  `jax_enable_x64` state at trace time. One hygiene note: the first cut
  imported jax directly and the AST hygiene test rejected it — the
  accessor (`x64_enabled`) lives in `_jax_compat`, where jax belongs.
- **`docs/preconditions.md`** — the user-facing guide: two worked
  examples (both runnable), what each verdict means, and the **honest
  scope** stated plainly: input-side only (the solve's behaviour is a
  planned separate layer); interval judges array obligations elementwise
  while the SMT step is currently scalar-only (array obligations stay
  UNKNOWN with the reason quoted); the ℝ-vs-ieee semantics line is in
  the stamp. Every limit surfaces in the verdict itself — that is the
  trust story.
- **The portability self-audit:** the documented path walked as a
  stranger — imports exactly as the doc shows, templates + `check()`
  only, no internal knowledge — on the magnetics SPD and mass checks.
  **All three verdicts reproduce the recorded harness's results**
  (VERIFIED / UNKNOWN / REFUTED-with-witness-0). One gap surfaced and
  closed: the doc had no install line. No other undocumented step was
  needed.

## Gates, verified

- The three widening harnesses: query hashes identical, outputs
  identical modulo addresses and solver ms (above).
- The full recorded set re-run: statuses byte-identical.
- Suites: **803 passed** (venv-jax) / **680 passed + 10 skipped**
  (zero-dep) — +11 tests over the 792/674 baselines (6 vacuity
  zero-dep; 2 multi-value, 1 magnetics-shape, 2 `check()`), none
  removed. *(The first draft of this paragraph stated counts before
  running them — caught by re-reading against the measured output and
  corrected here; L15 applies to pass records too.)*
- README claims test green with the new capability bullet (the
  precondition templates have their witness in `src/stelling/preconditions.py`).

## Ledger

- **L12 gains the vacuity-machinery instance** (recorded in the ledger
  beside the emission-sweep witnesses): the instrument that polices
  every count was itself convention-copied at three sites with one
  drifted variant — now structural (single implementation; the mode
  names the registered procedure).

## What this greenlights

The precondition class is now **portable, documented, and
honest-scoped**: a stranger with the doc can pose the class on their own
JAX code and get stamped verdicts, including REFUTED-with-witness, up to
the stated array-emission boundary. That is the concrete value story for
the eventual MIME advertising / open-sourcing move. The likely next
build remains **array-aware emission** on its accumulating demand (two
sightings; the third decides), with LA/affine roadmapped as deepening.
