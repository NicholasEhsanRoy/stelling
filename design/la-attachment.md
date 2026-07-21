# The LA contract attached to the real solver — and the round-2 critical audit

**Status:** PASS RECORD, 2026-07-21. Completes `design/roadmap.md`
item 2's remaining line ("attachment to a real solver call site").
Follows `design/la-contract-build.md` (the layer itself, round-1
audit). MIME results here are **usefulness evidence only** — held out
from all E2a counting, per the standing constraint.

## The attachment

`corpus/supply/mime_lsq_conditioning.py` binds `stelling.contracts` to
the call site the probe named from the start: MIME's least-squares
gradient normal system (`grad_least_squares`,
mime-engine @ 7ce1efb4311b, `nodes/environment/fvm/operators.py`) — per
cell, `M = Σ d⊗d (+ boundary d_b⊗d_b only for patches PRESENT in
boundary_face_values) + reg·I`, then `M⁻¹` per cell. One template
extension was needed and built: **`conditioning_2x2_field`** — the
conditioning contract over a caller-built FAMILY of matrices
(`transform(theta)` returning `[..., 2, 2]` or an `(a, b, c)` triple),
with **symmetry POSED, never assumed** on the array path (two closed
obligations `m01 ≤ m10 ∧ m10 ≤ m01`; a triple return is the caller's
explicit symmetry assertion, documented). The committed
`conditioning_2x2` is untouched.

The attachment story is configuration, not mesh: **the real default is
the hazard.** `boundary_face_values=None` (the signature default) feeds
no boundary geometry into M; on the real 2x1 Cartesian mesh that leaves
the rank-1 stencil `M = d⊗d + 1e-30·I`, cond ≈ 2.5e29 — the probe's
Part 1 matrix, reproduced here from MIME's own mesh object. The same
mesh boundary-fed is healthy (per-cell `diag(0.3125, 0.5)`, cond 1.6).

| known answer | configuration | verdict (measured) |
|---|---|---|
| KA-A | boundary-starved = the real default | REFUTED by intervals alone, both cells named, no solver |
| KA-B | same mesh, all four patches fed | VERIFIED — conjuncts interval-decided, symmetry pair trivially solver-discharged; the L20 inert-vacuity line present verbatim |
| KA-C narrow | delta boxes [0.4, 0.6] through the real assembly | interval UNKNOWN with the straddle quoted → solver VERIFIED (QF_NRA) |
| KA-C wide | delta boxes [0.4, 1.6] | solver REFUTED; witness dx = 7/16, dy = 2.0 — a realizable geometry (the auditor CONSTRUCTED `make_cartesian_mesh_2d(2, 1, 0.875, 2.0)` and measured cond 8.35918… > 8 on the real mesh) |

## Fidelity, and what it can distinguish (L21's birthplace)

`segment_sum` traces to `scatter-add` — no transfer row; the honest
decline is quoted in the script. So the assembly is transcribed with
the segment sums unrolled over the real mesh's static connectivity —
which raises the question the round-2 audit made precise: *what ties
the transcription to the real function?* The layered gate stack, each
layer's power now measured by a twelve-mutation battery:

- **F** (transcribed pipeline vs the real `grad_least_squares` import,
  random fields, bit-identity) pins the float-visible
  assembly-to-output path — and measurably CANNOT see a 10× `reg`
  error (absorbed in binary64 when fed; nullspace-aligned when
  starved).
- **Exact M cross-checks in BOTH configurations** (the starved one
  added by the round-2 fix — it is the only gate that catches
  `reg_10x`).
- **The signature read + assert** pins `reg` by construction.
- **A second, non-congruent fidelity mesh (3x2)** breaks the 2x1's
  cell congruence (bit-identical there too; keeping bit-identity
  required accumulating one subtotal per segment_sum in the real
  code's association order — float addition is non-associative, and
  the KA query hashes were verified byte-identical across that
  refactor).
- **The residual class is named algebraically, member by member**:
  mutations that are value-identical on every Cartesian box
  (transpose of a symmetric M, owner/neighbour swap of a
  role-symmetric sum, sign flips squared away by d⊗d, the 180° cell
  permutation of a centro-symmetric constructor) are undetectable by
  value comparison — which is why the transcription also cites and
  mirrors the source line by line.

Banked as **L21**: a fidelity check pins only what it can distinguish;
its discriminating power is a thing you measure with deliberately
wrong variants, not a thing you assert. The script now says exactly
what pins what, and warns adopters: an absorbed parameter that is not
signature-readable must be pinned by construction.

## Round-2 audit (distinct fresh context, report-only)

Seven findings, **zero UNSOUND** — every solver-VERIFIED survived
independent re-derivation (an 801² dense sweep plus exact-rational
corners on KA-C narrow; every REFUTED witness re-proven in exact
arithmetic; every operators.py/mesh.py line citation checked against
source). Adjudications:

1. **F1 MISLEADING** — the fidelity claim overstated F's pinning power
   (the reg_10x blindness above). Fixed: the layered stack + honest
   wording + adoption warning.
2. **F2 SHARP-EDGE** — fidelity evidence was mesh-degenerate
   (congruent cells). Fixed: the 3x2 mesh + the named residual class.
3. **F3 SHARP-EDGE** — the family template's solver escalation stops
   at N=46→47 matrices (517 > 512 element terms; the committed,
   deliberate budget) and nothing said so at the adoption surface.
   Fixed by documentation: the measured boundary quoted in the
   template docstring and at KA-C's family choice, with the adoption
   pattern for real meshes (pose deliberate sub-families — a region, a
   sampled subset, a known-worst class; interval-decidable obligations
   work at any N). The budget itself stands.
4. **F4 NOTE** — the "jax-free" contract tests never actually ran in
   the nojax venv (module-level importorskip skips the whole file at
   collection). Fixed: 18 tests moved to `tests/test_contracts_nojax.py`
   (no skip); they now run in both venvs.
5. **F5 NOTE — the finding was against the ORCHESTRATOR, recorded as
   such:** the round-2 mandate expected an inert-vacuity line on KA-A;
   KA-A is REFUTED, and the committed instrument stamps only VERIFIED
   ("widening cannot rescue an UNKNOWN/REFUTED"). The auditor was
   right, the mandate was wrong, no code changed.
6. **F6 NOTE** — the triple path silently outer-broadcast mismatched
   shapes. Fixed: same-shape validation, scalars exempt (documented).
7. **F7 NOTE** — one unasserted mesh dtype; one missing WHY
   (declarations-vs-consts). Fixed; the raw-numpy dtype refusal at
   trace stays, consistent with committed template posture.

The orchestrator's gate re-ran the suites, the script (exit 0, 92
asserts), and the adapted mutation battery (`reg_10x` CAUGHT by
exactly the new gate; survivors exactly the named residual class).

## Seams measured on the way (recorded for the next attachment)

- `jax.ops.segment_sum` → `scatter-add`, no transfer row: obligation ⊤,
  escalation declines naming the primitive. `jnp.stack` → `stack`
  primitive, no transfer row (assembly uses concatenate-of-reshapes).
- Outward rounding turns `0.0·θ` into `[-5e-324, 5e-324]`: the posed
  symmetry pair straddles even on all-point envelopes, so a
  boundary-fed point VERIFIED needs the trivial solver discharge of
  symmetry. Documented and pinned.
- `make_cartesian_mesh_2d` defaults to float32; built with float64
  under x64, and the mesh arrays' actual dtypes are asserted in the
  script (`d`/`patch.d` float64; `owner`/`neighbour` int32 by the
  constructor's choice).

## Suite state at commit

venv-jax **953 passed**; venv-nojax **764 passed + 14 skipped** (the
18 moved jax-free tests now genuinely run there). All KA statuses and
query hashes byte-identical across the round-2 fixes. Statuses flipped:
none, anywhere, in either round.
