<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The affine v1 held-out evaluation — exhibit

Frozen record (ARCHITECTURE.md Rule 1: campaign exhibits stay pinned).
The affine refinement was built BLIND to this directory: a scout
measured the practical correlation-loss cases first (`SCOUT_CASES.md`,
written at stelling 3f78fdd, before the build), the builder and its
auditor never read them, and the evaluation ran only after the audited
build was gated. The full reading lives in
`design/affine-refinement.md`; the scripts here re-run it.

- `case1_mime_symmetry.py` / `case1_REFINED.py` — the real MIME socket
  symmetry pair (KA-B), baseline vs `refine="affine"`. Baseline:
  interval-UNKNOWN, VERIFIED only via 4 QF_NRA invocations. Refined,
  measured 2026-07-22: **VERIFIED with zero solver invocations**, both
  symmetry obligations `discharged by affine refinement`.
- `case2_conditioning_dependency.py` / `case2_REFINED.py` — the
  quadratic conditioning obligation. Refined, measured: affine
  tightened the slack to `[-1.46875, 29.03125]` and did NOT separate —
  the probe's on-file prediction ("quadratic past plain affine")
  confirmed; the solver still closes it.
- `probe_case4_refined.py` — the sweep hits: `select_n` outside
  AFFINE_SUPPORTED (4a/4b), strict-root decline (4c). The v2 frontier,
  named by measurement.
- `probe_case5_refined.py` — the MADDENING HeatNode flagship: both
  obligations decline (`the obligation slice is unavailable: primitive
  'scatter' is outside the supported emission set`). The largest
  measured prize (18/20 elements, pure correlation loss) waits on
  scatter routing in the slice layer.

Every refine-off re-run of the scout's five runners showed NO DRIFT
(case 2 differed only in solver milliseconds) — the opt-in default
changed nothing. If a future refinement extends the frontier, these
scripts are the first thing to re-run, and their declines are expected
to move — maintain the pins the way the MIME socket's seam pin was
maintained: measure first, then re-pin, recording the transition.
