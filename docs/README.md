<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Documentation

**New here?** [Quickstart](quickstart.md) — install, one runnable
harness, a stamped verdict, in four files.

## Using stelling

| | |
|---|---|
| [Quickstart](quickstart.md) | the on-ramp: declare, assert, check, read the verdict |
| [The harness API](harness-api.md) | the import path and every primitive: `any_array`, `any_pytree`, `assert_`, `assume`, `nonvacuity`, `trace` |
| [Reading a verdict](reading-a-verdict.md) | the statuses, every stamp line, and the two vacuity instruments |
| [Checking the preconditions your solver assumes](preconditions.md) | the task guide: ready-made obligation templates, posing guidance, reading a CI verdict |
| [Choosing a solver backend](choosing-a-solver-backend.md) | z3, cvc5, or both: how obligations are routed by fragment, what each backend decided in a measured battery, and what installing only one costs |
| [Reproducing a witness](reproducing-a-witness.md) | emitting a runnable file that executes a REFUTED's witness through your own program, without importing stelling |
| [The overflow tripwire](overflow-tripwire.md) | one line in `conftest.py`: find the integer constants JAX silently narrows in the traced code you run, and read exactly which doors it does not watch |
| [Inductive step verification](inductive-step.md) | `check_inductive_step(body, state_bounds)`: prove a loop body preserves its invariant in one step — VERIFIED means it holds for all iterations |
| [Norms](norms.md) | twenty-six rules, each earned by a specific failure and each naming the instances that earned it — the reasoning behind what a contribution is asked for |

## Records

| | |
|---|---|
| [State of the project at 0.1.0](state-0.1.0.md) | what has been measured, and what has not |
| [Supported primitives](supported-primitives.md) | which jaxpr primitives have a registered transfer, generated from the live registry |
| [Verdict ledger](verdict-ledger.md) | every recorded verdict that moved, and why |
| [Gauge coverage](gauge-coverage.md) | which primitive faces are gauged, per face |

`proposed-*.md` are design proposals and studies, each in the state its own
header declares. The vocabulary is closed: `BUILT` or `APPLIED` (shipped, and
the header names the commit and a test that pins it), `PARTLY BUILT` (some of
the proposal landed and the page says which), `PROPOSED` / `PROPOSED, NOT
BUILT` / `PROPOSED, NOT APPLIED` (written, not built), and `MEASURED` (a study,
which may recommend building nothing). They are records, not user guides.

`tests/test_proposed_page_headers.py` is what makes that sentence checkable: it
requires every header to use a token from that set, every commit a status names
to be an ancestor of `HEAD`, every test a page names to exist, every shipped
status to name one — and this paragraph to mention every token the pages
actually carry. It cannot tell that a page saying `NOT BUILT` is wrong; five
did, for months, which is why the corrected direction is now load-bearing.

## Elsewhere in the repo

| | |
|---|---|
| [SOUNDNESS.md](../SOUNDNESS.md) | what a verdict is permitted to claim — the trust policy behind the stamp |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | normative rules: where things live and who may influence a verdict |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | how to work on stelling |
| [design/](../design/) | the evidence and history behind the decisions |
