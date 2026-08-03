<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Documentation

**New here?** [Quickstart](quickstart.md) — install, one runnable
harness, a stamped verdict, in three files.

## Using stelling

| | |
|---|---|
| [Quickstart](quickstart.md) | the on-ramp: declare, assert, check, read the verdict |
| [The harness API](harness-api.md) | the import path and every primitive: `any_array`, `any_pytree`, `assert_`, `assume`, `nonvacuity`, `trace` |
| [Reading a verdict](reading-a-verdict.md) | the statuses, every stamp line, and the two vacuity instruments |
| [Checking the preconditions your solver assumes](preconditions.md) | the task guide: ready-made obligation templates, posing guidance, reading a CI verdict |

## Records

| | |
|---|---|
| [State of the project at 0.1.0](state-0.1.0.md) | what has been measured, and what has not |
| [Verdict ledger](verdict-ledger.md) | every recorded verdict that moved, and why |
| [Gauge coverage](gauge-coverage.md) | which primitive faces are gauged, per face |

`proposed-*.md` are design proposals in the state their headers declare
(`BUILT`, or proposed and unbuilt); they are records, not user guides.

## Elsewhere in the repo

| | |
|---|---|
| [SOUNDNESS.md](../SOUNDNESS.md) | what a verdict is permitted to claim — the trust policy behind the stamp |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | normative rules: where things live and who may influence a verdict |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | how to work on stelling |
| [design/](../design/) | the evidence and history behind the decisions |
