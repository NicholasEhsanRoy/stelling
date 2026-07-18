# The ⊤-widening test — the third vacuity guard, registered before it runs

**Status:** REGISTRATION, 2026-07-18. Committed before the test executes
on any counted case.

Three vacuity guards were needed and two exist: **criterion (i)** guards
the box (does it contain `y0`? — mechanized, stamped), **fidelity** guards
the system (is it the incident's code? — censused, quoted). **Nothing
guards the obligation.** dfx#207 exposes the gap: the code is
`dt = jnp.maximum(dt, dtmin)` and the property is `dt ≥ dt_min` — which is
`max`'s *definition*, true for any input whatsoever. A proof of it uses no
system, no box, no region. The chain it completes: **defective property →
tautological obligation → VERIFIED → counts 1** — criterion (ii)'s
relation naming caught dfx#207's *gap* (property-recheck: defective
anchor) and not its *triviality*.

## The test — mechanical, the mutation control aimed at the inputs

Trace the harness as filed; then, **at the IR level**, rewrite every
`stelling_any` equation's bounds to `(−inf, +inf)` and re-propagate. If an
obligation still discharges, **the proof never used the declared bounds**
— it is structural, and the count says nothing about the incident's
region.

Scope notes, fixed now: judged on **obligations only** (`stelling_assert`);
nonvacuity rows are point declarations whose widening is meaningless and
they are excluded from the judgement. The IR rewrite touches top-level
`stelling_any` equations; the current harnesses declare all inputs at top
level (checked in the runner, loudly, so a nested declaration cannot
silently escape widening).

## The consequence — registered before anyone sees which cases are tautological

> **A tautological obligation cannot count 1.** A counted-1 case whose
> every obligation survives ⊤-widening is **voided, loudly** — the same
> enforcement shape as fidelity: if a count moves, the registration moved
> it. (A case with a *mix* of tautological and bounds-using obligations is
> reported with the split; it voids only if the discharge of the
> registered property rests entirely on tautological obligations.)

## Predictions, pre-committed before the run

- **dfx#207 → tautological** (`max(⊤, DT_MIN) ≥ DT_MIN` discharges by
  `max`'s definition). Its count of 1 voids.
- **npy#249 → NOT tautological**: under ⊤, `exp([−inf, inf]) = [0, inf]`
  and `> 0` straddles at the 0 endpoint — the declared `log_eps ≥ −20`
  floor is load-bearing. Its count stands.
- **dfx#417 → NOT tautological** (the face fluxes go ⊤ when `u, v` widen).
- **hit386 control → NOT tautological** (everything routes through the
  widened params).

If the predictions are wrong, the registered consequence still governs —
that is what pre-committing it is for.
