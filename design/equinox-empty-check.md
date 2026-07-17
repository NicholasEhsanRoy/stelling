# The equinox `jnp.empty` check — does the instance bite? Registered separately

**Status:** REGISTRATION, 2026-07-18. The drift probe found four live
`jnp.empty` call sites in equinox two days after jax 0.11.0 changed the
semantics from zeros to genuinely uninitialized, and correctly refused to
assess them (outside its registration). This is the new question: **does
this specific instance bite** — is any of the four sites read-before-write
under the new semantics?

**Method:** read the four sites in full (batch-norm EMA buffers and their
first-time flag foremost; the internal loop buffer; `eval_empty`), trace
whether uninitialized values can reach a computation before being
overwritten, and — if source reading indicates a bite — confirm
numerically on jax 0.11.0 with equinox, using a construction with
controls. Source verdict first; numeric confirmation second; no claim
travels on source reading alone.

**Outcomes:** bites (confirmed numerically) / does not bite (guard holds —
recorded as a fine result, no pushing) / cannot determine (recorded as
such).

**Rules, fixed:**
- **Verify before contacting anyone.** A wrong bug report against the
  ecosystem's substrate costs more than the finding is worth. Any upstream
  report is the maintainer's decision (Nick's), made outside this pass; this
  probe only establishes the fact to the standard the project applies to
  itself.
- If it does not bite, that is the result. No pushing.

Why it exists: it would be the first thing this project produced *for
someone else*. Secondary; it does not eat the supply probe.
