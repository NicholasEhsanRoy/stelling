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

---

# Reading (2026-07-18)

**Source verdict.** The training path is guarded: the EMA blend touches
the uninitialized buffers, but `lax.select(first_time, batch_stats, …)`
discards the blend on the first call, and `select` masks untaken values
(probed earlier, P2). **The inference path is unguarded**:
`mean, var = state.get(ema_state_index)` feeds `_norm` directly, so
**inference before the first training step reads uninitialized memory into
the output** — a legitimate sequence (evaluating a freshly initialized
model). Under jax ≤ 0.10 this returned the deterministic
`x / sqrt(eps)`; under 0.11.0 it returns whatever the allocator held.
(The batch-mode buffers are not affected — that branch's zero-debias math
requires and uses zero-initialized accumulators.)

**Numeric check (jax 0.11.0, equinox 0.13.8).** The buffers came back
**all-zero in practice** — fresh zero pages — and the
inference-before-training output reproduced the old behaviour
(`316.22778 = 1/sqrt(1e-5)`). The bite is **latent**: the semantics no
longer guarantee what the memory happened to provide.

**Verdict, per the registered outcomes: cannot-determine numerically;
bites-in-principle on a legitimate path.** The claim that travels is the
source-level fact only: an unguarded read of `jnp.empty` buffers on the
inference-before-training path, one line to harden (zeros-init or guard
the inference read). Whether to report it upstream is the maintainer's
decision, outside this pass; no contact was made. If it is reported, it is
the first artifact this project produced for someone else, and it is
exactly one sentence of claim with a two-command reproduction.
