# The any_pytree build — orchestration record

**Status:** orchestration record, 2026-07-18, written while the
fresh-context builder runs and before its output exists. The build was
ordered after `design/second-bill.md` fixed the number (mostly-trivial →
bounded) and `design/any-pytree-probe.md` established the target. The
main agent orchestrates and gates; it does not build.

## Scope, as ordered

`any_pytree` + the finite registry list reaches **array-state hits**
(diffrax/optimistix shape). It **cannot** touch the dependency wall —
keys (the 100% form) or normalizations (the partial form), one wall
behind the affine trigger. Building this advances one half of the corpus
and provably not the other; that is the correct scope, not a limitation
to engineer around.

## The gates, in order

1. **Carve-out (done, pre-build):** the two `convert_element_type` forms
   left the builder's list — whitelist widenings need witnessed
   real-trace tests through the audit gate, flagged as widenings
   (`design/second-bill.md`). The builder's registry scope explicitly
   excludes any `_t_convert` change.
2. **The builder** (fresh context, running): given the interval domain,
   IR, harness API, the two hand-declared worked examples
   (`corpus/supply/pytree_probe.py`), and the **closed list** — eight
   trivial rows (`abs`, `eq`, `ne`, `and`, `or`, `stop_gradient`,
   `reshape`, `pow` base-guarded/sound-libm) + three array-semantics
   items (scalar-selector `select_n`, `reduce_or` axis reduction, rank
   broadcasting for the elementwise ops). **Withheld:** the counts, the
   bands, the value model, design/, every other corpus file. Hard
   acceptance bar: `any_pytree` variants of both worked examples must
   produce **content-hash-equal** traces to the hand declarations — hash
   equality is the definition of faithful sugar. Leaf rules fixed in the
   spec: object-identity-aliased leaves declared once and reused; distinct
   leaves never merged; PRNG-key leaves refused at declaration with a
   message (authoring errors may raise; **analysis guards never do**).
3. **The guard rule** (verbatim in the spec): every guard degrades to ⊤
   with a quoted reason; no guard raises on a legal jax form — the
   guards-generate-hazards pattern's predicted fourth instance,
   pre-empted.
4. **The audit gate** (next, distinct fresh context): the built code +
   semantics, no counts. Demanded witnessed constructions: **leaf
   aliasing both directions** (shared object stays shared; distinct
   leaves never merge — the dependency problem *inside* the
   declaration); **the `_t_convert` whitelist byte-identical** (no
   widening crept in); **guard degradation** on a legal unhandled form;
   **hand-vs-sugar hash equality** re-verified independently; per-row
   soundness including the definite-FALSE directions.
5. **Adjudication** (main agent, with the history both subagents lack):
   findings become regression tests; a fix touching a soundness boundary
   gets its own witnessed construction before it is called done (the 4-B
   lesson: a fix is itself un-audited code).

## The build, returned (2026-07-18) — verified independently, adjudicated

The builder delivered against the closed list; verified from the gating
side before the audit: **190 tests green (jax venv), 135+5 (jax-free —
the builder recreated the destroyed venv), hooks pass (zero-dep and
jax-hygiene held), and the probe re-runs POSED with h_clean at 79% known
(3 ⊤: exactly the two carved-out convert forms + the mandated pow
decline) and h_hard at 64% (the key cone intact).** The acceptance bar
was met: **content-hash equality** between the `any_pytree` variants and
both hand-declared worked examples, from the builder's run (independent
re-verification is audit item 4).

**The builder's eight judgement flags, adjudicated:**

1. `_STRUCT_FMT` unsigned/small-int decoders — **accepted with audit
   coverage**: forced by the guard rule (the acceptance target crashed on
   an undecodable uint literal once `and`/`or` made the propagator read
   RNG mask constants); it is a literal-decoder change, not a transfer or
   whitelist change — but it touches the surface where audit finding 3
   lived, so the auditor is explicitly pointed at the new formats above
   2⁵³ (item 5).
2. Rank broadcasting in the shared `_pair_elements` (wider than item 11's
   letter) — **accepted**: uniform and sound in direction; audit item 6
   covers the index arithmetic.
3. The FRAGILE-5 regression test's vehicle swap (its old vehicle became a
   registered form) — **accepted**: the audited property is preserved on
   a new vehicle, with the swap recorded in-file.
4. `select_n` output shape now `cases[0].shape` — **accepted**
   (behavior-identical where previously supported; audit item 9 checks
   the new form against jax).
5. `any_pytree` API choices (TypeError for key refusal; `None` at static
   bound positions; alias bounds must agree) — **accepted** as within the
   spec's authoring-error latitude.
6. `_LIBM_ASSUMPTIONS` generic never-silent fallback — **accepted**.
7. Recreating the jax-free venv — **accepted** (it had been destroyed;
   both suites now run).
8. Docstring/scope-claim updates — **accepted**; this is the CONTRIBUTING
   convention, followed without being told it.

**Coverage findings reported by the builder (recorded, not built):** the
two carved-out `convert` forms now have observed instances (int64→float64
weak/strong in `adapt_step_size`'s promotion paths — the whitelist
correctly declines them; they remain widening candidates needing
witnesses); `pow` with base reaching ≤ 0 (the mandated decline);
`integer_pow` (a distinct primitive); bitwise integer `or` (threefry
plumbing — would need a bitset/congruence domain); and the h_hard
residual registry list (`reduce_sum`, `sqrt`, `split`, `log`, …) — all
census-recorded for any future round, none built.

## What this pass does not do

No count (`any_pytree` posing a case is not a mechanized case — that
needs the full registered pipeline under the corpus-expansion
registration). No corpus expansion. No affine, no key representation
(the dependency wall stays behind its registered trigger). MADDENING
stays held out as the post-audit generalisation check, exactly as in the
census.
