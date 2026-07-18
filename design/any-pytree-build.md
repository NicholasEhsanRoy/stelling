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

## What this pass does not do

No count (`any_pytree` posing a case is not a mechanized case — that
needs the full registered pipeline under the corpus-expansion
registration). No corpus expansion. No affine, no key representation
(the dependency wall stays behind its registered trigger). MADDENING
stays held out as the post-audit generalisation check, exactly as in the
census.
