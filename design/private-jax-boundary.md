# The private-jax boundary — what the two import rules protect, and the one exemption

**Status:** architecture decision, 2026-08-11. Written before the exemption
was implemented; the measurements below are what decided it. The feature that
forced the question is the overflow tripwire (`stelling._tripwire`), whose
plan is `PLAN-tripwire.md` in the sweeps tree.

## The two rules, in their own terms

They are stated together in `CONTRIBUTING.md` and enforced twice each — by
the `jax-import-hygiene` pre-commit hook and by `tests/test_import_hygiene.py`
— but they are not two halves of one rule. They protect different things and
one of them is much weaker than it reads.

### Rule 1 — jax may be imported only in `_jax_compat.py`

Stated by the founding commit `f4a7287` ("ir: jax-free IR mirror and the
single jax boundary") and by `_jax_compat.py`'s own docstring: *"When jax
churns — and `jax.extend` explicitly reserves the right to — the blast radius
is this file."*

`test_jax_imported_only_in_compat_module`'s docstring is unusually explicit
about how narrow the claim is, and it is the honest reading:

> No file outside `_jax_compat.py` spells `import jax` / `from jax`. **That is
> the whole claim, and it is a proxy for the churn boundary rather than the
> boundary itself.**

What it protects, and still needs protecting:

- **A grep-visible churn boundary.** A jax release that moves a symbol has
  one file to read first. `design/maintenance-treadmill.md` is the evidence
  that this is not hypothetical: the 0.10 → 0.11 bump merged `Jaxpr` and
  `ClosedJaxpr`, and **two of roughly two owned analyses broke**, one of them
  invisibly, for three weeks.
- **Enforceability without importing anything.** Both controls are static
  text scans, so they run in the zero-dep lane where jax is absent.

What it protects only **incidentally**, and what it explicitly does not:

- It is **not** the zero-dep guarantee. The file's own docstring records that
  `harness.py` carried no jax token and still died at import in a bare
  environment with the token scan green. That property is measured
  behaviourally, by the tests below `_NO_JAX_PRELUDE`, and is untouched here.
- It is **not** leak-proof, by its own admission: `from stelling._jax_compat
  import jax` and a module-scope `importlib.import_module("jax")` with a plain
  literal name both leave the scan green.
- Its exemption is by **base name**, not by path, in both controls
  (`--exclude=_jax_compat.py`; `path.name == "_jax_compat.py"`). The hook's
  comment already records this as measured — a planted
  `src/zzsub/_jax_compat.py` containing `import jax` was `Passed` in both. So
  rule 1's exempt set is "any file under `src/` with that name", by design and
  by agreement between the two controls. **This exemption does not touch that
  and does not widen it.**

### Rule 2 — `jax._src` is banned outright

`_jax_compat.py`: *"Only public and `jax.extend` surfaces are used. Private
jax modules are banned repo-wide."* `CONTRIBUTING.md`: *"Private jax modules
are banned everywhere."*

The scope is the tell. Rule 1 exempts the boundary module; **rule 2 exempts
nothing at all, including the boundary module**, and the test half of it
covers `tests/` as well as `src/` where the hook covers only `src/`. It is not
"jax-facing code goes in one place" — it is "no code here, anywhere, depends
on a jax surface that carries no stability contract".

Which makes it, in this order:

1. **Maintenance cost.** This is the primary and it is the same axis rule 1
   sits on, at its extreme. `jax.extend` reserves the right to churn and the
   treadmill measured what that costs; `jax._src` promises nothing at all.
2. **Soundness-adjacent, indirectly.** `ARCHITECTURE.md` Rule 3: anything able
   to influence a VERIFIED stays core-audited. Every verdict is derived from
   the IR that `_jax_compat.py` transcribes, so a private-surface dependency
   *in the transcription path* would put a silently-moving surface underneath
   a verdict. `design/maintenance-treadmill.md` names the exact hazard — the
   "third speed", a semantics change that produces the same jaxpr with a
   different meaning, which no instrument here reaches.
3. **Not packaging.** The zero-dep posture is carried by `_optional.require`
   and by the behavioural tests, not by this token scan. A `jax._src` mention
   costs a bare environment nothing until something imports it.

**The part that still needs protecting is (1) everywhere, and (2) on the
transfer path specifically.** A file that names `jax._src` and that no verdict
path imports is priced by (1) alone.

## The measurement that decided the shape

The tripwire needs
`jax._src.interpreters.partial_eval.const_fold_rules[convert_element_type_p]`.
Before relaxing anything, the question was whether the feature could be shaped
to respect the rule instead.

Driven on both tested series (`JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`, the two
shared venvs), asking each candidate module whether it exports
`const_fold_rules`:

| module | jax 0.10.2 | jax 0.11.0 |
|---|---|---|
| `jax.interpreters.partial_eval` | absent | absent |
| `jax.interpreters` | absent | absent |
| `jax.extend.core` | absent | absent |
| `jax.extend.core.primitives` | absent | absent |
| `jax.extend` | absent | absent |
| `jax.extend.linear_util` | absent | absent |
| `jax.core` | absent | absent |
| `jax._src.interpreters.partial_eval` | **present**, a 3-entry `dict` | **present**, a 3-entry `dict` |

`jax.interpreters.partial_eval` exists and imports cleanly on both series and
**does not re-export the registry** — so the obvious "public shim over the
private module" route is measured closed, not assumed closed. The three keys
are `convert_element_type`, `device_put`, `stage` on both series.

The primitive itself is public and is the same object
(`jax.extend.core.primitives.convert_element_type_p`, and it is a key of the
private registry on both series); only the registry keyed on it is private.

So: **there is no route to this hook that does not name a private jax module.**
The choice is between an exemption and dropping the feature, and dropping it
is the principal's call and is already made.

## Options, and why the others were worse

- **Relocate the adapter into `_jax_compat.py`.** Rejected on two counts, one
  of them measured. It would widen rule 2's exemption to *the transcription
  boundary itself* — the one file where a private-surface dependency sits
  underneath every verdict, which is precisely reading (2) above — so it is
  strictly worse on the axis the rule protects than exempting a leaf module no
  verdict path imports. And it contradicts the plan's own requirement that the
  adapter import cleanly with jax absent: `_jax_compat.py` calls
  `require("jax")` at module scope, and
  `tests/test_import_hygiene.py::test_the_lazy_call_sites_also_name_the_jax_extra`
  already pins that `import stelling._jax_compat` raises
  `OptionalDependencyError` in a jax-less environment.
- **A narrower rule distinguishing *importing* a private module from
  *depending on its internals*.** Rejected: that distinction is not
  statically checkable, so it could not be enforced by either control, and a
  rule that needs a judgement call per line is the duty-enforced rule
  `ARCHITECTURE.md` Rule 2 already refuses (L18). It also does not describe
  this case — locating a registry inside a private module *is* a dependency on
  jax internals.
- **Reach the module with a runtime-assembled name.** Available and rejected
  as dishonest: `test_jax_imported_only_in_compat_module`'s docstring already
  records name-splicing as a known evasion of the token scan. Using a
  documented hole to avoid amending a rule is the same defect as an exemption
  that quietly stops biting, wearing different clothes.
- **A second exempt path, pinned to exactly one file.** Taken.

## The mechanism, and why it is not the obvious one

The exempt file is `src/stelling/_tripwire/_adapter_jax.py`, and the exemption
is **rule 2 only**. The adapter stays subject to rule 1 with no exemption: it
obtains the private module through `importlib.import_module` with a plain
literal path — which is also what the fail-closed contract needs, since the
adapter must probe rather than import-and-die — so rule 1's exempt set stays
at exactly one name and an `import jax` added to the adapter later still
reddens the hook. That is the narrowest relaxation available: one rule, one
file.

**`--exclude` cannot express this, and that is measured rather than assumed.**
GNU grep 3.11, recursive, against a scratch tree carrying the same layout:

| exemption spelled as | the adapter | a decoy at another path with the same base name |
|---|---|---|
| `--exclude=src/stelling/_tripwire/_adapter_jax.py` | **still reported** — the exemption is dead | still reported |
| `--exclude=_adapter_jax.py` | exempt | **silently exempt too** |
| anchored filter on `^src/stelling/_tripwire/_adapter_jax\.py:` | exempt | **reported** |

The first row is the dangerous one in a different direction from usual: a
path-spelled `--exclude` reads exactly like a path pin and does nothing, so
the hook would redden on the adapter while the test — which compares
`path.relative_to(REPO)` — passes. Hook and test would disagree again, in the
same file whose comment records them disagreeing before.

So the hook filters grep's *output* on an anchored path prefix, the shape
`library-identifier-hygiene` in the same config already uses ("the second grep
anchors past `path:lineno:` so a marker in a file PATH cannot mask an unmarked
line"). Anchoring at `^` means file content cannot mask a hit, because grep's
output always begins with the path.

**The filter has a trap and the structure avoids it.** With no hits at all,
`printf "%s\n" "$hits"` emits one empty line and `grep -v` matches it, so a
naive post-filter reports "a violation survived" over a clean tree. Measured;
the fix is the same one `library-identifier-hygiene` uses — branch on grep's
own exit status *before* filtering, and keep 2 (error) distinguished from 1
(no match) on both greps.

## What this exemption does not relax

- Rule 1 is unchanged. `jax._src` being permitted in one file does not permit
  `import jax` there.
- Rule 2 still covers `tests/` in the test half, with no exemption: the
  tripwire's own tests reach the registry through the adapter's API, never by
  naming it.
- The exemption is one path, not a directory: another file under
  `src/stelling/_tripwire/` naming `jax._src` reddens both controls.
- Nothing outside the adapter may obtain the private module *object* from it.
  The adapter returns status codes and primitives, never a jax object — which
  is the plan's boundary rule for its own reasons (a serialisable xdist
  payload) and this rule's reason as well.
- `import stelling` still imports no jax. The tripwire package is private,
  is not imported by `stelling/__init__.py`, and the property is measured from
  `sys.modules` rather than read.
