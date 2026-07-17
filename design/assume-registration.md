# Inert `assume` semantics — a registration, filed before its cases run

**Status:** REGISTRATION, 2026-07-18. Re-filed from
`design/e2a-registration.md`'s withdrawn amendment 2, **content
unchanged** — the bookkeeping moved, not the substance. It fills a
silence: the E2a registration permits `assume` in harnesses and never
spoke to its propagation semantics. Filed before the two cases it will
bite have run.

## The facts

`assume` is **inert** in the MVP propagation: the constraint is dropped,
which is sound (propagation runs over a superset of the declared set) and
was, until this filing, silent — the dropped constraint hid inside a 100%
coverage line, because the counter measured primitive coverage, not
semantic fidelity. Two of the remaining twelve are implications and will
hit this (dfx#752: *nonlinear failure ⇒ step rejected*; bjx#969:
*non-finite proposal ⇒ step_size_max shrinks*).

## The registered semantics

- **The coverage line counts dropped constraints as their own category**
  (`inert`), named, outside the "known" fraction — a drop can never hide
  inside 100%.
- **A VERIFIED with dropped constraints stands** — more was proved than
  asked — with the drop disclosed in the stamp's coverage line and a
  rendered note.
- **An UNKNOWN with dropped constraints is recorded as `blocked (inert
  assume)`**: it counts 0 toward the mechanized number (conservative),
  and it **cannot be cited as a mechanization failure** — wherever an E2a
  count is reported, blocked cases are named in the same sentence. "The
  harness was wrong is a work item, not a band adjustment" stands; this
  defines which item the work is.
- The implication workaround (`¬P ∨ Q`) needs `not`/`or`, which are not
  in the census list — **they arrive by census when their targets run**,
  never by guessing.

## Why the content is trustworthy — an argument, not a rule

The test worth applying: *would this change have been made if it cut the
other way?* It already does. The inert counter fires on a
**VERIFIED**-with-drops — which proved more than asked, where disclosure
helps nobody — exactly as readily as on an UNKNOWN-with-drops, **by
construction rather than by intention**: the counter is wired to the
primitive, not to the verdict. The symmetry is recorded here as the
reason to trust the content, and deliberately not promoted into the
amendment rule, which stays strict.
