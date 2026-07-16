# Soundness

stelling is a verifier: its output is trusted, so its defects are not
ordinary bugs. This file is the project's public account of that trust.

## Policy

**SemVer governs the API. It does not govern verdicts.**

Any change that flips any verdict on any query — verified ↔ falsified ↔
unknown/vacuous, for any harness — is a **soundness event**, regardless of
whether the release is a patch, minor, or major bump. Every soundness event
gets an entry in the log below stating:

- what the defect or behavior change was,
- which stelling versions are affected,
- which prior verdicts are retroactively invalid, characterized as precisely
  as we can,
- what to re-run to re-establish trust.

Silent fixes are forbidden. A soundness fix that ships without a log entry
is itself a soundness event.

## What every verdict must carry

Every verdict object stamps, at minimum:

- stelling version,
- jax version used to trace the harness,
- solver name and version, **and transport** — Python wheel vs. external
  binary path; for an external cvc5, its `--show-config` feature set,
- **the exact solver options used**,
- **the precision configuration the verdict assumes, and the device class
  of any concrete execution it relies on** (counterexample replay,
  differential runs, fuzz crosschecks). Verified on jax 0.10.2:
  `jax_default_matmul_precision` is unset by default — the platform
  chooses — and a `dot_general`'s `precision` param travels in the jaxpr
  as `None`, a *request* resolved per device. XLA offers no default
  precision contract, so a jaxpr's f32 matmul is not a determinate
  computation until the device is known: **one jaxpr, three devices, three
  numerics** — precision configuration is part of what a verdict claims,
  exactly as solver options are,
- the query's content hash (`stelling.ir.ClosedJaxpr.content_hash()`; the
  hash covers semantic content and excludes source locations, so identical
  programs traced from different files share verdicts),
- once the transfer registry exists: the assumption tier of every transfer
  function involved (design commitment 5).

Solver options are not cosmetic. Observed on cvc5 1.3.4 (PyPI wheel): a goal
containing `exp` solves **unsat** under default options — cvc5 quietly routes
transcendental goals away from the coverings solver — while the same goal
with `nl-cov=true` forced hard-errors ("Term of kind exp is not compatible
with using the coverings-based solver"), and with `nl-cov=false nl-ext=full`
it solves again. **Three configs, three engines, one version string.**
"cvc5 1.3.4 said unsat" is not a reproducible claim; "cvc5 1.3.4, wheel,
options {…}, query `<hash>` said unsat" is.

Two further commitments bind every implementation of the stamp:

- **Never invoke a solver on defaults.** stelling always emits the complete
  option set explicitly — including options whose emitted value currently
  coincides with the solver's default — and the stamp records the emitted
  set. A verdict from "cvc5 1.3.4, defaults" is not reproducible across a
  solver upgrade even with a stamp: the stamp would record what was asked
  for, and a default is precisely what wasn't.
- **Cache the proof, not the report.** The content hash excludes source
  locations, so a cache hit can legitimately come from the same program
  traced in a different file. A rendered verdict must re-derive its
  file/line pointers from the *current* jaxpr's `source_info`; location
  data is never stored in or restored from the cache. The first violation
  of this reports a line number from someone else's file.

## Log

*(empty — no releases yet)*
