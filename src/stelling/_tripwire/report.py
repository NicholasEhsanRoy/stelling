# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Rendering. **No jax; testable in a bare interpreter.**

THE REPORT IS A WITNESS, NOT AN ASSERTION. The bar is the one a refuted
verdict's witness has to clear, and what makes that witness trustworthy is not
confident phrasing — it is that the verdict is *replayed before it is
reported* and withheld if the replay does not certify. The same discipline
applies here, for the same reason and one more: the tripwire is the first
thing many users will meet of this project, and one finding a user cannot
reproduce costs more trust than ten real ones earn.

So every finding printed below carries, in this order:

* **the user's own line, quoted**, with the innermost-writer rule stated, so
  the attribution can be confirmed or refuted at a glance;
* **both halves, observed** — what was written, what it became — neither
  inferred;
* **the arithmetic**, so a reader can check it with a pencil and not have to
  trust the instrument;
* **a runnable reproducer**, derived from the finding, not a template;
* **the independent recomputation**, done before printing. If it disagrees
  with what was observed, the disagreement is printed *instead of* the
  finding;
* **observation and inference labelled apart**. "The narrowing ran with
  operand 256 -> 0" is observed. "This may be a deliberate mask" is inference,
  and the one call site that fires across jax's own test suite is exactly
  that, so the inference is common and must not be dressed as a finding.

And two obligations that are absolute: the **denominator** is always printed,
and there is **never a clean bill of health**.
"""

from __future__ import annotations

from stelling._tripwire import record

HEADER = "stelling overflow tripwire"

#: The eager detector's own section header. A SEPARATE section and not extra
#: lines under the tripwire's, because the two instruments arm separately,
#: fail separately and are switched on separately -- a reader who ran only one
#: of them must not have to work out which half of one banner applies.
EAGER_HEADER = "stelling eager truncation detector"

#: How many frames of the call chain to print per finding. The chain is
#: already narrowed to the traced region; this bounds a deeply recursive one
#: without ever hiding the writer, which is the innermost frame and therefore
#: always in the window.
CHAIN_LIMIT = 8

#: What the instrument provably does not see. Printed on every run, findings
#: or not, because "N narrowings on traced code you executed" is true and "no
#: other problems" is the false-clearance error this project has already had
#: to withdraw twice.
#: EVERY ITEM WAS MEASURED WITH A LIVE CONTROL IN THE SAME PROCESS, on jax
#: 0.11.0 and 0.10.2 with x64 on and off, and the value still wraps in every
#: one. The list has never claimed to be complete and this run never claims a
#: clean bill of health — but a reader is invited to read it as the answer to
#: "what does it not see", so a door that is known and unnamed is a defect of
#: this tuple.
UNCOVERED = (
    "eager execution -- outside `jit`, the constant reaches XLA as an "
    "argument and is truncated there; the const-fold site is never reached "
    "(measured: 0 invocations, and the value still wraps)",
    "doors where the const-fold site is never reached AT ALL (0 "
    "invocations): `jnp.where(pred, N, x)`; `jnp.clip` at EITHER bound -- "
    "`jnp.clip(x, lo, N)` and `jnp.clip(x, N, None)`; "
    "`jnp.pad(x, k, constant_values=N)`. For `where` and `clip` the "
    "narrowing survives as a `convert_element_type` on a sub-jaxpr VARIABLE "
    "with the literal at the enclosing call site, so no constant is folded "
    "and nothing here can see it. THE VALUE STILL WRAPS.",
    # MEASURED on both series with x64 on and off, with a live control in the
    # same process. These matter more than the rest of this list because they
    # sit beside `x + N` and `x * N`, which ARE covered, so a reader scanning
    # for "operators are covered" would take the wrong answer away.
    "BINARY OPERATORS THAT ARE NOT COVERED, next to ones that are: `x % N`, "
    "`x // N` (and `jnp.remainder`, `jnp.floor_divide`, `divmod`), and "
    "`jnp.searchsorted(a, N)`. Measured on `int8` with N=300: `x % 300` on "
    "`[100, 50, 10]` gives `[12, 6, 10]` where the source says `[100, 50, "
    "10]`; `x // 300` gives `[2, 1, 0]` where the source says `[0, 0, 0]`; "
    "`searchsorted` gives 1 where the source says 3 -- all with 0 fires. Same "
    "mechanism as `where`/`clip`: the constant is an argument to an inner "
    "`jit` sub-jaxpr whose `convert_element_type` narrows a VARIABLE. `x % "
    "300` also logs 3 in-range visits into the denominator above. THE VALUE "
    "STILL WRAPS.",
    "doors where the value is ALREADY NARROWED BEFORE this site, so the rule "
    "receives something IN RANGE and does not fire -- AND THE VISIT IS "
    "COUNTED IN THE DENOMINATOR ABOVE, which is why a large denominator is "
    "not evidence of coverage: `jnp.full(shape, N, dt)`, "
    "`jnp.full_like(x, N)`, `lax.full(shape, N, dt)`, "
    "`lax.full_like(x, N)`, `lax.convert_element_type(N, dt)`, "
    "`lax.select(p, jnp.full(shape, N, dt), x)`, "
    "`jnp.stack([x, jnp.full(shape, N, dt)])` and anything else built on "
    "`full`, `jnp.take(x, i, mode='fill', fill_value=N)`, "
    "`np.asarray(N).astype(dt)` and every other value numpy narrows before "
    "it reaches jax at all, and an operand that was already an array (the "
    "rule declines to fold non-scalars). numpy "
    "truncates on the way in and the rule sees the wrapped value, so THE "
    "VALUE STILL WRAPS and nothing here can tell. THE SET IS ENUMERATED AND "
    "MEASURED, route by route, in "
    "`tests/test_tripwire_gate_coverage.py::GATE_COVERAGE` -- a door that "
    "moves bucket goes red there rather than going quiet here. "
    "THE OPT-IN EAGER DETECTOR CLOSES MOST OF THIS GROUP AND IS NOT ON BY "
    "DEFAULT: run with `--stelling-eager-truncation=error` and every route "
    "above whose constant is still a written integer when it reaches jax "
    "RAISES at the line that wrote it instead of narrowing. What it does NOT "
    "close is the part where numpy has already finished before jax is "
    "reached, and there are exactly two named routes into that residue: "
    "`np.asarray(N).astype(dt)`, which is PERMANENTLY unhookable -- "
    "`np.ndarray.astype` is an immutable type attribute, and numpy emits no "
    "warning for it even under `simplefilter(\'error\')` -- and "
    "`jnp.asarray(np.array(N), dtype=dt)`, a second spelling into the same "
    "residue, where numpy builds the array at its own default width and jax "
    "is handed a value that was destroyed before it arrived. Both measured "
    "on jax 0.11.0 and 0.10.2: 0 fires with the detector armed, and the "
    "value still wraps.",
    "anything inside a scoped `with jax.disable_jit():`, which swallows a "
    "door that is otherwise COVERED: `a + 200` on `int8` inside the block "
    "produces a jaxpr BYTE-IDENTICAL to the one that fires outside it and 0 "
    "fires, because the constant is narrowed before the site and the rule is "
    "handed -56 instead of 200. Process-wide `JAX_DISABLE_JIT=1` is a "
    "different case and is handled: `arm()` reports `not-invoked` and the "
    "tool disables itself rather than reporting a quiet zero. Inside the "
    "scoped block THE VALUE STILL WRAPS. "
    "THE OPT-IN EAGER DETECTOR CLOSES THIS DOOR, and it is the clearest "
    "case of why: the reason the rule is handed -56 is that the constant was "
    "narrowed at `lax._convert_element_type` on the way in, which is exactly "
    "where the eager detector sits. Measured on jax 0.11.0 and 0.10.2 with "
    "`--stelling-eager-truncation=error`: `a + 200` inside the block raises "
    "`EagerTruncationError` naming 200 -> -56 at the line that wrote it, "
    "inside a scoped `with jax.disable_jit():` and under a process-wide "
    "`JAX_DISABLE_JIT=1` alike. `CLOSES THIS ONE OUTRIGHT` is what this "
    "sentence used to say and it claimed more than was measured: with the "
    "detector armed, `jit` off is also the configuration in which jax "
    "evaluates ITS OWN constants eagerly -- the threefry PRNG mask, "
    "4294967295 -> -1 at int32 -- and the first version of the detector "
    "raised on those, in eight of this project's 33 census workloads. It "
    "does not now: `eager._origin` attributes a narrowing to whoever the "
    "data came from, and jax's own are counted and printed rather than "
    "raised on. What is NOT closed in this mode is the same residue as "
    "everywhere else (the numpy routes above), plus the origin filter's own "
    "lenient edge, which the eager section names.",
    "anything replayed from a WARM TRACE CACHE instead of being traced: "
    "jax's cache is keyed on the jitted callable and its avals, so a "
    "`@jax.jit` function any earlier trace already reached -- before this "
    "tripwire armed OR after it, in an earlier test -- is never traced again "
    "and the rule never runs over its body. `jax.jit(f, inline=True)` does "
    "this while leaving NO nested jaxpr behind to notice it by. THE GATE IN "
    "`preconditions.check()` AND `contracts.check_contract()` CLOSES THIS "
    "FOR JAX'S OWN CACHES: it calls `jax.clear_caches()` before the trace it "
    "watches, so a verdict's observation is complete WITH RESPECT TO JAX'S "
    "CACHES, IN A SINGLE-THREADED PROCESS (B15) -- and both halves of that "
    "qualifier are load-bearing, so the next two entries say what is outside "
    "it. This session report has "
    "no such moment -- it watches whatever your suite happens to trace -- so "
    "the door stays open for this report's findings and is closed only "
    "for the verdicts.",
    # RESTORED, and it is the broader claim the warm-cache entry above
    # narrowed: `jax.clear_caches()` empties JAX's caches and nothing else,
    # so a value narrowed earlier and held anywhere ELSE is still not
    # observed. This bullet is what the three MEASURED constructs below sit
    # under; dropping it and then asserting completeness over the difference
    # is the shape B15 exists to close, repeated one layer down.
    "ANYTHING NARROWED BEFORE THE TRACE THE GATE WATCHES -- a constant "
    "destroyed earlier in the process and handed to the trace already "
    "wrapped, so there is nothing left at trace time to fire on. The "
    "eviction above closes this only where the thing holding the narrowed "
    "value is JAX'S OWN cache. Where it is not, nothing re-traces it and "
    "nothing sees it. Three such constructs, measured on jax 0.11.0 with "
    "x64 on, each writing "
    "40000 into an int16 program: `jax.extend.core.jaxpr_as_fun(saved)`, "
    "where the narrowed -25536 is already inside the saved jaxpr; a user "
    "`functools.lru_cache` (or any memo) returning a value that was narrowed "
    "when the memo was filled; and `jax.closure_convert`, A PUBLIC JAX API, "
    "which traces at setup and hoists the already-narrowed constant into the "
    "consts it returns. All three: VERIFIED, 0 fires, and the program jax "
    "executes returns [-25536, -25436] where the program as written returns "
    "[40000, 40100]. THE VALUE STILL WRAPS.",
    # B16, and it is the honest half of a hole B15's audit found and this
    # batch closed. The gate now asks whether stelling's wrapper is still the
    # live registry entry; what it cannot ask is whether it was the live
    # entry for every instant in between.
    "anything traced while stelling's HOOK IS DISPLACED and put back inside "
    "one call. The trace gate now asks `_tripwire.displaced()` after the "
    "trace it watches, so a rebind that pre-dates the trace or is still in "
    "place at the end of it makes the verdict UNKNOWN and names the hook -- "
    "before B16 that case returned VERIFIED on a WATCHED route, because "
    "rebinding the registry entry leaves the recorder's identity and the "
    "fire count untouched while the wrapper is never called (measured, and "
    "byte-identical on the tree before this batch). A patch INSTALLED AND "
    "REMOVED inside that window is invisible to the check, exactly as a "
    "competing thread's re-warmed jit body is: the question asked is 'is "
    "our wrapper live now', and 'was it live throughout' is not answerable "
    "from here.",
    "anything traced while ANOTHER THREAD is also tracing: jax's trace cache "
    "is process-global and this gate's fire counter is per-thread, so the "
    "window between the eviction and the trace it protects is NOT ATOMIC -- "
    "another thread can re-warm a jit body inside it, and the fires that "
    "thread causes are counted on its own stack and not on the gate's. "
    "Measured on jax 0.11.0, one harness whose narrowing sits in a shared "
    "jitted helper: 0/400 wrong VERIFIED single-threaded, 247/400 (61.8%) "
    "with four threads calling that helper while the gate traced. THE RATE "
    "IS A FUNCTION OF HOW WIDE THE WINDOW IS -- how much the harness traces "
    "before it reaches the shared helper -- so it is a range and not a "
    "constant: with the same four threads, 1/100 when the helper is the "
    "first thing traced, 52/100 after 50 preceding primitives, 247/400 "
    "after 100, and 100/100 after 200. Before the eviction existed it was "
    "399/400 single-threaded, so this is a large improvement and not a "
    "guarantee. stelling makes no thread-safety claim anywhere; run gated "
    "checks on one thread.",
)


#: What the origin filter actually separates, said in the report's own words.
#: IT IS "NOT JAX", NOT "YOURS". A constant written inside a third-party
#: library in site-packages passes the filter and is a real narrowing at a real
#: site -- but calling it "your own code" and "you wrote 128" tells a reader to
#: go and look for something they did not write. Driven with a real module in a
#: venv's site-packages: the file and line were correct and checkable, and the
#: framing around them was wrong.
OUTSIDE_JAX = "outside jax"


def _rule_line() -> str:
    return (
        "attribution: the innermost frame OUTSIDE JAX inside the traced "
        "region -- your own code, or a library you called that is not jax. "
        "Not the entry point, and not jax's caller"
    )


def render_status(status) -> list[str]:
    """The lines that say whether the instrument is live.

    §4 REQUIRES THREE THIRDS OF EVERY MESSAGE: what happened, what it means,
    and what still works. This rendered ``status.detail`` as the middle third,
    and ``arm()`` leaves ``detail`` empty for every one of the failure codes —
    so the primary channel printed ``NOT ARMED [no-module] --`` with a dangling
    dash and no middle third at all. The middle third only ever reached the
    ``require`` ``UsageError`` and the canary, both of which read
    ``Status.explanation``. It is ``Status.meaning`` that is never empty, so it
    is what goes here; ``detail`` is the extra, printed when there is one.
    """
    lines = []
    if status.armed:
        detail = f"armed -- {status.detail}" if status.detail else "armed"
        lines.append(detail)
    else:
        lines.append(f"NOT ARMED [{status.code}] -- {status.meaning}")
        if status.detail:
            lines.append(f"    detail: {status.detail}")
        lines.append(
            "    Static checking is unaffected: `stelling.preconditions.check` "
            "and every verdict path work exactly as before. What is disabled "
            "is the runtime narrowing tripwire, and nothing else."
        )
    if status.rule_hash:
        # THREE STATES, not two. This read `== status.known_hash else
        # "CHANGED upstream"`, which was right while `known_hash` was one
        # constant and became wrong the day it became a lookup keyed on the
        # running release: a release with NO row now yields `None`, and
        # calling that "CHANGED upstream" would assert a comparison nobody
        # has made. `Status.hash_state` is the one place the case lives.
        state = status.hash_state
        if state == "as-tested":
            known = " (as tested)"
        elif state == "changed":
            known = (
                f" (CHANGED: jax {status.jax_version or '?'} is recorded as "
                f"{status.known_hash})"
            )
        else:
            # ``never-read``. ``unreadable`` is unreachable here: it means
            # ``rule_hash`` is falsey, and the ``if`` above already excluded
            # that — this branch is not a catch-all standing in for it.
            known = (
                f" (jax {status.jax_version or '?'} has NEVER BEEN READ: no "
                "row records a rule for this release)"
            )
        lines.append(
            f"    rule: {status.rule_name or '?'} sha1 {status.rule_hash}{known}"
        )
    return lines


def render_denominator(rec: record.Recorder) -> list[str]:
    """§8's first disclosure obligation. Never omitted, findings or not."""
    lines = [
        f"denominator: {rec.int_narrowings} integer const-folds inspected "
        f"({rec.folded} constants folded, {rec.invocations} rule invocations)."
    ]
    if rec.int_narrowings == 0 and rec.invocations > 0:
        lines.append(
            "    No integer const-fold was reached. A zero here is not a "
            "clean result; it means this run had nothing for the instrument "
            "to look at."
        )
    if rec.invocations == 0:
        lines.append(
            "    ZERO invocations. The hook did not run at all this session, "
            "so a zero finding count below is a fact about the instrument, "
            "not about your code."
        )
    if rec.unmodelled:
        lines.append(
            f"    {rec.unmodelled} folded constants were not integer-to-integer "
            "(float, bool, complex or a non-scalar) and were not range-checked."
        )
    if rec.internal_errors:
        lines.append(
            f"    {rec.internal_errors} internal errors inside the instrument "
            "were swallowed so they could not break your suite. The counts "
            "above are therefore a lower bound."
        )
    if rec.dropped_over_cap:
        lines.append(
            f"    TRUNCATED: {rec.dropped_over_cap} further distinct findings "
            f"were dropped after the cap of {rec.cap}. This report is a "
            "sample, not a total."
        )
    return lines


def render_uncovered() -> list[str]:
    lines = ["what this run did NOT look at (never a clean bill of health):"]
    lines.extend(f"  - {item}" for item in UNCOVERED)
    return lines


def render_suppressed(rec: record.Recorder) -> list[str]:
    """§10a.9: suppressed fires are named and counted, never silently dropped.

    PRINTED EVEN WHEN THE COUNT IS ZERO. A filter that only announces itself
    when it catches something is invisible on exactly the runs a reader would
    use to judge the instrument, and "no mention of a filter" and "no filter"
    then look the same. The line below says the filter is on and what it took,
    including nothing.
    """
    lines = [
        f"jax-internal filter: ON. {rec.suppressed_jax} narrowing(s) written "
        f"by jax itself and {rec.unattributed} with no attributable frame were "
        "kept out of the findings above."
    ]
    if not (rec.suppressed_jax or rec.unattributed or rec.suppressed):
        lines.append(
            "    Nothing was suppressed this run. The filter is stated anyway: "
            "a filter that only speaks up when it catches something is "
            "indistinguishable from no filter at all."
        )
        return lines
    lines.append("    Named, not dropped:")
    for finding in rec.sorted_suppressed():
        lines.append(
            f"  - {finding.written} ({finding.from_dtype}) -> {finding.became} "
            f"({finding.to_dtype}), origin {finding.origin}, "
            f"nearest frame {finding.file}:{finding.line} in {finding.func}"
            + (f", seen {finding.count}x" if finding.count > 1 else "")
        )
    lines.append(
        "    These are filtered out of the findings above because the "
        "constant was written inside jax rather than outside it -- jax's "
        "PRNG seed mask "
        "(0xFFFFFFFF -> -1) is the known instance and is deliberate. The "
        "filter is on by default; they are counted here so that a filter and "
        "a blind instrument do not look the same."
    )
    return lines


def reproducer(finding: record.Finding) -> list[str]:
    """Two lines a user can paste into a REPL, derived from the finding.

    THE PREDICTED OUTPUT IS THE POINT OF IT, and it was wrong every time: this
    said ``it prints 44:int8[]`` and a jaxpr prints ``44:i8[]``. Measured over
    the findings of a real session, 13 of 13 reproduced the value and 0 of 13
    printed the predicted text. A reproducer nothing runs is a template, and a
    prediction that is never right is worse than none — it is the one line
    here a reader can catch the instrument out on.

    ``tests/test_tripwire_arm.py`` now EXECUTES these lines and matches the
    comment against the real jaxpr, which is also what makes a wrong predicted
    value fail rather than survive.
    """
    return [
        "import jax, jax.numpy as jnp",
        f"print(jax.make_jaxpr(lambda a: a + {finding.written})"
        f"(jnp.zeros((), jnp.{finding.to_dtype})))",
        f"# the {finding.written} is not in the jaxpr: it prints "
        f"{finding.recomputed}:{record.jaxpr_dtype(finding.to_dtype)}[]",
    ]


def render_finding(index: int, finding: record.Finding) -> list[str]:
    """One witness. Requirements 1-6 of §10a, in order, labelled."""
    lines = [
        f"[{index}] {finding.file}:{finding.line} in {finding.func}"
        + (f"  (seen {finding.count}x)" if finding.count > 1 else "")
    ]
    text = finding.source
    if text:
        lines.append(f"    {finding.line} | {text}")
        if not finding.literal_visible:
            lines.append(
                f"    NOTE      the literal {finding.written} is not "
                "textually on that line. It may be a named constant, or "
                "computed, or come from an inlined caller -- check the chain "
                "below before assuming the line is wrong."
            )
    else:
        lines.append(
            "    (source line unavailable -- the file may have changed or be "
            "generated; the frame is still what the hook recorded)"
        )
    lines.append(f"    RULE      {_rule_line()}")

    if not finding.agrees:
        # §10a.5. The replay did not certify, so the finding is NOT reported.
        lines.append(
            f"    WITHHELD  observed {finding.written} ({finding.from_dtype}) "
            f"-> {finding.became} ({finding.to_dtype}), but recomputing the "
            f"narrowing independently from ({finding.written}, "
            f"{finding.to_dtype}) gives {finding.recomputed}. These disagree, "
            "so this is reported as a DISAGREEMENT and not as a finding: "
            "either this tool's arithmetic is wrong or jax's narrowing is not "
            "what it is modelled as. Please report it -- it is a bug in the "
            "instrument until shown otherwise."
        )
        return lines

    lines.append(
        f"    OBSERVED  the constant written there is {finding.written}; "
        f"{finding.to_dtype} holds that as {finding.became}. Both halves read "
        f"at the site: the rule received {finding.written} "
        f"({finding.from_dtype}) and returned {finding.became} "
        f"({finding.to_dtype})."
    )
    lines.append(f"    ARITHMETIC {record.arithmetic_sentence(finding.written, finding.to_dtype)}")
    lines.append(
        f"    CONFIRMED recomputed from ({finding.written}, "
        f"{finding.to_dtype}) without the hook: {finding.recomputed}. Agrees "
        "with what ran, so this is reported."
    )
    lines.append("    REPRODUCE")
    lines.extend(f"        {line}" for line in reproducer(finding))
    if len(finding.chain) > 1:
        lines.append(
            "    CHAIN     your frames inside the traced call, outermost first:"
        )
        shown = finding.chain[-CHAIN_LIMIT:]
        if len(finding.chain) > CHAIN_LIMIT:
            lines.append(
                f"        ... {len(finding.chain) - CHAIN_LIMIT} outer frame(s) "
                "not shown"
            )
        lines.extend(f"        {f}:{ln} in {fn}" for f, ln, fn in shown)
    lines.append("    INFERENCE (not observed -- these are suggestions, not claims)")
    lines.extend(f"        {line}" for line in _suggestions(finding))
    return lines


def _suggestions(finding: record.Finding) -> list[str]:
    """Labelled inference. Never a fix applied, never a fix asserted.

    The correct fix depends on intent that is not in the source: `q - 128` on
    `uint8` in a quantization routine is a true positive where "widen the
    dtype" and "this saturation is intended" are both defensible.
    """
    bounds = record.dtype_range(finding.to_dtype)
    span = f"{bounds[0]}..{bounds[1]}" if bounds else "?"
    return [
        f"- if {finding.written} was meant literally, {finding.to_dtype} "
        f"cannot hold it ({span}); a wider dtype is the fix.",
        (
            f"- if the wrap was intended -- masking is common and legitimate, "
            f"and jax's own PRNG does it -- write it explicitly, e.g. "
            f"`x & 0x{(1 << (record.INT_DTYPES[finding.to_dtype][1])) - 1:X}`, "
            "so the next reader does not have to guess."
            if finding.to_dtype.startswith("uint")
            # MEASURED, and this bullet used to say the same thing for signed
            # dtypes, where it is wrong twice over. An all-ones mask written as
            # a POSITIVE hex literal is itself out of range for a signed dtype:
            # on int8 `x & 0xFF` reaches the jaxpr as `and a -1:i8[]`, so it
            # would trip THIS check on the next run -- and `x & -1` is the
            # identity, so it masks nothing. int8/int16/int32 fire, uint8/uint16
            # do not. A tool whose suggested fix is an instance of the defect it
            # reports is not one to take advice from.
            else f"- if the wrap was intended, note that on a SIGNED dtype "
            f"there is no all-ones mask to write: the constant would itself be "
            f"out of range for {finding.to_dtype} and would trip this check, "
            f"and masking with every bit set is the identity in any case. "
            f"Record the intent in a comment, or narrow deliberately with "
            f"`.astype(jnp.{finding.to_dtype})`."
        ),
        f"- hoisting the constant to its own definition site turns this "
        f"silent wrap into an immediate error: `jnp.array({finding.written}, "
        f"jnp.{finding.to_dtype})` and `jnp.asarray(...)` raise OverflowError "
        "for a Python int (measured, both tested jax series, x64 on and off).",
        "- `jax.numpy_dtype_promotion('strict')` is a discipline, not a "
        "weaker form of this check. Measured over an 11-door grid, both "
        "tested jax series, x64 on and off, the same in all four cells: for "
        "a CONCRETE-dtype operand (np.int64(N), jnp.int32(N), np.bool_(True)) "
        "it raises TypePromotionError at the SIX doors that promote an operand "
        "against an array -- x + N, x >= N, x.at[i].set(N), jnp.where, "
        "jnp.clip, jnp.maximum -- and is silent at the FIVE construction "
        "doors -- jnp.array, jnp.asarray, jnp.int8, jnp.full, jnp.full_like "
        "-- where that same operand narrows without a word. It rejects the "
        "IN-RANGE np.int64(3) at exactly those same six, so it separates "
        "DTYPES and not values; and it rejects a Python int at none of the "
        "eleven, which is the spelling in front of you.",
    ]


def measured_anything(rec: record.Recorder) -> bool:
    """Whether this recorder holds anything at all worth printing.

    The discriminator between "not armed and nothing was measured" — a
    ``no-module`` session, where the status line IS the whole report — and
    "not armed and something was measured anyway", which is every partial:
    an xdist controller whose armed workers reported, or a single process
    whose hook was detached part-way through.
    """
    return bool(
        rec.invocations
        or rec.findings
        or rec.suppressed
        or rec.internal_errors
        or rec.dropped_over_cap
    )


#: §6's lost-worker disclosure, generalised. A partial is a partial whether the
#: missing part is a worker that died or a stretch of the session that ran
#: uninstrumented, and it gets the same words for the same reason.
PARTIAL_BANNER = (
    "    PARTIAL: the status above is not `armed`, and the figures below were "
    "collected anyway -- by an armed xdist worker, or before this session's "
    "hook stopped being the live one. They cover the INSTRUMENTED PART of "
    "this run and no more.",
    "    That is a PARTIAL and not a total, in the same sense as a lost xdist "
    "worker and for the same reason: presenting what arrived as a total "
    "would be a confident wrong answer. A zero below is a fact about the "
    "part that was measured.",
)


def render(status, rec: record.Recorder, notes: tuple[str, ...] = ()) -> list[str]:
    """The whole section, in a stable order. Two runs must agree byte for byte.

    A NON-ARMED STATUS USED TO RETURN HERE, and under xdist that threw away
    everything the armed workers had collected: one broken worker of two makes
    the controller's status ``mixed``, and every finding the other worker
    serialised back was dropped with no count and no mention. The status is
    the report's headline, not its gate — what is withheld from a partial is
    the CLAIM that it is a total, and that is what the banner withholds.
    """
    lines = [HEADER, "=" * len(HEADER)]
    lines.extend(render_status(status))
    lines.extend(notes)
    if not status.armed:
        if not measured_anything(rec):
            return lines
        lines.extend(PARTIAL_BANNER)

    lines.append("")
    lines.extend(render_denominator(rec))

    findings = rec.sorted_findings()
    disagreements = [f for f in findings if not f.agrees]
    lines.append("")
    if findings:
        lines.append(
            f"{len(findings)} distinct out-of-range integer narrowing(s) "
            f"written {OUTSIDE_JAX}"
            + (
                f", of which {len(disagreements)} are WITHHELD as "
                "disagreements (see below)"
                if disagreements
                else ""
            )
            + ":"
        )
        lines.append(
            f"({OUTSIDE_JAX}, which is not the same as BY YOU: the filter "
            "separates jax's own constants from everything else, so a "
            "constant written inside a library you called is reported at that "
            "library's file and line. The site is named for exactly that "
            "reason -- check it before assuming it is yours.)"
        )
        lines.append("")
        for index, finding in enumerate(findings, 1):
            lines.extend(render_finding(index, finding))
            lines.append("")
    else:
        lines.append(
            f"no out-of-range integer narrowings {OUTSIDE_JAX}, over the "
            "denominator above."
        )
        lines.append("")

    suppressed = render_suppressed(rec)
    if suppressed:
        lines.extend(suppressed)
        lines.append("")
    lines.extend(render_uncovered())
    lines.append(
        "  - arm order: the tripwire arms at pytest_configure. A function "
        "your conftest traced before that is cached and never re-traced, so "
        "it is invisible here. THIS REPORT never calls `jax.clear_caches()` "
        "-- it would change your suite's timing and behaviour to flatter a "
        "report. `preconditions.check()` DOES call it, once per gated trace, "
        "and so does `contracts.check_contract()`, which reaches the same "
        "gate -- because a VERDICT has to be able to say it watched the "
        "whole program and a report does not (B15); so if your suite calls "
        "either of them, its caches are being emptied and the sentence "
        "above about arm order does not describe what they see."
    )
    return lines


#: What the EAGER detector provably does not see. Printed in its own section
#: on every run it is armed for, findings or not, for the reason
#: :data:`UNCOVERED` is: "no undeclared truncation" is true and "your constants
#: are safe" is the false-clearance error this project has had to withdraw
#: twice. Every item is measured on jax 0.11.0 and 0.10.2.
EAGER_UNCOVERED = (
    "VALUES NUMPY HAS ALREADY DESTROYED before jax is reached. The detector "
    "sits inside jax, so a constant that never arrives as a written integer "
    "cannot be seen. Two named routes: `np.asarray(N).astype(dt)`, which is "
    "PERMANENTLY unhookable -- `np.ndarray.astype` is an immutable type "
    "attribute, so there is nothing to patch, and numpy emits no warning for "
    "it even under `warnings.simplefilter('error')` -- and "
    "`jnp.asarray(np.array(N), dtype=dt)`, a second spelling into the same "
    "residue. Measured: 0 fires with the detector armed, and the value still "
    "wraps.",
    "ARRAYS, as opposed to scalar constants. The detector fires on a written "
    "scalar integer and never on a non-scalar operand, because a whole array "
    "reaching `.astype` is a program CONVERTING DATA rather than an author "
    "writing a constant, and firing on it would make the rule unusable on any "
    "real program. A wide array of out-of-range values narrowed by an "
    "`.astype` is not reported here and is not claimed to be.",
    "VALUES THE PROGRAM COMPUTES. A `convert_element_type` over a jax "
    "`Array` or a tracer is a narrowing the program performs at RUN time on a "
    "value that depends on its inputs -- the `deferred` bucket in "
    "`tests/test_tripwire_gate_coverage.py::GATE_COVERAGE`. It is not a "
    "transcription loss and this instrument says nothing about it; "
    "`stelling.preconditions.check` does, by declining the form.",
    "THE INLINE DOOR, `x + 256` on an `int8` array, which is the OTHER "
    "instrument's: the constant survives into the trace and dies in jax's "
    "const-fold rule for `convert_element_type`, which is where "
    "`stelling._tripwire.arm()` watches. The two detectors are complementary "
    "and neither subsumes the other -- measured, with both armed, on the "
    "eight routes each claims.",
    "COUNTS TAKEN WHILE ANOTHER THREAD IS CONSTRUCTING. The RULE is "
    "thread-safe -- the decision is per call and the region stack is a "
    "context variable, so it is per-thread and per-asyncio-task and no "
    "truncation escapes because of a race -- but the two "
    "counters above are plain module integers and a concurrent increment can "
    "be lost. The denominator is therefore a floor under threads. stelling "
    "makes no thread-safety claim anywhere; this is the one place where the "
    "consequence is a number rather than a verdict.",
    "A REBIND PERFORMED AND UNDONE INSIDE ONE CALL. The displacement check "
    "asks whether stelling's wrapper is the live attribute at the end of a "
    "region; a patch installed and removed inside that region is invisible "
    "to it, exactly as it is to the trace gate.",
    "A CONSTANT THE ORIGIN FILTER CANNOT ESTABLISH AS YOURS. The detector "
    "raises only on a narrowing whose value crossed from your code into jax "
    "as an argument of the call you made -- that is what keeps it from "
    "raising inside jax's own PRNG under `jax.disable_jit()`, "
    "`JAX_DISABLE_JIT=1` and `chex.fake_jit()`. Its two edges are both "
    "disclosed and they lean opposite ways. A constant reaching jax inside a "
    "custom object rather than a plain value, tuple, list or dict is not "
    "found in that scan, so it is attributed to jax and does NOT raise: a "
    "missed narrowing, and the only one this rule can produce. And a "
    "constant jax wrote that happens to EQUAL something you passed at the "
    "same call IS attributed to you and raises -- measured, "
    "`jax.random.PRNGKey(4294967295)`, where jax's mask and your seed are "
    "the same integer.",
    "A NARROWING THAT HAPPENS WHILE A GENERATOR IS SUSPENDED INSIDE AN "
    "`expected_truncation` REGION. The region is dynamically scoped to one "
    "context's stack -- isolated across threads and across asyncio tasks, "
    "measured both ways -- but a plain generator shares its caller's "
    "context, so a region it entered and has not left is open in the code "
    "that resumed it. Nothing in Python fixes this; `intentional_wrap` at "
    "the site does not have it.",
)


def render_eager(status, snapshot) -> list[str]:
    """The eager detector's section, as lines. **No jax; primitives in, text out.**

    Takes the status and a :func:`eager.snapshot` dict rather than the module's
    live globals, because under xdist the numbers printed are a SUM of several
    processes' and none of them is this one's. Same discipline as
    :func:`render`: what is printed is what was carried back, not what happens
    to be in memory here.
    """
    if status is None and not snapshot:
        return []
    lines: list[str] = []
    lines.extend(render_status(status) if status is not None else ["status unknown"])
    snapshot = snapshot or {}
    conversions = snapshot.get("conversions", 0)
    truncations = snapshot.get("truncations", 0)
    declared = snapshot.get("declared") or {}
    permitted = snapshot.get("permitted") or {}

    # THE DENOMINATOR, FIRST AND ALWAYS. This detector's success case is that
    # nothing happened, and "0 truncations" is also what a hook that was never
    # called reports. The conversion count is what tells those apart.
    lines.append(
        f"    {conversions} scalar integer conversion(s) observed at jax's "
        f"construction site; {truncations} of them were out of range"
    )
    # PERMITTED IS THE ONLY THING THAT SUBTRACTS FROM THE NUMERATOR, and
    # `declared` is not: `intentional_wrap` returns a value that is already in
    # range, so a declaration never reaches the hook and never appears in
    # `truncations` at all. An earlier version wrote `not (declared or
    # permitted)` and therefore said nothing about a session that carried one
    # declaration and one raise.
    resets = snapshot.get("resets", 0)
    if resets:
        lines.append(
            f"    ...and these figures are PARTIAL: the counters were reset "
            f"{resets} time(s) during this session, so they cover only the "
            f"period since the last reset. Nothing in the shipped path resets "
            f"them; a suite that tests this detector does."
        )
    suppressed = snapshot.get("suppressed") or {}
    suppressed_jax = snapshot.get("suppressed_jax", 0)
    permitted_total = sum(row[0] for row in permitted.values())
    # THE NUMERATOR SPLITS THREE WAYS AND THE LINE BELOW IS THE REMAINDER.
    # `truncations` counts every out-of-range narrowing the hook saw; a
    # narrowing jax itself wrote is attributed away, one inside a region is
    # permitted, and what is left is what stopped the program. An earlier
    # version subtracted only the permitted ones, so on a session where the
    # origin filter did its job it announced a raise that never happened.
    if truncations > permitted_total + suppressed_jax:
        lines.append(
            f"    {truncations - permitted_total - suppressed_jax} of those "
            "was/were yours, neither declared nor inside an "
            "expected_truncation region, which means it RAISED: this session "
            "did not finish normally. (A truncation that neither raised, nor "
            "was permitted, nor was attributed to jax would be a defect in "
            "this instrument -- please report one if you see this line on a "
            "green run.)"
        )
    if suppressed_jax or suppressed:
        lines.append(
            f"    {suppressed_jax} of those was/were written BY JAX ITSELF "
            f"below your call, at {len(suppressed)} call site(s), and did not "
            "raise. jax narrows its own constants -- the threefry PRNG mask "
            "is 4294967295 -> -1 at int32 -- and a constant you did not write "
            "is not a constant you can declare, so it is attributed and "
            "counted here instead:"
        )
        for site, (count, text) in sorted(suppressed.items()):
            lines.append(f"      {site}  x{count}  {text}")
    inconclusive = snapshot.get("inconclusive", 0)
    if inconclusive:
        lines.append(
            f"    {inconclusive} of the origin decisions above could NOT be "
            "established: the scan of the boundary call's arguments ran out "
            "of budget, and those were attributed to you rather than to jax, "
            "because over-reporting is visible to you and a suppression is "
            "not. If one of them raised on a constant you did not write, that "
            "is the residue and it is worth reporting."
        )
    if declared:
        lines.append(
            f"    {sum(row[0] for row in declared.values())} wrap(s) DECLARED "
            f"with stelling.intentional_wrap, at {len(declared)} site(s):"
        )
        for site, (count, text) in sorted(declared.items()):
            lines.append(f"      {site}  x{count}  {text}")
    if permitted:
        lines.append(
            f"    {sum(row[0] for row in permitted.values())} truncation(s) "
            f"PERMITTED by an expected_truncation region, at "
            f"{len(permitted)} site(s). A region permits ANY truncation "
            f"inside it, so each is named here with the reason its author "
            f"gave:"
        )
        for site, (count, reason) in sorted(permitted.items()):
            lines.append(f"      {site}  x{count}  {reason}")
    internal = snapshot.get("internal_errors", 0)
    if internal:
        lines.append(
            f"    {internal} internal error(s) inside the hook were caught and "
            f"counted rather than raised into your program. Findings from this "
            f"run may be incomplete; please report this."
        )
    lines.append("    what this detector does NOT see:")
    lines.extend(f"      - {item}" for item in EAGER_UNCOVERED)
    return lines
