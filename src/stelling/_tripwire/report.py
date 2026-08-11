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

#: What the instrument provably does not see. Printed on every run, findings
#: or not, because "N narrowings on traced code you executed" is true and "no
#: other problems" is the false-clearance error this project has already had
#: to withdraw twice.
UNCOVERED = (
    "eager execution -- outside `jit`, the constant reaches XLA as an "
    "argument and is truncated there; the const-fold site is never reached "
    "(measured: 0 invocations, and the value still wraps)",
    "`jnp.where(pred, N, x)` and `jnp.clip(x, lo, N)` -- measured 0 "
    "invocations on both tested jax series, traced and jitted alike. The "
    "narrowing survives as a `convert_element_type` on a sub-jaxpr VARIABLE "
    "with the literal at the enclosing call site, so no constant is folded "
    "and nothing here can see it. The value still wraps.",
    "anything traced BEFORE the tripwire was armed -- jit caches mean a "
    "function traced earlier in the process is never re-traced",
    "narrowings whose operand was already an array (measured: the rule "
    "declines to fold non-scalars, so the wrap has happened before this site)",
)


def _rule_line() -> str:
    return (
        "attribution: the innermost frame of YOUR code inside the traced "
        "region -- not the entry point, and not jax's caller"
    )


def render_status(status) -> list[str]:
    """The one-or-two lines that say whether the instrument is live."""
    lines = []
    if status.armed:
        detail = f"armed -- {status.detail}" if status.detail else "armed"
        lines.append(detail)
    else:
        lines.append(f"NOT ARMED [{status.code}] -- {status.detail}")
        lines.append(
            "    Static checking is unaffected: `stelling.preconditions.check` "
            "and every verdict path work exactly as before. What is disabled "
            "is the runtime narrowing tripwire, and nothing else."
        )
    if status.rule_hash:
        known = " (as tested)" if status.rule_hash == status.known_hash else " (CHANGED upstream)"
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
    """§10a.9: suppressed fires are named and counted, never silently dropped."""
    if not (rec.suppressed_jax or rec.unattributed or rec.suppressed):
        return []
    lines = [
        f"suppressed: {rec.suppressed_jax} narrowing(s) written by jax itself "
        f"and {rec.unattributed} with no attributable frame. Named, not dropped:"
    ]
    for finding in rec.sorted_suppressed():
        lines.append(
            f"  - {finding.written} ({finding.from_dtype}) -> {finding.became} "
            f"({finding.to_dtype}), origin {finding.origin}, "
            f"nearest frame {finding.file}:{finding.line} in {finding.func}"
            + (f", seen {finding.count}x" if finding.count > 1 else "")
        )
    lines.append(
        "    These are filtered out of the findings above because the "
        "constant was written inside jax, not by you -- jax's PRNG seed mask "
        "(0xFFFFFFFF -> -1) is the known instance and is deliberate. The "
        "filter is on by default; they are counted here so that a filter and "
        "a blind instrument do not look the same."
    )
    return lines


def reproducer(finding: record.Finding) -> list[str]:
    """Two lines a user can paste into a REPL, derived from the finding."""
    expected = finding.recomputed
    return [
        "import jax, jax.numpy as jnp",
        f"print(jax.make_jaxpr(lambda a: a + {finding.written})"
        f"(jnp.zeros((), jnp.{finding.to_dtype})))",
        f"# the {finding.written} is not in the jaxpr: it prints "
        f"{expected}:{finding.to_dtype}[]",
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
        f"    OBSERVED  you wrote {finding.written}; {finding.to_dtype} holds "
        f"that as {finding.became}. Both halves read at the site: the rule "
        f"received {finding.written} ({finding.from_dtype}) and returned "
        f"{finding.became} ({finding.to_dtype})."
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
        lines.append("    CHAIN     your frames, outermost first:")
        lines.extend(f"        {f}:{ln} in {fn}" for f, ln, fn in finding.chain)
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
        f"- if the wrap was intended -- masking is common and legitimate, and "
        f"jax's own PRNG does it -- write it explicitly, e.g. "
        f"`x & 0x{(1 << (record.INT_DTYPES[finding.to_dtype][1])) - 1:X}`, "
        "so the next reader does not have to guess.",
        f"- hoisting the constant to its own definition site turns this "
        f"silent wrap into an immediate error: `jnp.array({finding.written}, "
        f"jnp.{finding.to_dtype})` raises OverflowError for a Python int "
        "(measured on both tested jax series).",
        "- `jax.numpy_dtype_promotion('strict')` makes several of these "
        "spellings raise a TypePromotionError. It fires for in-range values "
        "too, so it is a discipline to adopt, not a wrap detector.",
    ]


def render(status, rec: record.Recorder, notes: tuple[str, ...] = ()) -> list[str]:
    """The whole section, in a stable order. Two runs must agree byte for byte."""
    lines = [HEADER, "=" * len(HEADER)]
    lines.extend(render_status(status))
    lines.extend(notes)
    if not status.armed:
        return lines

    lines.append("")
    lines.extend(render_denominator(rec))

    findings = rec.sorted_findings()
    disagreements = [f for f in findings if not f.agrees]
    lines.append("")
    if findings:
        lines.append(
            f"{len(findings)} distinct out-of-range integer narrowing(s) in "
            "your own code"
            + (
                f", of which {len(disagreements)} are WITHHELD as "
                "disagreements (see below)"
                if disagreements
                else ""
            )
            + ":"
        )
        lines.append("")
        for index, finding in enumerate(findings, 1):
            lines.extend(render_finding(index, finding))
            lines.append("")
    else:
        lines.append(
            "no out-of-range integer narrowings in your own code, over the "
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
        "it is invisible here. `jax.clear_caches()` is NOT called -- that "
        "would change your suite's timing and behaviour to flatter a report."
    )
    return lines
