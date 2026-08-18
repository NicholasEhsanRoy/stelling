# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Arm the SHIPPED tripwire, print what it attached to, and say whether it worked.

Used by `.github/workflows/nightly-jax-canary.yml` for both legs — the nightly
and the pinned control — so the two are the same probe by construction and a
red nightly beside a green control means upstream, not us.

IT CALLS `stelling._tripwire.arm()`. That is the entire design constraint: it
is the same call `pytest_configure` makes, so a canary that goes green is a
statement about the code that ships rather than about a re-implementation of
it that drifted. Everything else here is printing.

EXIT CODES, ALL OF THEM — because this paragraph used to name two of them
while the script had three. It said *"0 armed; 1 with `--require` and not
armed; without `--require` it reports and exits 0"*, and the live-control
check below returned 1 with no regard for `--require` at all. There are now
five ways out and every one of them is here:

  1  `--require` was passed and the tripwire could not arm.
  1  the tripwire armed and its LIVE CONTROL DID NOT FIRE, `--require` or
     not: `arm()` says the hook is attached and the control says nothing
     reached it, so every figure below it is unverified.
  1  the tripwire armed and its LIVE CONTROL RAISED, `--require` or not.
     A control that could not COMPLETE is not a control that passed: the
     instrument's own state is broken, so nothing below it is verified.
     This used to exit 0 -- the check above tested for the substring
     "DID NOT FIRE", which a raised control does not contain, so the one
     state in which the probe never ran was the one state that reported
     success. A broken instrument must page.
  1  the rule hash CONTRADICTS the row recorded for this exact release,
     `--require` or not. See `_hash_row` for why that one is fatal and
     "this release has never been read" is not.
  0  anything else. That includes NOT ARMED without `--require` — the shape
     a human wants when running this by hand to see what a given jax does —
     and a release with NO ROW in the version -> hash map, which is loud on
     stdout and in the step summary and still exits 0.
"""

from __future__ import annotations

import argparse
import os
import sys


def _control_verdict(state: str, rendered: str) -> tuple[str | None, bool]:
    """``(the sentence to print on stderr, is this fatal)`` for the live control.

    FOUR OUTCOMES, NOT TWO, AND THEY ARE KEYED ON A STATE. The predecessor
    asked ``"DID NOT FIRE" in control`` -- a substring test against a line
    built for a human to read -- which is the same shape of instrument this
    repository keeps having to withdraw: the rendered message is not the
    decision, and re-parsing it makes the decision depend on wording. The
    caller records what happened; this reads the record.

    ``fired``
        the probe ran and the hook saw it. The only clean state.
    ``did-not-fire``
        the probe RAN and the hook saw nothing. ``arm()` says the hook is
        attached; the control says nothing reached it. Fatal.
    ``raised``
        the probe did NOT run. Different finding, different remedy, so it
        gets its own sentence rather than being folded into the one above:
        `did-not-fire` means a dead hook, `raised` means an environment in
        which the probe could not execute at all. Fatal, and it did not use
        to be -- see the exit-code list at the top of this file.
    ``not-run``
        the tripwire did not arm, so there was nothing to control. Not
        fatal HERE; whether not-arming is fatal is ``--require``'s question
        and is answered before this one is asked.

    THE UNKNOWN STATE IS FATAL, deliberately. A state this function does not
    recognise means the caller grew an outcome nobody taught this decision
    about, and the whole point of the change that added this function is
    that an instrument whose own state is broken must page rather than pass.
    """
    if state == "fired" or state == "not-run":
        return None, False
    if state == "did-not-fire":
        return (
            "the tripwire armed and its live control did not fire. `arm()` "
            "says the hook is attached; the control says nothing reached "
            "it. Treat the armed status as unverified.",
            True,
        )
    if state == "raised":
        return (
            f"the tripwire armed and its live control RAISED -- {rendered}. "
            "A control that could not COMPLETE is not a control that "
            "passed: the probe never ran, so nothing on this page is "
            "verified. This is a DIFFERENT finding from `did not fire`, "
            "which means the probe ran and the hook was dead. Read the "
            "exception, then compare the two legs: if the pinned `control` "
            "job is GREEN, the nightly's jax broke the probe and this is "
            "upstream; if it is RED TOO, it is this repository.",
            True,
        )
    return (
        f"the live control reported a state this script does not recognise "
        f"({state!r}). That is a bug in this script, not a measurement, and "
        "it pages rather than passing because an instrument that cannot say "
        "what happened has not said that nothing happened.",
        True,
    )


def _hash_row(status) -> tuple[str, bool]:
    """``(what to print beside the sha1, is this fatal)``.

    THE MAP APPLIES HERE TOO, and it has three states, not two. This script
    used to compare against one constant and print `CHANGED from …` for
    anything that was not it — which, once the constant became
    `_KNOWN_HASHES` keyed on the running release, would have called a
    release nobody has ever read "changed".

    The three, and the exit code each gets:

    * **the row matches** — quiet, exit 0. Nothing to say.
    * **no row for this release** — LOUD, exit 0. Nobody has read the rule
      on this jax yet. It is the state a jax NIGHTLY is in by construction —
      a dev build can never be given a row — and it is the state the
      `control` leg enters the day jax ships a release, since that leg
      installs `.[jax]` and therefore resolves to whatever is newest. Paging
      on it would redden this workflow on every jax release and every night
      it runs against a nightly, which is how an alarm stops being read.
      `PLAN-tripwire.md` §5 is about arming and does not decide this; the
      canary's own Property 3 — infrastructure must not page — is the closer
      analogy, and this is the same shape: a true statement that no one can
      act on tonight. The remedy is a human reading the rule and adding a
      row, which a red build does not speed up.
    * **the row exists and disagrees** — LOUD, exit 1, `--require` or not.
      This is the one that cannot be explained by upstream moving: a
      released wheel is immutable, so the row is wrong, or the environment
      is not the jax it claims. Either makes every other line here
      unverified, which is exactly the standing the live-control check
      already exits 1 for.
    """
    state = status.hash_state
    if state == "as-tested":
        return "as tested", False
    if state == "never-read":
        return (
            f"jax {status.jax_version or '?'} HAS NEVER BEEN READ — no row "
            "in _adapter_jax._KNOWN_HASHES records a rule for this release. "
            "Read the rule, diff it against the nearest row, and add an "
            "entry naming what changed. Not a failure: this is what a "
            "nightly, and any jax released since the last row was written, "
            "looks like",
            False,
        )
    if state == "changed":
        return (
            f"CONTRADICTS the row for jax {status.jax_version or '?'}, which "
            f"records {status.known_hash} — read the rule before trusting "
            "anything on this page",
            True,
        )
    return "not read", False  # `unreadable`: the source could not be read at all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="store_true",
        help="exit non-zero if the tripwire could not arm (what the canary uses)",
    )
    args = parser.parse_args()

    from stelling import _tripwire
    from stelling._tripwire import _adapter_jax as adapter

    status, recorder = _tripwire.arm()

    # A LIVE POSITIVE CONTROL, in this process, after arming. Without it the
    # canary's own figures are the beautiful zero this project keeps finding:
    # `arm()` deliberately leaves the recorder empty (its self-check must not
    # inflate a user's denominator), so a canary that printed the recorder
    # straight after arming would print `0 invocations` on a perfectly live
    # hook and on a dead one alike.
    #
    # `_probe.over` and `_jax_compat` rather than an `import jax`: the shipped
    # probe is the program whose narrowing is already known, and this script
    # keeps the same jax boundary the package does.
    control_state = "not-run"
    control = "not run"
    if status.armed:
        try:
            from stelling._jax_compat import jax as _jax
            from stelling._jax_compat import jnp as _jnp
            from stelling._tripwire import _probe

            _jax.make_jaxpr(_probe.over)(_jnp.zeros((7,), _jnp.int8))
            found = recorder.sorted_findings()
            control_state = "fired" if found else "did-not-fire"
            control = (
                f"{len(found)} finding over {recorder.int_narrowings} "
                f"narrowing(s)"
                + (
                    f"; {found[0].written} -> {found[0].became} "
                    f"({found[0].to_dtype})"
                    if found
                    else " -- THE CONTROL DID NOT FIRE"
                )
            )
        except Exception as exc:  # noqa: BLE001
            control_state = "raised"
            control = f"raised {type(exc).__name__}: {exc}"

    disarmed = _tripwire.disarm()

    hash_note, hash_fatal = _hash_row(status)
    control_note, control_fatal = _control_verdict(control_state, control)

    rows = [
        ("status", status.code),
        ("jax", status.jax_version or "?"),
        ("rule", status.rule_name or "?"),
        ("rule sha1", f"{status.rule_hash or '?'} ({hash_note})"),
        ("registry size", str(status.registry_size)),
        ("locate()", adapter.locate()),
        ("live control", control),
        ("disarm()", disarmed),
        ("detail", status.explanation),
    ]

    for name, value in rows:
        print(f"{name}: {value}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        verdict = "ARMED" if status.armed else f"NOT ARMED [{status.code}]"
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### nightly jax canary: {verdict}\n\n")
            handle.write("| what | value |\n|---|---|\n")
            for name, value in rows:
                handle.write(f"| {name} | {value} |\n")
            handle.write(
                "\nARMING is never gated on the rule sha1: a cosmetic edit "
                "upstream must not disable the tool, and it does not — the "
                "tool armed or did not arm above without consulting the row. "
                "THIS SCRIPT'S EXIT CODE is a different question and does "
                "read it: a release contradicting its own recorded row exits "
                "1, and a release with no row at all exits 0 and says so. "
                "The sha1 is here so a red run is diagnosable from this page "
                "without re-running it.\n"
            )

    if args.require and not status.armed:
        print(
            f"\ncanary: the tripwire could not arm [{status.code}].\n"
            "If the control leg on the pinned series is GREEN, this is "
            "upstream: jax moved the const-fold registry, the rule keyed on "
            "convert_element_type, or the semantics the probe checks. If the "
            "control is RED TOO, it is this repository.",
            file=sys.stderr,
        )
        return 1
    if control_fatal:
        print(f"\ncanary: {control_note}", file=sys.stderr)
        return 1
    # LAST, deliberately. The two checks above are about whether the
    # instrument works at all, which is this job's headline; a contradicted
    # hash row is about whether its report can be believed, which is only a
    # question once it produced one. Both exit 1, so the order changes only
    # which sentence a reader sees first, and "it could not arm" is the one
    # that should be first.
    if hash_fatal:
        print(
            f"\ncanary: {hash_note}\n"
            "A released wheel does not change, so this is not upstream moving "
            "under us: either the row in `_adapter_jax._KNOWN_HASHES` is "
            "wrong for this release, or this environment is not running the "
            "jax it reports. Read the rule, and fix whichever of the two it "
            "is, before believing anything else on this page.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
