# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Arm the SHIPPED tripwire, print what it attached to, and say whether it worked.

Used by `.github/workflows/nightly-jax-canary.yml` for both legs — the nightly
and the control — so the two are the same probe by construction.

IT CALLS `stelling._tripwire.arm()`. That is the entire design constraint: it
is the same call `pytest_configure` makes, so a canary that goes green is a
statement about the code that ships rather than about a re-implementation of
it that drifted. Everything else here is printing.

THE EXIT CODE IS THE DECISION AND THE MESSAGE IS NOT. This file has had to
learn that twice. The first version decided with ``"DID NOT FIRE" in control``
— a substring test against a line written for a human — so the one state in
which the probe never ran was the one state that reported success. The second
kept a state machine but published only prose, so a test could not tell which
of several reasons had produced a 1 and an alarm could be deleted without any
of them noticing. Both halves are fixed the same way: **every reason to exit
non-zero has a stable code, the codes are printed, and the exit status is
`1 if reasons else 0`.**

THE OUTPUT CONTRACT, which CI and the tests both read:

* stdout carries one ``name: value`` row per line. ``control state`` and
  ``control report`` are the two words the decision is keyed on; ``live
  control`` is the same facts for a human and is NEVER re-parsed by anything.
* every reason to exit 1 prints one line on stderr, ALL of them and not
  only the first, in a fixed order -- arming, then the control, then the
  hash::

      canary [<reason code>]: <sentence>

* anything that is a note rather than a reason prints ``canary note: …`` and
  does not touch the exit code.
* ``$GITHUB_STEP_SUMMARY``, when set, gets the same rows and the same reasons.

EXIT CODES, ALL OF THEM. This paragraph has been wrong three times — once
naming two of three, once counting ``return`` statements instead of reasons,
once claiming completeness while two exits went unnamed — so it is now
checked: ``tests/test_tripwire_record.py`` parses the codes out of the list
below and asserts that the set matches the set a driven ``main()`` can
actually produce, in both directions.

  2  argparse rejected the command line (a mistyped ``--require``, say).
     argparse's own exit, not this script's, and it is in this list because
     an exit code a reader can meet is an exit code this list owes them.
  1  `not-armed` — ``--require`` was passed and the tripwire could not arm.
  1  `control:did-not-fire` — the tripwire armed and its live control DID NOT
     FIRE, ``--require`` or not: ``arm()`` says the hook is attached and the
     control says nothing reached it, so every figure below it is unverified.
  1  `control:raised` — the tripwire armed and its live control DID NOT
     COMPLETE, ``--require`` or not. A control that could not run is not a
     control that passed. This used to exit 0.
  1  `control:indeterminate` — the probe RAN and this script could not read
     the recorder at all, so whether the hook fired is unknown. Unreachable
     without editing this repository: it means ``Recorder.sorted_findings``
     moved under the line that calls it.
  1  `control:unrenderable` — the probe RAN, the hook's verdict was read, and
     this script could not RENDER what it saw. Unreachable without editing
     this repository, for the same kind of reason. Separate from the two
     above because ``raised`` is documented to mean the probe did not run and
     ``did-not-fire`` to mean it ran and the hook was dead; folding a
     rendering fault into either makes one of those false.
  1  `control:unknown-state` — the control reported a state this script has
     no answer for. Unreachable without editing this repository, and fatal on
     purpose: an instrument that cannot say what happened has not said that
     nothing happened.
  1  `hash:contradicted` — the rule hash CONTRADICTS the row recorded for
     this exact release, ``--require`` or not. See `_hash_row`.
  1  `hash:unknown-state` — ``Status.hash_state`` reported a fourth thing.
     Unreachable without editing this repository; fatal for the same reason
     as `control:unknown-state`, and see `_hash_row` for why that is the
     answer here too.
  0  anything else. That includes NOT ARMED without ``--require`` — the shape
     a human wants when running this by hand to see what a given jax does —
     a release with NO ROW in the version -> hash map, which is loud on
     stdout and in the step summary and still exits 0, and a
     ``$GITHUB_STEP_SUMMARY`` that could not be written, which is
     infrastructure and says so rather than paging.
"""

from __future__ import annotations

import argparse
import os
import sys

#: What the two legs of `nightly-jax-canary.yml` are, in one place because
#: two copies of this paragraph 190 lines apart is how the file came to carry
#: two different and both-wrong versions of it. `tests/test_tripwire_record.py`
#: reads the workflow and checks these descriptions against the install steps
#: they describe.
#:
#: THE CLOSING CLAIM IS DELIBERATELY NEGATIVE. An earlier version ended "which
#: the jax versions on the two pages tell apart", and that is false: both legs
#: run this repository's code, so both being red says the fault is common to
#: them and does not say which of the two it is. Knowing the two jax versions
#: does not separate "the released jax regressed" from "stelling is broken".
_TWO_LEGS = (
    "Compare the two legs. NEITHER IS PINNED TO A SERIES: `nightly` installs "
    "jax from the nightly index, and `control` installs `.[jax]`, which "
    "resolves to whatever jax is NEWEST RELEASED on the day it runs. So a "
    "red `nightly` beside a green `control` is the nightly's jax and "
    "therefore upstream; red on BOTH is either a released-jax regression or "
    "this repository, and nothing on these two pages separates those -- both "
    "legs run this repository's code, so agreeing tells you only that the "
    "fault is common to them. Read the rule."
)

#: The states the live control can be recorded in. Declared rather than
#: implied so that `_control_reasons` can recognise a state it was never
#: taught, and so a test can assert every one of them is REACHABLE -- a
#: declared state nobody can drive is a claim, not a state.
CONTROL_STATES = ("not-run", "raised", "ran", "did-not-fire", "fired")

#: Whether this script could say what the control saw. A SECOND axis, because
#: conflating it with the first is what made `raised` -- documented to mean
#: the probe did not run -- get reported for a probe that ran fine and a
#: recorder field that had moved.
RENDER_STATES = ("not-run", "rendered", "unrenderable")


def _control_reasons(
    control_state: str, render_state: str, rendered: str
) -> list[tuple[str, str]]:
    """Every ``(reason code, sentence)`` the live control owes, possibly none.

    TWO AXES, NOT ONE, AND THEY ARE KEYED ON STATES. The predecessor asked
    ``"DID NOT FIRE" in control`` -- a substring test against a line built for
    a human to read -- which is the same shape of instrument this repository
    keeps having to withdraw: the rendered message is not the decision, and
    re-parsing it makes the decision depend on wording. The caller records
    what happened; this reads the record.

    A LIST AND NOT A ``(sentence, fatal)`` PAIR. The pair could disagree with
    itself -- a sentence with ``fatal=False``, or ``None`` with ``True`` --
    and a test had to be written to check that it did not. A list cannot: a
    reason is a reason, and no reason is an empty list.

    The first axis, ``control_state``:

    ``fired``
        the probe ran and the hook saw it. The only clean state.
    ``not-run``
        the tripwire did not arm, so there was nothing to control. Not a
        finding HERE; whether not-arming is fatal is ``--require``'s question
        and is answered before this one is asked.
    ``did-not-fire``
        the probe RAN and the hook saw nothing. ``arm()`` says the hook is
        attached; the control says nothing reached it. A dead hook.
    ``raised``
        the probe did NOT run. Different finding, different remedy:
        `did-not-fire` means a dead hook, `raised` means an environment in
        which the probe could not execute at all.
    ``ran``
        the probe ran and the recorder could not be read, so whether the hook
        fired is unknown. Not the same as `did-not-fire`: silence and an
        unreadable answer are different answers.

    The second axis, ``render_state``, is INDEPENDENT of the first and its
    reason is emitted alongside rather than instead. That independence is the
    fix for a case that used to lose half of itself: a dead hook whose
    narrowing count would not format reported ``unrenderable``, overwrote
    ``did-not-fire``, and told the operator the probe completing "is not in
    doubt" -- while never mentioning that the hook was dead.

    THE UNKNOWN STATE IS FATAL on both axes, deliberately. A state this
    function does not recognise means the caller grew an outcome nobody
    taught this decision about, and the whole point of the change that added
    this function is that an instrument whose own state is broken must page
    rather than pass.
    """
    reasons: list[tuple[str, str]] = []

    if control_state == "did-not-fire":
        reasons.append((
            "control:did-not-fire",
            "the tripwire armed and its live control did not fire. `arm()` "
            "says the hook is attached; the control says nothing reached "
            "it. Treat the armed status as unverified. " + _TWO_LEGS,
        ))
    elif control_state == "raised":
        reasons.append((
            "control:raised",
            f"the tripwire armed and its LIVE CONTROL DID NOT COMPLETE -- "
            f"{rendered}. A control that could not run is not a control "
            "that passed: the rows above come from `arm()` and stand, but "
            "nothing about whether the hook is ALIVE has been established. "
            "This is a DIFFERENT finding from `did not fire`, which means "
            "the probe ran and the hook was dead. Read the exception, then "
            + _TWO_LEGS,
        ))
    elif control_state == "ran":
        reasons.append((
            "control:indeterminate",
            f"the tripwire armed, its live control RAN, and this script "
            f"could not read the recorder to find out whether the hook saw "
            f"it -- {rendered}. That is NOT `did not fire`: the hook was "
            "never asked. It is a defect HERE and not upstream, so do not go "
            "reading jax's changelog for it.",
        ))
    elif control_state not in CONTROL_STATES:
        reasons.append((
            "control:unknown-state",
            f"the live control reported a state this script does not "
            f"recognise ({control_state!r}). That is a bug in this script, "
            "not a measurement, and it pages rather than passing because an "
            "instrument that cannot say what happened has not said that "
            "nothing happened.",
        ))

    if render_state == "unrenderable":
        reasons.append((
            "control:unrenderable",
            f"the tripwire armed, its live control ran AND WAS READ, and "
            f"this script could not report what it saw -- {rendered}. The "
            "probe "
            "completing is not in doubt; what moved is the shape of the "
            "recorder this script formats. That is a defect HERE and not "
            "upstream, so do not go reading jax's changelog for it. Any "
            "`control:` line beside this one still stands: this reason is "
            "about the REPORT and says nothing about the hook.",
        ))
    elif render_state not in RENDER_STATES:
        reasons.append((
            "control:unknown-state",
            f"the control REPORT reported a state this script does not "
            f"recognise ({render_state!r}). Same argument as the control "
            "state above: an instrument whose own state is broken pages.",
        ))

    return reasons


def _hash_row(status) -> tuple[str, tuple[str, str] | None]:
    """``(what to print beside the sha1, the reason to exit 1 or None)``.

    THE MAP APPLIES HERE TOO, and it has four states, not two. This script
    used to compare against one constant and print `CHANGED from …` for
    anything that was not it — which, once the constant became
    `_KNOWN_HASHES` keyed on the running release, would have called a
    release nobody has ever read "changed".

    The four ``Status.hash_state`` can be, and the exit code each gets:

    * **the row matches** (``as-tested``) — quiet, exit 0. Nothing to say.
    * **no row for this release** (``never-read``) — LOUD, exit 0. Nobody has
      read the rule on this jax yet. It is the state a jax NIGHTLY is in by
      construction — a dev build can never be given a row — and it is the
      state the `control` leg enters the day jax ships a release, since that
      leg installs `.[jax]` and therefore resolves to whatever is newest.
      Paging on it would redden this workflow on every jax release and every
      night it runs against a nightly, which is how an alarm stops being
      read. `PLAN-tripwire.md` §5 is about arming and does not decide this;
      the canary's own Property 3 — infrastructure must not page — is the
      closer analogy, and this is the same shape: a true statement that no
      one can act on tonight. The remedy is a human reading the rule and
      adding a row, which a red build does not speed up.
    * **the row exists and disagrees** (``changed``) — LOUD, exit 1,
      ``--require`` or not. This is the one that cannot be explained by
      upstream moving: a released wheel is immutable, so the row is wrong, or
      the environment is not the jax it claims. Either makes every other line
      here unverified, which is exactly the standing the live-control check
      already exits 1 for.
    * **the source could not be read** (``unreadable``) — quiet, exit 0.
      Nothing was compared, so nothing is contradicted; there is no finding
      to page on, only an absent one.

    AND A FIFTH THING IS FATAL, which resolves a disagreement this file used
    to contain. The fallback here was ``return "not read", False`` — an
    unrecognised state swallowed as non-fatal — while `_control_reasons`
    treats an unrecognised state as fatal on a stated principle. Two
    decisions in one file answering the same question two ways is one of them
    being wrong, and it is this one: ``unreadable`` is a state with an
    argument for exiting 0 (nothing was compared), and "some fifth string"
    has no argument at all. It is now handled BY NAME and the fallback pages.
    """
    state = status.hash_state
    if state == "as-tested":
        return "as tested", None
    if state == "unreadable":
        return "not read -- the rule's source could not be read", None
    if state == "never-read":
        return (
            f"jax {status.jax_version or '?'} HAS NEVER BEEN READ — no row "
            "in _adapter_jax._KNOWN_HASHES records a rule for this release. "
            "Read the rule, diff it against the nearest row, and add an "
            "entry naming what changed. Not a failure: this is what a "
            "nightly, and any jax released since the last row was written, "
            "looks like",
            None,
        )
    if state == "changed":
        return (
            f"CONTRADICTS the row for jax {status.jax_version or '?'}, which "
            f"records {status.known_hash} — read the rule before trusting "
            "anything on this page",
            (
                "hash:contradicted",
                f"the rule hash CONTRADICTS the row for jax "
                f"{status.jax_version or '?'}, which records "
                f"{status.known_hash}. A released wheel does not change, so "
                "this is not upstream moving under us: either the row in "
                "`_adapter_jax._KNOWN_HASHES` is wrong for this release, or "
                "this environment is not running the jax it reports. Read "
                "the rule, and fix whichever of the two it is, before "
                "believing anything else on this page.",
            ),
        )
    return (
        f"UNKNOWN HASH STATE {state!r}",
        (
            "hash:unknown-state",
            f"`Status.hash_state` reported {state!r}, which this script has "
            "no answer for. That is a bug in this repository, not a "
            "measurement, and it pages for the same reason an unrecognised "
            "control state does: a rule hash nobody can classify has not "
            "been shown to agree with its row.",
        ),
    )


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
    #
    # THREE `try` REGIONS, NOT ONE, AND THE SPLIT IS THE POINT. Each answers
    # exactly one question, and the boundary is where the answer changes:
    #
    #   1. did the probe RUN?          -> `raised` if not
    #   2. what did the recorder SAY?  -> `fired` / `did-not-fire`, or `ran`
    #                                     if the recorder could not be read
    #   3. can this page RENDER it?    -> `unrenderable` if not
    #
    # A single block spanning all three made (1) false: a control that ran,
    # fired and produced a finding was reported as `raised` if the f-string
    # tripped over a recorder field that had moved. Moving the reads out of
    # the guarded region instead made (2) and (3) indistinguishable and let a
    # rendering fault overwrite a dead hook. Only the trace call decides (1),
    # only the recorder read decides (2), and only the formatting decides (3).
    control_state = "not-run"
    render_state = "not-run"
    control = "not run"
    if status.armed:
        try:
            from stelling._jax_compat import jax as _jax
            from stelling._jax_compat import jnp as _jnp
            from stelling._tripwire import _probe

            _jax.make_jaxpr(_probe.over)(_jnp.zeros((7,), _jnp.int8))
        except Exception as exc:  # noqa: BLE001
            control_state = "raised"
            control = f"raised {type(exc).__name__}: {exc}"
        else:
            # The probe RAN. That fact is established here and nothing below
            # withdraws it -- which is exactly what the old single block did.
            control_state = "ran"
            try:
                found = recorder.sorted_findings()
            except Exception as exc:  # noqa: BLE001
                control = (
                    f"the probe ran and reading the recorder raised "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                control_state = "fired" if found else "did-not-fire"
                try:
                    control = (
                        f"{len(found)} finding over "
                        f"{recorder.int_narrowings} narrowing(s)"
                        + (
                            f"; {found[0].written} -> {found[0].became} "
                            f"({found[0].to_dtype})"
                            if found
                            else " -- THE CONTROL DID NOT FIRE"
                        )
                    )
                    render_state = "rendered"
                except Exception as exc:  # noqa: BLE001
                    render_state = "unrenderable"
                    control = (
                        f"{len(found)} finding(s), and rendering them raised "
                        f"{type(exc).__name__}: {exc}"
                    )

    disarmed = _tripwire.disarm()

    hash_note, hash_reason = _hash_row(status)

    # EVERY REASON TO EXIT 1, IN ONE LIST, BUILT BEFORE ANYTHING IS PRINTED.
    # The exit status is `1 if reasons else 0` and there is no other route
    # out, so a reason that stops being appended is a reason that stops
    # paging -- and the code beside each sentence is what lets a test say
    # WHICH reason it measured rather than only that something exited 1.
    #
    # ORDER. Arming first: it is this job's headline and the sentence a
    # reader should meet first. Then the control -- whether the instrument
    # works at all. Then the hash, which is about whether the instrument's
    # report can be believed and is only a question once it produced one.
    # ALL of them print, not just the first: two independent things can be
    # wrong at once, and a page that names one of them sends half an
    # operator's afternoon in the wrong direction.
    reasons: list[tuple[str, str]] = []
    if args.require and not status.armed:
        reasons.append((
            "not-armed",
            f"the tripwire could not arm [{status.code}]. " + _TWO_LEGS,
        ))
    reasons.extend(_control_reasons(control_state, render_state, control))
    if hash_reason is not None:
        reasons.append(hash_reason)

    rows = [
        ("status", status.code),
        ("jax", status.jax_version or "?"),
        ("rule", status.rule_name or "?"),
        ("rule sha1", f"{status.rule_hash or '?'} ({hash_note})"),
        ("registry size", str(status.registry_size)),
        ("locate()", adapter.locate()),
        # THE TWO WORDS THE DECISION IS KEYED ON, on the page. `live control`
        # below says the same things to a human and is never re-parsed; these
        # are what `_control_reasons` was given, so a page that exits 1 says
        # which branch produced it and a page that exits 0 says which state it
        # was in when it did.
        ("control state", control_state),
        ("control report", render_state),
        ("live control", control),
        ("disarm()", disarmed),
        ("detail", status.explanation),
    ]

    for name, value in rows:
        print(f"{name}: {value}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            _write_summary(summary, status, rows, reasons)
        except OSError as exc:  # noqa: BLE001
            # INFRASTRUCTURE MUST NOT PAGE -- Property 3 of the workflow, and
            # this file's own `_hash_row` invokes it. The step summary is a
            # convenience channel for a human reading the run page; it is not
            # the measurement, and it is not this script's to fix. It used to
            # raise straight through `main()`, so a runner that handed us an
            # unwritable path produced a traceback and a 1 -- infrastructure
            # paging, with no `canary:` sentence, in the script that argues
            # against exactly that.
            print(
                f"canary note: $GITHUB_STEP_SUMMARY ({summary!r}) could not "
                f"be written -- {type(exc).__name__}: {exc}. That is "
                "infrastructure, not signal: the rows above are the same "
                "ones the summary would have carried and the exit status "
                "below is unchanged.",
                file=sys.stderr,
            )

    for code, sentence in reasons:
        print(f"\ncanary [{code}]: {sentence}", file=sys.stderr)
    return 1 if reasons else 0


def _write_summary(path, status, rows, reasons) -> None:
    """The same rows and the same reasons, on the run's summary page.

    THE REASONS GO HERE TOO. The workflow's own header says a failure must be
    diagnosable from this page without re-running anything, and a table of
    facts with the verdict left on stderr is not that.
    """
    verdict = "ARMED" if status.armed else f"NOT ARMED [{status.code}]"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"### nightly jax canary: {verdict}\n\n")
        handle.write("| what | value |\n|---|---|\n")
        for name, value in rows:
            handle.write(f"| {name} | {value} |\n")
        if reasons:
            handle.write("\n**exit 1**, for every reason below:\n\n")
            for code, sentence in reasons:
                handle.write(f"* `{code}` — {sentence}\n")
        else:
            handle.write("\nNo reason to exit 1. **exit 0**.\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
