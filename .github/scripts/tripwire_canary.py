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
below, parses the reason literals out of THIS FILE, and asserts the two sets
are equal in both directions. THE SECOND SET USED TO BE THE CODES A TABLE OF
DRIVEN RUNS EXPECTED, which is a weaker thing than it reads as: a reason
appended under a condition no row of that table drives is a code the table
never names, so the list below could go on being silent about it. The codes
those runs really printed are still read, now as a control on the parse.

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
  1  `control:unknown-state` — one of the control's TWO AXES reported a state
     this script has no answer for, and BOTH sites emit this one code: a
     `control state` outside ``CONTROL_STATES``, or a `control report`
     outside ``RENDER_STATES``. One code because the remedy is one remedy and
     it is in this file; the sentence names which axis it read and what it
     read there, so a page that carries this code still says which. Two
     emission sites and one documented is how this list was wrong the third
     time. Unreachable without editing this repository, and fatal on purpose:
     an instrument that cannot say what happened has not said that nothing
     happened.
  1  `hash:contradicted` — the rule hash CONTRADICTS the row recorded for
     this exact release, ``--require`` or not. See `_hash_row`.
  1  `hash:unknown-state` — ``Status.hash_state`` reported a fourth thing.
     Unreachable without editing this repository; fatal for the same reason
     as `control:unknown-state`, and see `_hash_row` for why that is the
     answer here too.
  1  `eager:not-armed` — ``--require`` was passed and the EAGER
     construction-site detector (Mode 2) could not attach. Its own arm-time
     self-check drives every construction route it claims and refuses on any
     one of them going blind, so this code covers a moved module, a moved
     signature and a route jax stopped sending through the site alike; the
     sentence names which.
  1  `eager:did-not-fire` — the eager detector attached and its live control
     DID NOT RAISE, ``--require`` or not. `arm_eager()` says the hook is
     attached and a construction that must be refused was allowed. Every
     figure beside it is unverified, exactly as for the tripwire's control.
  1  `eager:cries-wolf` — the eager detector attached and refused an
     IN-RANGE construction. Every alarm it would raise this run is noise, so
     it pages rather than being trusted.
  1  `eager:raised` — the eager detector's live control DID NOT COMPLETE:
     something other than `EagerTruncationError` came out of it, so whether
     the hook works has not been established. A control that could not run is
     not a control that passed.
  1  `hooks:displaced` — a hook this script armed was displaced before it
     disarmed: something else in this environment is bound over stelling's
     wrapper. ONE question covering BOTH hooks, asked while both are live.
     Same finding the trace gate's displacement check makes, in the one
     process where nothing else should be running.
  1  `eager:hash-contradicted` — the eager narrowing site's source hash
     CONTRADICTS the row recorded for this exact release. Same argument as
     `hash:contradicted`, about the other site: a released wheel is
     immutable, so either the row is wrong or this is not the jax it claims.
  1  `eager:unknown-state` — the eager control, or its hash, reported a state
     this script has no answer for. Unreachable without editing this
     repository, and fatal for the reason `control:unknown-state` is.
  1  `eager:unenumerated-jax-constant` — the sweep found jax performing an
     eager truncation of its OWN that `_adapter_jax._JAX_EAGER_CONSTANTS` has
     no row for. Until a row is written, that narrowing is attributed to
     whoever called jax and RAISES at a line inside jax they did not write.
     Unlike a rowless RELEASE at the hash map — which is loud and exits 0 —
     this one is already producing wrong alarms, so it pages. See
     `_eager_sweep_row`.
  1  `perimeter:not-armed` — ``--require`` was passed and the DUNDER
     PERIMETER (Mode 3) could not attach. Its own arm-time self-check drives
     the reference defect through the live slots in both directions and
     refuses on either going wrong, so this code covers a moved type, a
     missing slot, a blind guard and a guard that refuses everything alike;
     the sentence names which.
  1  `perimeter:did-not-fire` — the perimeter attached and its live control
     DID NOT REFUSE a literal that does not survive the conversion. Every
     figure beside it is unverified, exactly as for the tripwire's control.
  1  `perimeter:cries-wolf` — the perimeter attached and refused a literal
     that IS exactly representable. Every refusal it would make this run
     would be noise, so it pages rather than being trusted.
  1  `perimeter:raised` — the perimeter's live control DID NOT COMPLETE:
     something other than ``NarrowingError`` came out of it, so whether the
     hook works has not been established.
  1  `perimeter:facts-moved` — one of the facts the perimeter RESTS ON no
     longer holds on this jax: the type could not be located, it is no longer
     a heap type, it no longer owns the slots, ``setattr`` no longer rebinds
     and restores identity, a WARM operation no longer enters Python, or an
     in-place slot has appeared so that ``x += N`` no longer falls back to
     the forward slot. Each is a positive assertion, so anything that
     disappears turns it red rather than passing quietly. The sentence names
     the rows that moved.
  1  `perimeter:promotion-drift` — the dtype the predicate says a literal is
     converted INTO disagrees with the dtype jax actually converts it into.
     That identity is the whole of the predicate's F6, and every verdict it
     gives is computed against the target it names; a disagreement means the
     answers are being computed about the wrong dtype.
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

#: The states the EAGER detector's live control can be recorded in. Declared
#: for the same reason ``CONTROL_STATES`` is: so that ``_eager_reasons`` can
#: recognise a state it was never taught rather than passing it.
#:
#: FOUR STATES AND NOT FIVE, and the missing one is the tell. The tripwire's
#: control has an ``indeterminate`` state because reading its answer means
#: reading a RECORDER, which can move. This control's answer is whether an
#: exception came out of one line, so there is no second read to fail: the
#: probe raised the right exception (`fired`), raised nothing (`did-not-fire`),
#: raised something else (`raised`), or refused an in-range value
#: (`cries-wolf`).
EAGER_CONTROL_STATES = ("not-run", "raised", "did-not-fire", "cries-wolf", "fired")



#: The states the PERIMETER's live control can be recorded in. Declared for
#: the reason :data:`CONTROL_STATES` is: so that :func:`_perimeter_reasons`
#: can recognise a state it was never taught rather than passing it.
PERIMETER_CONTROL_STATES = ("not-run", "raised", "did-not-fire", "cries-wolf", "fired")


def _perimeter_facts(face, located):
    """The facts the perimeter RESTS ON for one face, each a positive assertion.

    Returns ``(rows, moved)``: rows for the page, and the names of the ones
    that did not hold. **Every row asserts that something is STILL TRUE**, so
    a jax that drops the heap-type flag, renames the type, stops routing a
    warm operation through Python, or grows an ``__iadd__`` turns a row RED
    rather than passing quietly. That direction is the whole design: this is
    an alarm about upstream drift, and an alarm that only fires when something
    NEW appears cannot report a disappearance.

    ``located`` is the type ``perimeter_locate`` returned, or the status code
    string it returned instead -- in which case every row below is unanswerable
    and that is itself the finding.

    TWO FACES, ONE FUNCTION, and the rows are named with the face they are
    about: the array face rests on the same five facts as the tracer face,
    asked of a different type, and a page that reported one set for "the
    perimeter" would be reporting half of what is armed.
    """
    rows: list[tuple[str, str]] = []
    moved: list[str] = []

    def record(name, ok, detail):
        label = f"{face}: {name}"
        rows.append((label, f"{'PASS' if ok else 'RED '} {detail}"))
        if not ok:
            moved.append(label)

    if isinstance(located, str):
        record("type", False, f"could not be located: {located}")
        return rows, moved

    from stelling._jax_compat import jax as _jax
    from stelling._jax_compat import jnp as _jnp
    from stelling._tripwire import perimeter as _perimeter

    record("type", True, f"{located.__module__}.{located.__qualname__}")
    # CPython's own flag, read off the type. A static type cannot have its
    # slots rebound at all, so this row is the one that says `setattr` is even
    # legal here -- and it is checked as well as `setattr` being driven,
    # because the two fail in different ways.
    heap = bool(located.__flags__ & (1 << 9))
    record("Py_TPFLAGS_HEAPTYPE", heap, f"__flags__=0x{located.__flags__:x}")

    slots = _perimeter.FACE_SLOTS[face]
    missing = [s for s in slots if s not in located.__dict__]
    record(
        "slots are own attributes",
        not missing,
        f"{len(slots) - len(missing)}/{len(slots)}"
        + (f", missing {missing}" if missing else ""),
    )

    # The slot each face's warm-op probe drives, and the program that drives
    # it. `__le__` on a tracer is what `check()` runs through; `__add__` on an
    # array is the eager door's headline.
    probe_slot = "__le__" if face == "tracer" else "__add__"
    if face == "tracer":
        def _run():
            _jax.clear_caches()
            _jax.make_jaxpr(lambda z: z <= 1)(_jnp.zeros((3,), _jnp.float32))
    else:
        _arr = _jnp.zeros((3,), _jnp.int16)

        def _run():
            _arr + 1

    entered = [0]
    restored = False
    if probe_slot not in missing and probe_slot in located.__dict__:
        original = located.__dict__[probe_slot]

        def _counting(a, b):
            entered[0] += 1
            return original(a, b)

        # WARM FIRST, then rebind. The question is not "does a cold op enter
        # Python" -- it is whether a WARM one still does. A jax that routed a
        # warm operation entirely through C++ would leave every `setattr`
        # succeeding and every wrapper cold, which is the silent failure this
        # row exists for.
        try:
            _run()
            setattr(located, probe_slot, _counting)
            for _ in range(2):
                _run()
        finally:
            setattr(located, probe_slot, original)
            restored = located.__dict__[probe_slot] is original
    record(
        "setattr rebinds and restores identity", restored, f"identity back={restored}"
    )
    record(
        f"a WARM {probe_slot} still enters Python", entered[0] == 2, f"{entered[0]}/2"
    )

    inplace = [s for s in _perimeter.INPLACE_SLOTS if s in located.__dict__]
    # `x += N` falls back to the forward slot only while there is no in-place
    # slot to prefer. If jax ever adds one, the perimeter grows a hole in a
    # spelling people write constantly, and this row is how anyone finds out.
    record(
        "no in-place slots to bypass the forward ones",
        not inplace,
        f"{inplace or 'none'}",
    )
    return rows, moved


def _perimeter_promotion(sample=None):
    """Does the predicate name the dtype jax actually converts the literal into?

    ``(note, disagreements)``. This is F6, the identity every verdict the
    predicate gives is computed against: it asks jax's own promotion for the
    target and then checks each answer against what jax DOES -- the dtype of
    ``x + 3`` for the dtype-preserving slots, and of ``x / 3`` for the two
    that redirect into a float. Two different code paths in jax, which is what
    makes the agreement worth something.
    """
    from stelling._jax_compat import jnp as _jnp
    from stelling._tripwire import prop_guard

    names = sample or (
        "bool", "int8", "int16", "int32", "uint8", "uint16", "uint32",
        "float16", "float32", "bfloat16", "complex64",
    )
    agree = 0
    disagree: list[str] = []
    for name in names:
        try:
            x = _jnp.zeros((1,), name)
        except Exception:  # noqa: BLE001 - a dtype this build does not have
            continue
        for slot, probe in (("add", lambda a: a + 3), ("truediv", lambda a: a / 3)):
            try:
                predicted = prop_guard._target_dtype(x.dtype, slot)
                actual = probe(x).dtype
            except Exception as exc:  # noqa: BLE001
                disagree.append(f"{name}/{slot}: {type(exc).__name__}")
                continue
            if predicted is not None and str(predicted) == str(actual):
                agree += 1
            else:
                disagree.append(f"{name}/{slot}: says {predicted}, jax uses {actual}")
    note = f"{agree} agree, {len(disagree)} disagree"
    if disagree:
        note += " -- " + "; ".join(disagree[:4])
    return note, disagree


def _perimeter_reasons(status, control_state, control, facts_moved, promotion_drift,
                       require):
    """Every ``(reason code, sentence)`` the perimeter owes, possibly none."""
    reasons: list[tuple[str, str]] = []
    if require and not status.armed:
        reasons.append((
            "perimeter:not-armed",
            f"the dunder perimeter could not attach [{status.code}]: "
            f"{status.explanation} " + _TWO_LEGS,
        ))
    if control_state not in PERIMETER_CONTROL_STATES:
        reasons.append((
            "perimeter:facts-moved",
            f"the perimeter's live control reported the state "
            f"{control_state!r}, which this script has no answer for. An "
            "instrument that cannot say what happened has not said that "
            "nothing happened.",
        ))
    elif control_state == "did-not-fire":
        reasons.append((
            "perimeter:did-not-fire",
            f"the perimeter attached and its live control did not refuse "
            f"{_probe_moved()}: {control}. Attached-but-blind is what a jax "
            "release actually produces, and it is the failure a presence "
            "check misses.",
        ))
    elif control_state == "cries-wolf":
        reasons.append((
            "perimeter:cries-wolf",
            f"the perimeter refused a literal that IS exactly representable: "
            f"{control}. Every refusal it would make this run would be noise.",
        ))
    elif control_state == "raised":
        reasons.append((
            "perimeter:raised",
            f"the perimeter's live control did not complete: {control}. A "
            "control that could not run is not a control that passed.",
        ))
    if facts_moved:
        reasons.append((
            "perimeter:facts-moved",
            "the perimeter rests on these facts and they no longer hold: "
            + ", ".join(facts_moved)
            + ". Each row is a positive assertion about this jax, so this is "
            "a disappearance rather than a new problem, and the perimeter's "
            "attachment is the thing to re-derive before trusting it again.",
        ))
    if promotion_drift:
        reasons.append((
            "perimeter:promotion-drift",
            "the dtype the predicate says a literal is converted into "
            "disagrees with the dtype jax actually converts it into: "
            + "; ".join(promotion_drift[:4])
            + ". Every verdict the predicate gives is computed against the "
            "target it names.",
        ))
    return reasons


def _probe_moved() -> str:
    from stelling._tripwire import _probe

    return (
        f"{_probe.PERIMETER_OVER} against {_probe.PERIMETER_DTYPE}, or "
        f"{_probe.ARITH_OVER} into {_probe.ARITH_DTYPE}"
    )


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


def _eager_reasons(status, control_state, control, displaced, require):
    """Every ``(reason code, sentence)`` the eager half owes, plus the shared
    displacement question, possibly none.

    THE SAME SHAPE AS :func:`_control_reasons`, DELIBERATELY, and not a
    generalisation of it. The two instruments' states are not the same states
    -- this one has no ``indeterminate`` and gains ``cries-wolf`` -- and a
    shared function would have had to grow a mode switch, which is how one
    decision becomes two decisions that can disagree. What IS shared is the
    principle: the caller records what happened, this reads the record, an
    unrecognised state pages, and a list of reasons cannot contradict itself
    the way a ``(sentence, fatal)`` pair can.

    ARMING IS ``--require``'s QUESTION and the control's is not, exactly as
    above. A human running this by hand against a jax that moved the site
    wants to SEE that and not to be paged by it; a CI job that depends on the
    detector passes ``--require``. But a detector that armed and then failed
    its own control is a broken instrument in either mode, so those page
    regardless.
    """
    reasons = []
    if require and not status.armed:
        reasons.append((
            "eager:not-armed",
            f"the eager construction-site detector could not attach "
            f"[{status.code}] -- {status.explanation} " + _TWO_LEGS,
        ))

    if control_state == "did-not-fire":
        reasons.append((
            "eager:did-not-fire",
            "the eager detector attached and its live control DID NOT RAISE: "
            "a construction this tool must refuse was allowed through. "
            "`arm_eager()` says the hook is attached and the control says "
            "nothing reached it. " + _TWO_LEGS,
        ))
    elif control_state == "cries-wolf":
        reasons.append((
            "eager:cries-wolf",
            f"the eager detector refused an IN-RANGE construction -- "
            f"{control}. Every alarm it would raise this run is noise, so it "
            "is reported as broken rather than trusted. This is a DIFFERENT "
            "finding from `did not fire`: the hook is alive and its range "
            "arithmetic disagrees with jax's.",
        ))
    elif control_state == "raised":
        reasons.append((
            "eager:raised",
            f"the eager detector's LIVE CONTROL DID NOT COMPLETE -- {control}. "
            "Something other than the exception this tool raises came out of "
            "the probe, so nothing about whether the hook works has been "
            "established. Read the exception, then " + _TWO_LEGS,
        ))
    elif control_state not in EAGER_CONTROL_STATES:
        reasons.append((
            "eager:unknown-state",
            f"the eager control reported a state this script does not "
            f"recognise ({control_state!r}). That is a bug in this script, "
            "not a measurement, and it pages because an instrument that "
            "cannot say what happened has not said that nothing happened.",
        ))

    if displaced:
        reasons.append((
            "hooks:displaced",
            f"a hook this script armed was DISPLACED before it disarmed: "
            f"{', '.join(displaced)}. Something else in this environment is "
            "binding over the same private jax surface, so an unmeasured part "
            "of this run was not watched by the instrument named. THE "
            "QUESTION COVERS BOTH HOOKS -- it is asked once, while both are "
            "live -- because a displaced const-fold rule and a displaced "
            "construction site are the same failure at two sites and used to "
            "have one detector between them and none. Nothing was clobbered: "
            "whatever replaced it is left in place.",
        ))
    return reasons


def _eager_sweep_row(armed, enabled=True) -> tuple[str, tuple[str, str] | None]:
    """``(what to print, the reason to exit 1 or None)`` for the constant sweep.

    THE OTHER MAP THIS HOOK CARRIES, and the one that decides a SUPPRESSION.
    ``_adapter_jax._JAX_EAGER_CONSTANTS`` records the eager truncations jax
    performs itself -- one row today, the threefry PRNG mask -- and a
    narrowing that matches none of them is attributed to whoever called jax
    and RAISES. That is the direction the design fails in on purpose, and it
    means a jax release that adds a second internal eager truncation turns
    into a false alarm inside jax, in every user's code, on the day it ships.

    So the map is RE-DERIVED here rather than trusted, by the shipped sweep
    the suite drives: every key implementation and seed spelling, then
    ``jax.random``'s consumers and ``jnp``'s integer ops over six integer
    dtypes, under ``jax.disable_jit()``, with every value handed in IN RANGE
    -- so every truncation observed is jax's own.

    IT PAGES, unlike :func:`_eager_hash_row`'s ``never-read``, and the
    difference is what the two states mean. A rowless RELEASE is an expected
    state that costs nothing until somebody reads it; an unenumerated jax
    CONSTANT is already producing wrong alarms. This job's control leg meets a
    new release before any CI lane does, which is the earliest anyone can be
    told.

    IT CAN ONLY FIND ONE WITH ``JAX_ENABLE_X64`` OFF, and it SAYS SO when it
    ran with x64 on. With x64 on jax's own threefry mask widens to ``int64``,
    it fits, and nothing of jax's narrows at all -- so ``unmatched`` is empty
    by construction and this row cannot page, whatever the installed jax has
    grown. Measured on jax 0.11.0: 729 conversion(s), 0 truncation(s), 0
    row(s) at x64=1 against 675, 13, 1 at x64=0. Both nightly legs therefore
    run this script twice, once per cell, and a test reads the workflow for
    the x64-off run; the note below is the second guard, for a reader of a
    single x64-on run who would otherwise read "0 unmatched" as a clearance.

    Everything that is not a measurement is a note: with no jax, with the hook
    not attached, or with ``--no-sweep``, there is nothing to sweep and
    nothing to say.

    ``--no-sweep`` EXISTS FOR ONE CALLER AND IT IS NOT A WORKFLOW. The sweep
    runs ~700 jax operations and costs about twelve seconds; this repository's
    own exit-code battery drives this script in a subprocess a dozen times to
    check which reason produced which status, and paying for twelve identical
    sweeps to learn nothing new is waste. It is DEFAULT-ON, so a workflow that
    says nothing gets it, and
    ``tests/test_tripwire_record.py::test_the_nightly_workflow_still_runs_the_canary``
    asserts neither leg passes it.
    """
    if not enabled:
        return "not run -- --no-sweep", None
    if not armed:
        return "not run -- the eager hook is not attached", None
    try:
        # IMPORTED HERE AND NOT AT MODULE SCOPE, like every other stelling
        # import in this file: it must be importable in a lane with no jax,
        # and `main()` is where the environment has been established.
        from stelling._tripwire import _adapter_jax as adapter

        swept = adapter.eager_jax_constant_sweep()
    except Exception as exc:  # noqa: BLE001 - a canary may not raise
        return f"could not run -- {type(exc).__name__}: {exc}", None
    code = swept.get("code")
    if code != "swept":
        return f"not run -- {code}", None
    conversions = swept.get("conversions", 0)
    unmatched = swept.get("unmatched") or ()
    matched = swept.get("matched") or ()
    summary = (
        f"{conversions} conversion(s), {swept.get('truncations', 0)} "
        f"truncation(s) of jax's own, {len(matched)} row(s) exercised"
    )
    if unmatched:
        return (
            f"{summary} -- {len(unmatched)} UNENUMERATED",
            (
                "eager:unenumerated-jax-constant",
                "jax performs an eager truncation of its OWN that "
                "`_adapter_jax._JAX_EAGER_CONSTANTS` has no row for: "
                f"{unmatched!r}. Until somebody reads it and adds a row, the "
                "eager detector attributes it to whoever called jax and "
                "RAISES there, at a line inside jax they did not write -- "
                "which is the failure the origin question was built to end. "
                "Read the jax frames named above, write the row, and say what "
                "the constant is.",
            ),
        )
    # THE QUALIFICATION TRAVELS WITH THE NUMBER, and it belongs on THIS
    # return rather than on the one above: a sweep that found something has
    # looked, whatever the setting, and a "could not have found one" clause
    # printed beside an `UNENUMERATED` count would contradict itself. Here
    # the figures are zeroes, and at x64 on zeroes are what a sweep that
    # could not look reports.
    if swept.get("x64"):
        return (
            summary
            + " -- taken at JAX_ENABLE_X64=1, where jax's own mask widens to "
            "int64 and NOTHING of jax's narrows, so these zeroes mean this "
            "sweep could not have found an unenumerated constant. The x64=0 "
            "run is the one that can",
            None,
        )
    return summary, None


def _eager_hash_row(status) -> tuple[str, tuple[str, str] | None]:
    """``(what to print beside the sha1, the reason to exit 1 or None)``.

    THE SAME FOUR STATES AND THE SAME VERDICTS as :func:`_hash_row`, computed
    by the same ``Status.hash_state``, about the OTHER site. A separate
    function and not a parameterised one because every sentence it emits names
    a different thing -- the eager narrowing site rather than the const-fold
    rule -- and a shared function would have had to take the prose as an
    argument, which is a template, not a decision.

    The four hash states move on DIFFERENT releases for the two sites, which
    is the concrete reason each has its own map and its own row here: 0.10.2
    and 0.11.0 are byte-identical at the const-fold rule and differ at this
    one, and 0.11.0 and 0.11.1 are the other way round.
    """
    state = status.hash_state
    if state == "as-tested":
        return "as tested", None
    if state == "unreadable":
        return "not read -- the site's source could not be read", None
    if state == "never-read":
        return (
            f"jax {status.jax_version or '?'} HAS NEVER BEEN READ at the eager "
            "narrowing site -- no row in `_adapter_jax._KNOWN_EAGER_HASHES`. "
            "Read the function, diff it against the nearest row, and add an "
            "entry naming what changed. Not a failure: this is what a nightly "
            "looks like, and what any jax released since the last row was "
            "written looks like",
            None,
        )
    if state == "changed":
        return (
            f"CONTRADICTS the eager row for jax {status.jax_version or '?'}, "
            f"which records {status.known_hash}",
            (
                "eager:hash-contradicted",
                f"the EAGER narrowing site's source hash CONTRADICTS the row "
                f"for jax {status.jax_version or '?'}, which records "
                f"{status.known_hash}. A released wheel does not change, so "
                "either the row in `_adapter_jax._KNOWN_EAGER_HASHES` is "
                "wrong for this release or this environment is not running "
                "the jax it reports. Nothing about ARMING is gated on this "
                "hash -- the detector armed or did not arm above without "
                "consulting it -- so read the function before believing "
                "anything else on this page.",
            ),
        )
    return (
        f"UNKNOWN HASH STATE {state!r}",
        (
            "eager:unknown-state",
            f"`Status.hash_state` reported {state!r} for the eager narrowing "
            "site, which this script has no answer for. A bug in this "
            "repository, not a measurement, and it pages for the reason an "
            "unrecognised control state does.",
        ),
    )


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


#: The probe input length, bumped per drive so no trace is answered from the
#: cache. A list because the driver that reads it is a closure inside
#: :func:`main`.
_perimeter_shape = [16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="store_true",
        help="exit non-zero if the tripwire could not arm (what the canary uses)",
    )
    # DEFAULT-ON, and the opt-out is for this repository's own exit-code
    # battery rather than for a workflow: see `_eager_sweep_row`.
    parser.add_argument(
        "--no-sweep",
        dest="sweep",
        action="store_false",
        help="skip re-deriving the map of jax's own eager truncations "
             "(~700 jax ops, about twelve seconds)",
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

    # ---------------------------------------------------------------------
    # THE SECOND HOOK. Mode 2 patches a private jax FUNCTION rather than a
    # registry entry, and the failure that matters is not the function
    # disappearing -- that is loud -- but the function surviving while jax
    # stops routing a construction route through it, which is silent. That is
    # upstream release drift, which is what this workflow exists for, so it is
    # watched here rather than in a second job: two canaries on two schedules
    # is two pages to read and one of them eventually stops being read.
    #
    # ARMED AFTER the tripwire has armed AND run its control, deliberately.
    # `arm()`'s own self-check TRACES a program that narrows a constant; with
    # the eager detector already live that trace would raise, and the tripwire
    # would report `unexpected:EagerTruncationError` on a perfectly healthy
    # pair of hooks. Sequential arming, so neither instrument's self-check
    # runs inside the other's jurisdiction -- and the tripwire is NOT disarmed
    # first, so that the one displacement question below is asked while both
    # hooks are live and therefore answers for both.
    eager_status = _tripwire.arm_eager()
    eager_control_state = "not-run"
    eager_control = "not run"
    if eager_status.armed:
        try:
            from stelling import EagerTruncationError
            from stelling._jax_compat import jnp as _jnp
            from stelling._tripwire import _probe

            try:
                _probe.construct_over(_jnp)
            except EagerTruncationError as exc:
                eager_control_state = "fired"
                eager_control = (
                    f"refused {exc.written} -> {exc.became} ({exc.to_dtype})"
                )
            else:
                eager_control_state = "did-not-fire"
                eager_control = (
                    f"{_probe.EAGER_OVER} into {_probe.EAGER_DTYPE} was "
                    "ALLOWED THROUGH -- THE CONTROL DID NOT FIRE"
                )
            if eager_control_state == "fired":
                # ...and the negative direction, because a hook replaced by
                # "refuse everything" passes the positive one.
                try:
                    _probe.construct_under(_jnp)
                except EagerTruncationError as exc:
                    eager_control_state = "cries-wolf"
                    eager_control = (
                        f"an IN-RANGE {exc.written} into {exc.to_dtype} was "
                        "refused"
                    )
                else:
                    eager_control += (
                        f"; and allowed the in-range {_probe.EAGER_UNDER}"
                    )
        except Exception as exc:  # noqa: BLE001
            eager_control_state = "raised"
            eager_control = f"raised {type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------------
    # THE THIRD HOOK. Mode 3 rebinds OPERATOR SLOTS on a jax type, and what it
    # rests on is a short list of facts about that type -- it is a heap type,
    # it owns the slots, `setattr` rebinds them, and a WARM operation still
    # enters Python. Every one of those is a fact about jax that a release can
    # take away silently, which is what this workflow exists for, so each is
    # asserted positively below rather than assumed.
    #
    # ARMED LAST, and for the same sequencing reason the eager detector is
    # armed second: neither instrument's arm-time self-check should run inside
    # another's jurisdiction. Nothing is disarmed first, so the one
    # displacement question below is asked while all three hooks are live and
    # therefore answers for all three.
    perimeter_status = _tripwire.arm_perimeter(owner="canary")
    perimeter_control_state = "not-run"
    perimeter_control = "not run"
    if perimeter_status.armed:
        try:
            from stelling._jax_compat import jax as _jax
            from stelling._jax_compat import jnp as _jnp
            from stelling._tripwire import _probe
            from stelling._tripwire.perimeter import NarrowingError

            def _drive(program):
                # A FRESH SHAPE EVERY TIME. jax's trace cache is process-wide,
                # so re-tracing at the same avals never re-runs the probe's
                # Python -- and a hook that is never entered looks exactly
                # like a hook that is entered and stays quiet.
                _perimeter_shape[0] += 1
                return _jax.make_jaxpr(program)(
                    _jnp.zeros((_perimeter_shape[0],), _jnp.float32)
                )

            def _drive_eager(program):
                # The ARRAY face, driven WARM: the operation runs twice, and
                # the second one is the question. The eager door this face
                # closes is specifically the warm one -- cold and warm are
                # both `-25536` with nothing else armed, and a cold-only
                # control would pass on a jax answering warm ops from C++.
                arr = _jnp.zeros((3,), _jnp.int16)
                program(arr)
                return program(arr)

            try:
                _drive(_probe.compare_over)
                _drive_eager(_probe.arith_over)
            except NarrowingError as exc:
                perimeter_control_state = "fired"
                perimeter_control = (
                    f"refused {exc.finding.literal} -> "
                    f"{exc.finding.narrowed_to} ({exc.finding.target_dtype})"
                    f" at {exc.file}:{exc.line}"
                )
            else:
                perimeter_control_state = "did-not-fire"
                perimeter_control = (
                    f"{_probe.PERIMETER_OVER} compared against a "
                    f"{_probe.PERIMETER_DTYPE} tracer, or {_probe.ARITH_OVER} "
                    f"added to a warm {_probe.ARITH_DTYPE} array, was ALLOWED "
                    "THROUGH -- THE CONTROL DID NOT FIRE"
                )
            if perimeter_control_state == "fired":
                # ...and the negative direction, because a perimeter replaced
                # by "refuse every int" passes the positive one.
                try:
                    _drive(_probe.compare_under)
                    _drive_eager(_probe.arith_under)
                except NarrowingError as exc:
                    perimeter_control_state = "cries-wolf"
                    perimeter_control = (
                        f"an exactly representable {exc.finding.literal} was "
                        "refused"
                    )
                else:
                    perimeter_control += (
                        f"; and allowed the exact {_probe.PERIMETER_UNDER} and "
                        f"the in-range {_probe.ARITH_UNDER}"
                    )
        except Exception as exc:  # noqa: BLE001
            perimeter_control_state = "raised"
            perimeter_control = f"raised {type(exc).__name__}: {exc}"

    from stelling._tripwire import perimeter as _perimeter_mod

    perimeter_rows: list = []
    perimeter_moved: list = []
    for _face in _perimeter_mod.FACES:
        _rows, _moved = _perimeter_facts(_face, adapter.perimeter_locate(_face))
        perimeter_rows.extend(_rows)
        perimeter_moved.extend(_moved)
    perimeter_promotion_note, perimeter_promotion_drift = _perimeter_promotion()

    # THE OTHER MAP, RE-DERIVED, AND IT RUNS WHILE THE HOOK IS STILL LIVE.
    # `_eager_sweep_row` needs the attached wrapper -- it swaps a collector in
    # for the observer and puts it back -- so it cannot go after the disarm
    # below. It displaces nothing, which is why the one displacement question
    # underneath it still answers for both hooks.
    eager_sweep_note, eager_sweep_reason = _eager_sweep_row(
        eager_status.armed, args.sweep
    )

    # ONE QUESTION, BOTH HOOKS, and it is asked BEFORE either disarm because
    # disarming empties the record the question is asked of: afterwards there
    # is no armed hook left to be displaced. `_tripwire.displaced()` reports
    # only hooks THIS PROCESS ARMED, so with both live it answers for both --
    # which is the whole reason B15's finding and this hook share one
    # instrument rather than getting one each.
    displaced = _tripwire.displaced()
    disarmed = _tripwire.disarm()
    eager_disarmed = _tripwire.disarm_eager()
    perimeter_disarmed = _tripwire.disarm_perimeter("canary")

    hash_note, hash_reason = _hash_row(status)
    eager_hash_note, eager_hash_reason = _eager_hash_row(eager_status)

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
    reasons.extend(
        _eager_reasons(
            eager_status, eager_control_state, eager_control,
            displaced, args.require,
        )
    )
    if eager_hash_reason is not None:
        reasons.append(eager_hash_reason)
    if eager_sweep_reason is not None:
        reasons.append(eager_sweep_reason)
    reasons.extend(
        _perimeter_reasons(
            perimeter_status, perimeter_control_state, perimeter_control,
            perimeter_moved, perimeter_promotion_drift, args.require,
        )
    )

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
        # --- the eager construction-site detector, same three-part shape ---
        ("eager status", eager_status.code),
        ("eager site", eager_status.rule_name or "?"),
        ("eager site sha1", f"{eager_status.rule_hash or '?'} ({eager_hash_note})"),
        ("eager jax constants", eager_sweep_note),
        ("eager control state", eager_control_state),
        ("eager live control", eager_control),
        ("displaced hooks", ", ".join(displaced) or "none"),
        ("disarm_eager()", eager_disarmed),
        ("eager detail", eager_status.explanation),
        # --- the dunder perimeter, same three-part shape ---
        ("perimeter status", perimeter_status.code),
        ("perimeter control state", perimeter_control_state),
        ("perimeter live control", perimeter_control),
        ("perimeter promotion identity", perimeter_promotion_note),
        *perimeter_rows,
        ("disarm_perimeter()", perimeter_disarmed),
        ("perimeter detail", perimeter_status.explanation),
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
