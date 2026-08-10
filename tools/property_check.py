# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Point the property suite at an arbitrary tree, revision, or mutant.

    python tools/property_check.py --tree /path/to/some/worktree
    python tools/property_check.py --rev  fb34e0d
    python tools/property_check.py --controls
    python tools/property_check.py --control widen -v

**The separation this tool exists for**: the PROPERTIES come from the checkout
this file lives in; the CODE UNDER TEST comes from wherever you point it. They
are joined by ``PYTHONPATH=<tree>/src`` and nothing else — no install, no
editable wheel, no worktree of the tests. That is what makes a property usable
as an independent third check on somebody else's branch: you do not have to
merge, rebase, or trust their copy of the tests.

It is also the whole mechanism behind ``--controls``. A positive control runs
**today's property** against **yesterday's defect** and asserts the run comes
back RED. Without that, a green property is indistinguishable from a property
whose strategy generates nothing — the failure mode this project has shipped
more than any other.

MATERIALISING A TREE. ``--rev`` uses ``git archive``, not ``git worktree``:
this repository is worked on by several agents at once, and ``git archive``
touches nothing under ``.git`` while ``git worktree add`` writes state that
somebody else's ``git worktree list`` then has to reason about. Only ``src/``
is extracted, because that is all ``PYTHONPATH`` needs.

EXIT CODE. 0 if every requested run had the outcome it was supposed to have.
For ``--controls`` that means every control FAILED where it was supposed to
fail; a control that passes is reported as ``CONTROL DID NOT FIRE`` and exits
non-zero, because a control that cannot demonstrate its property is worth
nothing.

WHERE ``expect_message`` IS MATCHED, and why it is not the captured output.
pytest's long traceback prints a frame's ENTIRE function source — docstring
included — from ``def`` down to the failing line. So the strings a property
BUILDS its failure messages from, and any string quoted in the docstring of a
function on the traceback path, are echoed into the run's output whether or
not the property's oracle ever ran. Matching ``expect_message`` against that
output makes a CRASH indistinguishable from a demonstration. Measured, on
``tests/property/test_cvc5_protocol.py`` with its two clause sentences quoted
in ``_judge``'s docstring: a one-place defect that raises ``TypeError`` before
the oracle is evaluated scored ``3/3 controls fired`` against three probe
controls carrying the three shipped guard strings. So the match is made
against the failure pytest itself RECORDS — the ``message`` attribute of each
``<failure>``/``<error>`` in ``--junitxml``. The same run now scores ``0/3``.

WHAT THAT ATTRIBUTE ACTUALLY HOLDS. It was described here as "the crash line
and the exception's own text and nothing else", and that is not what it is.
Measured on pytest 9.1.1 with hypothesis 6.165.2, reading one real failure of
``test_the_parent_never_trusts_an_unspoken_transcript_flat`` out of the XML::

    AssertionError: <clause (2)'s sentence> as 'unknown' [flat]  <- the
      read : '…'                                             exception's text
      full : '…'
    Failing test case: search(                              <- hypothesis note
        item=('…', '…', 0),
    )
    Explanation:                                            <- hypothesis note
        These lines were always and only run by failing test cases:
            …/tests/property/test_cvc5_protocol.py:443
    You can reproduce this test case by temporarily adding
    @reproduce_failure('6.165.2', b'…') as a decorator …    <- print_blob=True

and, where the failure is a bare ``assert``, pytest's assertion-rewrite
explanation as well — which embeds the **reprs of the asserted expression's
operands**, i.e. generated data. So the attribute carries strategy output, and
whether some generator could draw a guard string into it is a per-guard
question rather than something this file settles by construction. Of the twelve
guards registered today, eleven are shouted English phrases or bracketed leg
tags; ``reorder``'s ``transposed`` is the one lower-case word, and it is the
one to look at first if that question is ever asked in earnest. NOTHING here
is exploitable by any guard registered today.

THE BLIND SPOT THIS RULE HAS AND THE OUTPUT MATCH DID NOT. When hypothesis
finds MORE THAN ONE distinct failure it raises an ``ExceptionGroup``, and the
sub-exceptions' texts live in the traceback BODY only. Driven, two assert
sites in one ``@given`` function::

    <failure message="ExceptionGroup: Hypothesis found 2 distinct failures.
                      (2 sub-exceptions)">

Neither sentinel appears in the attribute. So a genuine demonstration that
found two defects at once is scored "wrong failure" here, where the old
output match would have scored it FIRED. Not live for any of the twelve today
— every one of them shrinks to a single interesting origin — but it is not far
off. Counted over the suite's ``@given`` functions: the reordering property
and the conjunct property have THREE raise/assert sites each, the wrap-class
oracle leg and the refutation property TWO each, and ``reorder``'s own guard
``transposed`` is written at two of the reordering property's three. A
second interesting origin is one mutant away, and the failure mode is a
control reported NOT DEMONSTRATED while demonstrating twice over. That is the
SAFE direction — it refuses, it never passes — which is why it is disclosed
here rather than worked around: reading sub-exception texts out of the
traceback body would put the guard back in reach of the property's own echoed
source, which is the defect this whole rule exists to close.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PROPERTY_DIR = REPO / "tests" / "property"

sys.path.insert(0, str(PROPERTY_DIR))
import positive_controls as pc  # noqa: E402


# ── materialising the tree under test ────────────────────────────────────────


def _materialise(rev: str, into: pathlib.Path, repo: pathlib.Path) -> pathlib.Path:
    """Put ``<rev>``'s ``src/`` under ``into`` and return the tree root."""
    into.mkdir(parents=True, exist_ok=True)
    if rev in ("HEAD", "WORKTREE"):
        # The WORKING TREE's src, not the committed one: a control run while
        # you are editing the remedy must test what you are editing.
        shutil.copytree(repo / "src", into / "src")
        return into
    tar = subprocess.run(
        ["git", "-C", str(repo), "archive", rev, "src"],
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=tar, check=True)
    return into


def _apply(mutation, tree: pathlib.Path) -> None:
    path = tree / mutation.path
    text = path.read_text()
    n = text.count(mutation.old)
    if n != 1:
        raise SystemExit(
            f"MUTATION DID NOT APPLY: {mutation.path} contains "
            f"{n} occurrences of the target text, expected exactly 1.\n"
            f"  looking for: {mutation.old!r}\n"
            "The registry has drifted from the source. Fix the registry — a "
            "control that silently stops mutating is a control that always "
            "passes."
        )
    path.write_text(text.replace(mutation.old, mutation.new))


# ── running ──────────────────────────────────────────────────────────────────


def _run(tree, targets, *, python, profile, scale, extra_env=None, runxfail=False,
         verbose=False, extra_args=(), junit=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(tree) / "src")
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
    env["STELLING_PROPERTY_PROFILE"] = profile
    env["STELLING_PROPERTY_SCALE"] = str(scale)
    env.pop("STELLING_PROPERTY_DB", None)
    if extra_env:
        env.update(extra_env)
    argv = [
        python, "-m", "pytest", "-ra", "-p", "no:cacheprovider",
        "--no-header", *extra_args,
    ]
    if runxfail:
        argv.append("--runxfail")
    if junit is not None:
        argv.append(f"--junitxml={junit}")
    argv += list(targets)
    if verbose:
        print("   $ PYTHONPATH=%s STELLING_PROPERTY_PROFILE=%s %s"
              % (env["PYTHONPATH"], profile, " ".join(argv)))
    proc = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True)
    return proc


def _tail(proc, n=25):
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = out.strip().splitlines()
    return "\n".join("   | " + ln for ln in lines[-n:])


def _crashes(junit) -> list[str]:
    """What the run REPORTED as its failures, as pytest itself records them.

    One string per ``<failure>``/``<error>``: pytest's ``message`` attribute,
    and NOT the traceback body. That distinction is the whole point (see this
    module's docstring): the traceback body carries the property's own source,
    so a guard string is present there even when nothing evaluated the oracle.

    IT IS NOT "THE CRASH LINE AND NOTHING ELSE", which is what this said. The
    attribute is the exception type and its own text, PLUS pytest's assertion
    -rewrite explanation where the failure is a bare ``assert`` (which embeds
    the reprs of the asserted expression's operands, i.e. generated data),
    PLUS hypothesis's own notes — ``Failing test case: …``, the
    ``Explanation:`` file:line list, and the ``@reproduce_failure`` blob that
    ``print_blob=True`` in ``_profiles.py`` asks for. Measured, verbatim, in
    this module's docstring.

    AND ONE SHAPE IT CANNOT SEE: when hypothesis finds more than one distinct
    failure it raises an ``ExceptionGroup``, whose ``message`` is
    ``"Hypothesis found N distinct failures. (N sub-exceptions)"`` — the
    sub-exceptions' texts are in the traceback body alone. A control whose
    property found two defects at once is therefore scored "wrong failure"
    here. Disclosed rather than closed, because the safe direction is to
    refuse; this module's docstring has the driven measurement and the reason.

    An empty list is the honest answer for a run that produced no XML at all —
    a pytest usage error, an interpreter that died before collection — and the
    caller treats it as "did not carry", which is the safe direction.
    """
    try:
        root = ET.parse(junit).getroot()
    except (OSError, ET.ParseError):
        return []
    return [
        child.get("message") or ""
        for case in root.iter("testcase")
        for child in case
        if child.tag in ("failure", "error")
    ]


# ── the two modes ────────────────────────────────────────────────────────────


def check_tree(args) -> int:
    with tempfile.TemporaryDirectory(prefix="stelling-prop-") as tmp:
        if args.tree:
            tree = pathlib.Path(args.tree).resolve()
            label = f"tree {tree}"
        else:
            tree = _materialise(args.rev, pathlib.Path(tmp) / "t",
                                pathlib.Path(args.repo).resolve())
            label = f"rev {args.rev}"
        if args.mutant:
            control = pc.by_name(args.mutant)
            if control.mutation is None:
                raise SystemExit(f"control {args.mutant!r} is not a mutant")
            _apply(control.mutation, tree)
            label += f" + mutant {args.mutant}"
        targets = args.select or [str(PROPERTY_DIR)]
        print(f"== property suite against {label}")
        print(f"   properties from {PROPERTY_DIR}")
        print(f"   profile {args.profile} x{args.scale}, python {args.python}")
        proc = _run(tree, targets, python=args.python, profile=args.profile,
                    scale=args.scale, verbose=True,
                    extra_env=_cross_env(args))
        print(_tail(proc, 40 if proc.returncode else 12))
        print(f"== exit {proc.returncode}")
        return proc.returncode


def _cross_env(args):
    return {"STELLING_PROPERTY_OTHER_PYTHON": args.other_python} \
        if args.other_python else None


def check_controls(args) -> int:
    wanted = ([pc.by_name(n) for n in args.control]
              if args.control else list(pc.CONTROLS))
    failures = []
    for control in wanted:
        if control.series == "both" and not args.other_python:
            print(f"-- {control.name}: SKIPPED (needs --other-python, an "
                  f"interpreter with the other jax series AND hypothesis — "
                  f"tools/property_venv.sh builds one)")
            failures.append((control.name, "not demonstrated: no second series"))
            continue
        with tempfile.TemporaryDirectory(prefix="stelling-ctl-") as tmp:
            tree = _materialise(control.at, pathlib.Path(tmp) / "t",
                                pathlib.Path(args.repo).resolve())
            if control.mutation is not None:
                _apply(control.mutation, tree)
            print(f"-- {control.name}  [{control.kind} {control.at}]")
            print(f"   {control.nodeid}")
            # The control's own scale multiplies the caller's: some searches
            # need more room than a per-push budget, and burying that in a
            # global flag would make every control pay for the slowest one.
            scale = float(args.scale) * control.scale
            junit = pathlib.Path(tmp) / "junit.xml"
            proc = _run(tree, [control.nodeid], python=args.python,
                        profile=args.profile, scale=scale,
                        runxfail=True, verbose=args.verbose,
                        extra_env=_cross_env(args), junit=junit)
            out = (proc.stdout or "") + (proc.stderr or "")
            crashes = _crashes(junit)
            fired = proc.returncode != 0
            carried = any(control.expect_message in c for c in crashes)
            if fired and carried:
                print("   FIRED — the property failed where it is supposed to")
                if args.verbose:
                    print(_tail(proc, 30))
            elif fired and control.expect_message in out:
                # The string is in the run's OUTPUT but not in any failure the
                # run recorded, which is the shape a docstring or a message
                # template echoed by the traceback produces. Reported as its
                # own outcome rather than folded into "wrong failure", because
                # the remedy is different: nothing is wrong with the control.
                print(f"   FIRED, but the failure ITSELF did not carry "
                      f"{control.expect_message!r} — the string is only in the "
                      f"run's echoed output (a traceback prints the property's "
                      f"own source, docstring included)")
                for c in crashes:
                    print(f"   what pytest recorded: {c.splitlines()[0][:120]}")
                print(_tail(proc, 30))
                failures.append((control.name, "echoed, not raised"))
            elif fired:
                print(f"   FIRED, but the failure did not carry "
                      f"{control.expect_message!r}")
                for c in crashes:
                    print(f"   what pytest recorded: {c.splitlines()[0][:120]}")
                print(_tail(proc, 30))
                failures.append((control.name, "wrong failure"))
            else:
                print("   CONTROL DID NOT FIRE — this property cannot be shown "
                      "to detect anything")
                print(_tail(proc, 30))
                failures.append((control.name, "passed where it must fail"))
    print()
    print(f"== {len(wanted) - len(failures)}/{len(wanted)} controls fired")
    for name, why in failures:
        print(f"   NOT DEMONSTRATED: {name} — {why}")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = p.add_mutually_exclusive_group()
    where.add_argument("--tree", help="a checkout to test (uses <tree>/src)")
    where.add_argument("--rev", help="a revision to materialise and test")
    p.add_argument("--repo", default=str(REPO),
                   help="the git repository --rev is read from")
    p.add_argument("--mutant", help="also apply this registered control's mutation")
    p.add_argument("--controls", action="store_true",
                   help="run every positive control and assert each FAILS")
    p.add_argument("--control", action="append",
                   help="run one positive control by name (repeatable)")
    p.add_argument("--select", action="append",
                   help="pytest target(s); default is the whole property suite")
    p.add_argument("--profile", default="ci", choices=("ci", "dev", "nightly"))
    p.add_argument("--scale", default="1.0")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter with hypothesis and jax installed")
    p.add_argument("--other-python",
                   help="interpreter with the OTHER jax series AND hypothesis, "
                        "for the cross-series property (tools/property_venv.sh "
                        "builds one; the child imports _grammar, so a bare jax "
                        "venv fails there rather than differing)")
    p.add_argument("--list", action="store_true", help="list the controls")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for c in pc.CONTROLS:
            print(f"{c.name:20s} {c.kind:7s} {c.at:10s} x{c.scale:<5g} {c.nodeid}")
        return 0
    if args.controls or args.control:
        return check_controls(args)
    if not (args.tree or args.rev):
        args.rev = "HEAD"
    return check_tree(args)


if __name__ == "__main__":
    raise SystemExit(main())
