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
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

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
         verbose=False, extra_args=()):
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
    wanted = [pc.by_name(args.control)] if args.control else list(pc.CONTROLS)
    failures = []
    for control in wanted:
        if control.series == "both" and not args.other_python:
            print(f"-- {control.name}: SKIPPED (needs --other-python, an "
                  f"interpreter with the other jax series)")
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
            proc = _run(tree, [control.nodeid], python=args.python,
                        profile=args.profile, scale=scale,
                        runxfail=True, verbose=args.verbose,
                        extra_env=_cross_env(args))
            out = (proc.stdout or "") + (proc.stderr or "")
            fired = proc.returncode != 0
            carried = control.expect_message in out
            if fired and carried:
                print("   FIRED — the property failed where it is supposed to")
                if args.verbose:
                    print(_tail(proc, 30))
            elif fired:
                print(f"   FIRED, but the failure did not carry "
                      f"{control.expect_message!r}")
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
    p.add_argument("--control", help="run one positive control by name")
    p.add_argument("--select", action="append",
                   help="pytest target(s); default is the whole property suite")
    p.add_argument("--profile", default="ci", choices=("ci", "dev", "nightly"))
    p.add_argument("--scale", default="1.0")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter with hypothesis and jax installed")
    p.add_argument("--other-python",
                   help="interpreter with the OTHER jax series, for the "
                        "cross-series property")
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
