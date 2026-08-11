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

Exit codes: 0 armed; 1 with `--require` and not armed. Without `--require` it
reports and exits 0, which is the shape a human wants when running it by hand
to see what a given jax does.
"""

from __future__ import annotations

import argparse
import os
import sys


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
    control = "not run"
    if status.armed:
        try:
            from stelling._jax_compat import jax as _jax
            from stelling._jax_compat import jnp as _jnp
            from stelling._tripwire import _probe

            _jax.make_jaxpr(_probe.over)(_jnp.zeros((7,), _jnp.int8))
            found = recorder.sorted_findings()
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
            control = f"raised {type(exc).__name__}: {exc}"

    disarmed = _tripwire.disarm()

    hash_note = "as tested"
    if status.rule_hash and status.rule_hash != adapter._KNOWN_HASH:
        hash_note = f"CHANGED from {adapter._KNOWN_HASH} — read the rule before trusting"

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
                "\nThe rule sha1 is RECORDED and never gated on: a cosmetic "
                "edit upstream must not disable the tool. It is here so a red "
                "run is diagnosable from this page without re-running it.\n"
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
    if status.armed and "DID NOT FIRE" in control:
        print(
            "\ncanary: the tripwire armed and its live control did not fire. "
            "`arm()` says the hook is attached; the control says nothing "
            "reached it. Treat the armed status as unverified.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
