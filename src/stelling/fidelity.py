# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The fidelity gauging loop — measured discriminating power, shipped.

The L21 rule, which this module is the structural form of: **a fidelity
check pins only what it can distinguish**. A gate stack's discriminating
power is not a property you assert — it is a thing you MEASURE, by
running the stack against deliberately wrong variants (mutations) of the
subject it claims to pin; a mutation that survives every gate is either
a measured hole in the stack or a value-identical variant whose survival
has an explanation that is itself a checkable claim. This module refuses
to bless a stack that has not been measured: no battery, a failing
baseline, an unexplained survivor, or a stale survivor explanation each
refuse loudly instead of returning a report.

Everything here is pure orchestration: zero dependencies, jax-free at
import, and deliberately ignorant of what a "subject" is. Gates are
caller callables ``gate(subject) -> bool`` (True = the subject passes;
False = the gate CAUGHT the subject) and may close over anything —
meshes, jax pipelines, solver oracles. A gate that raises is a broken
gate, not a catch: exceptions propagate to the caller undisturbed (wrap
the gate if, for your stack, "blows up on the mutant" should count as
caught).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FidelityReport", "gauge"]


@dataclass(frozen=True)
class FidelityReport:
    """The measured result of one gauging run.

    ``gates``: the gate names, in stack order — the baseline passed every
    one of them (a report is only constructed past that refusal).
    ``caught_by``: the per-mutation table, in battery order — each entry
    is ``(mutation_name, tuple_of_gate_names_that_caught_it)``; an empty
    tuple marks a survivor, and every survivor is residual-listed (the
    gauge refused otherwise). ``residual``: the survivors' explanations,
    ``(mutation_name, explanation)``, in battery order.

    ``scope``: what this gauge's gates actually REACH, in the author's own
    words. Carried into :meth:`render` because a gauge's zero is only
    meaningful against the surface it drove — see the "instrument must declare
    its scope" norm in ``CONTRIBUTING.md``, whose fifth instance was a scatter
    gauge reporting zero survivors while never running the interval transfer
    the change under test had altered.
    """

    gates: tuple[str, ...]
    caught_by: tuple[tuple[str, tuple[str, ...]], ...]
    residual: tuple[tuple[str, str], ...]
    scope: str

    def render(self) -> str:
        lines = [
            "== fidelity gauge — measured discriminating power (L21)",
            f"SCOPE — what these gates reach: {self.scope}",
            f"baseline: passes every gate ({', '.join(self.gates)})",
        ]
        width = max((len(name) for name, _ in self.caught_by), default=0)
        for name, gates in self.caught_by:
            if gates:
                lines.append(f"  {name:<{width}s}  CAUGHT by {', '.join(gates)}")
            else:
                lines.append(
                    f"  {name:<{width}s}  survives every gate (residual, "
                    f"explained below)"
                )
        lines.append(
            "== residual class (value-identical to the gauged stack; the "
            "explanation is the claim) =="
        )
        if not self.residual:
            lines.append("  (empty: every mutation was caught)")
        for name, why in self.residual:
            lines.append(f"  {name}: {why}")
        lines.append(
            f"== every count above is scoped to: {self.scope} — a surface "
            f"these gates do not drive is NOT gauged by this report =="
        )
        return "\n".join(lines)


def gauge(baseline, gates, mutations, *, residual, scope):
    """Run every gate against the baseline and every mutation; return the
    measured :class:`FidelityReport`, or refuse loudly.

    ``baseline``: the unmutated subject. ``gates``: mapping ``name ->
    callable(subject) -> bool`` (True = passes). ``mutations``: mapping
    ``name -> subject``, each a deliberately wrong variant. ``residual``:
    mapping ``mutation name -> explanation`` for exactly the mutations
    expected to survive every gate (value-identical variants); the
    explanation is a measurable claim, and this function measures it.

    Structural refusals — the point of the module, not conveniences:

    * **no battery**: empty ``mutations`` — a gate stack with no battery
      is ungauged;
    * **no gates**: an empty stack gauges nothing;
    * **invalid stack**: the baseline fails a gate — the stack is wrong
      about the true subject and its catches mean nothing;
    * **unexplained survivor**: a mutation caught by NO gate and absent
      from ``residual`` — a measured hole nobody has owned;
    * **stale residual claim**: a ``residual``-listed mutation that IS
      caught — the explanation is measurably wrong and must not ride
      along as documentation;
    * **unknown residual name**: a ``residual`` entry naming no known
      mutation.

    Gate exceptions propagate (see the module docstring).
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(
            "fidelity.gauge: no scope — state what these gates REACH, and "
            "what they do not. A gauge's zero is a claim about the surface it "
            "drove, and a report that does not name that surface gets quoted "
            "past it; this argument is required and undefaulted so omitting it "
            "is an error rather than a silent empty claim"
        )
    gate_items = tuple(gates.items()) if gates is not None else ()
    if not gate_items:
        raise ValueError(
            "fidelity.gauge: no gates — an empty stack gauges nothing; "
            "there is no discriminating power to measure"
        )
    for name, gate in gate_items:
        if not callable(gate):
            raise ValueError(
                f"fidelity.gauge: gate {name!r} is not callable "
                f"({type(gate).__name__}); a gate is "
                f"callable(subject) -> bool"
            )
    mutation_items = tuple(mutations.items()) if mutations is not None else ()
    if not mutation_items:
        raise ValueError(
            "fidelity.gauge: no mutations — a gate stack with no battery "
            "is ungauged; discriminating power is measured, not assumed"
        )
    residual_map = dict(residual) if residual is not None else {}
    mutation_names = {name for name, _ in mutation_items}
    unknown = [name for name in residual_map if name not in mutation_names]
    if unknown:
        raise ValueError(
            f"fidelity.gauge: residual entr{'ies' if len(unknown) > 1 else 'y'} "
            f"{sorted(unknown)!r} name no known mutation — an explanation "
            f"with no measured subject explains nothing"
        )
    for name, why in residual_map.items():
        if not isinstance(why, str) or not why.strip():
            raise ValueError(
                f"fidelity.gauge: residual entry {name!r} carries no "
                f"explanation — a survivor's explanation is the claim, and "
                f"an empty one is not a claim"
            )

    baseline_failures = [
        name for name, gate in gate_items if not gate(baseline)
    ]
    if baseline_failures:
        raise ValueError(
            f"fidelity.gauge: the BASELINE fails gate(s) "
            f"{baseline_failures!r} — the stack is invalid: a gate the "
            f"true subject cannot pass measures nothing about mutations"
        )

    caught_by = []
    for mut_name, subject in mutation_items:
        catchers = tuple(
            gate_name for gate_name, gate in gate_items if not gate(subject)
        )
        caught_by.append((mut_name, catchers))

    survivors = [name for name, catchers in caught_by if not catchers]
    unexplained = [name for name in survivors if name not in residual_map]
    if unexplained:
        raise ValueError(
            f"fidelity.gauge: mutation(s) {unexplained!r} survive every "
            f"gate and are not residual-listed — an unexplained survivor "
            f"is a measured hole in the stack; either add a gate that "
            f"distinguishes it or own the survival with a residual "
            f"explanation"
        )
    stale = [
        (name, dict(caught_by)[name])
        for name in residual_map
        if dict(caught_by)[name]
    ]
    if stale:
        detail = "; ".join(
            f"{name!r} caught by {list(catchers)!r}" for name, catchers in stale
        )
        raise ValueError(
            f"fidelity.gauge: stale residual claim(s): {detail} — a "
            f"residual explanation asserts the mutation is value-identical "
            f"to the stack, and the measurement just falsified it; delete "
            f"or correct the entry"
        )

    return FidelityReport(
        gates=tuple(name for name, _ in gate_items),
        caught_by=tuple(caught_by),
        residual=tuple(
            (name, residual_map[name])
            for name, catchers in caught_by
            if not catchers
        ),
        scope=scope.strip(),
    )
