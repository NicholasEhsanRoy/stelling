# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The stamp contract — the banked can't-drift instance, landed.

Two were banked (`design/semantics-drift-probe.md` bookkeeping, SOUNDNESS
commitments): *never invoke a solver on defaults* (lands with the first
solver call — still banked) and *never emit a verdict without a complete
stamp* (lands with the first verdict — this file). Every field of
SOUNDNESS.md's contract must be populated; a missing field fails loudly at
construction rather than defaulting.
"""

from __future__ import annotations

import dataclasses

import pytest

from stelling import ir
from stelling.propagate import propagate
from stelling.verdict import (
    Stamp,
    StampError,
    Verdict,
    make_verdict,
    solver_absent,
)
from test_propagate import exp_lt_harness


def full_stamp_kwargs():
    return dict(
        stelling_version="0.1.0",
        jax_version="0.11.0",
        query_content_hash="deadbeef",
        arithmetic_mode="interval/f64/outward-1ulp",
        precision_config="jax_enable_x64=True",
        device_class="none: no concrete execution in this verdict",
        solver=solver_absent("interval arithmetic discharged everything"),
        transfer_tiers=(("exp", "sound-libm"),),
        transfer_provenance=(("exp", "core"),),
        assumptions=("libm exp faithful",),
        coverage="4 eqns: 4 known (100%)",
    )


def test_every_contract_field_is_required():
    for missing in full_stamp_kwargs():
        kwargs = full_stamp_kwargs()
        del kwargs[missing]
        with pytest.raises(TypeError):
            Stamp(**kwargs)


def test_empty_fields_fail_loudly_instead_of_defaulting():
    for field in (
        "stelling_version",
        "jax_version",
        "query_content_hash",
        "arithmetic_mode",
        "precision_config",
        "device_class",
        "coverage",
    ):
        kwargs = full_stamp_kwargs()
        kwargs[field] = ""
        with pytest.raises(StampError):
            Stamp(**kwargs)


def test_solver_absence_is_recorded_not_implied():
    with pytest.raises(StampError):
        solver_absent("")  # absence without a reason is not recorded absence
    from stelling.verdict import SolverStamp

    with pytest.raises(StampError):  # invoked without the emitted option set
        SolverStamp(
            invoked=True, reason="z3 on the residual goal",
            name="z3", version="4.13", transport="wheel", options=None,
        )
    with pytest.raises(StampError):  # absent but carrying solver fields
        SolverStamp(
            invoked=False, reason="not invoked",
            name="z3", version=None, transport=None, options=None,
        )


def test_first_verdict_carries_the_full_stamp():
    closed = exp_lt_harness(8.0)
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"
    assert v.stamp.query_content_hash == closed.content_hash()
    assert v.stamp.solver.invoked is False and v.stamp.solver.reason
    assert ("exp", "sound-libm") in v.stamp.transfer_tiers
    assert all(origin == "core" for _, origin in v.stamp.transfer_provenance)
    rendered = v.render()
    for needle in ("VERIFIED", "solver: none", "assumes:", "coverage:", "query "):
        assert needle in rendered
    # the stamp is frozen: no post-hoc field surgery
    with pytest.raises(dataclasses.FrozenInstanceError):
        object.__delattr__  # placate linters; the real check:
        v.stamp.jax_version = "other"  # type: ignore[misc]


def test_undischarged_obligations_never_verify():
    closed = exp_lt_harness(2.0)
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "UNKNOWN"  # definite violation reports, never refutes
    assert isinstance(v, Verdict)
    assert v.obligations[0].status == "violated-over-set"
