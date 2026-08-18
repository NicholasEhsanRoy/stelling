# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`from_dict` must refuse an equation missing a param jax always supplies.

The defect this pins. Readers were taught to test key PRESENCE rather than
`.get()`, because for `scatter`/`scatter-add` a key present with value `None`
is jax's REPLACE combiner while an ABSENT key is the hand-built form, where the
primitive name is the semantic authority. Correct for a traced equation and for
hand-built IR — and wrong for a DESERIALIZED one, because it turns DELETION
INTO BLESSING.

An earlier audit concluded that an absent key is structurally unreachable from
jax. That is true of `bind` and false of `from_dict`, and this is the door that
distinction left open.
"""
from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp
import numpy as np

from stelling import ir
from stelling._jax_compat import transcribe
from stelling.interval import IntervalArray
from stelling.propagate import _t_scatter, _t_scatter_add


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _persisted_without(fn, args, primitive, key):
    """Trace `fn`, persist through real JSON, delete `key` from `primitive`."""
    d = json.loads(json.dumps(transcribe(jax.make_jaxpr(fn)(*args)).to_dict()))
    dropped = 0
    for eq in d["jaxpr"]["eqns"]:
        if eq["primitive"] == primitive:
            before = len(eq["params"])
            eq["params"] = [p for p in eq["params"] if p[0] != key]
            dropped += before - len(eq["params"])
    assert dropped == 1, f"fixture did not remove exactly one {key!r}"
    return d


@pytest.mark.parametrize(
    "kind,primitive",
    [("set", "scatter"), ("add", "scatter-add")],
    ids=["scatter", "scatter-add"],
)
def test_deleting_update_jaxpr_from_a_persisted_query_is_refused(kind, primitive):
    def build(x, v):
        return x.at[1].set(v) if kind == "set" else x.at[1].add(v)

    d = _persisted_without(
        build, (jnp.zeros(4, jnp.float64), jnp.float64(1.0)),
        primitive, "update_jaxpr")

    with pytest.raises(ir.TranscriptionError) as e:
        ir.ClosedJaxpr.from_dict(d)
    assert "update_jaxpr" in str(e.value)


def test_the_round_trip_that_looks_like_this_one_but_is_not():
    """A plain round trip must still load, and must preserve key ABSENCE vs
    presence-with-`None`.

    Kept because testing the round trip is what made this defect look
    unreproducible: the round trip is faithful, and the claim was about a
    persisted query someone EDITED.
    """
    def build(x, v):
        return x.at[1].add(v)

    cj = transcribe(jax.make_jaxpr(build)(jnp.zeros(4, jnp.float64),
                                          jnp.float64(1.0)))
    back = ir.ClosedJaxpr.from_dict(json.loads(json.dumps(cj.to_dict())))
    assert back.content_hash() == cj.content_hash()


def test_the_transfer_would_have_produced_a_box_excluding_the_truth():
    """The consequence, pinned so the refusal above is known to be load-bearing.

    Built by hand at exactly the state `from_dict` used to hand back — this is
    the blessed hand-built form, so the transfer admits it and models a `.set`.
    jax's `.apply(max)` computes something else, and the box excludes it. What
    changed is that this state is no longer REACHABLE from a persisted query.
    """
    def build(x):
        return x.at[1].apply(lambda a: jnp.maximum(a, 2.0))

    cj = transcribe(jax.make_jaxpr(build)(jnp.zeros(4, jnp.float64)))
    scat = [e for e in cj.jaxpr.eqns if str(e.primitive) == "scatter"][0]
    stripped = ir.JaxprEqn(
        primitive=scat.primitive, invars=scat.invars, outvars=scat.outvars,
        params=tuple((k, v) for k, v in scat.params if k != "update_jaxpr"),
        effects=scat.effects, source_info=scat.source_info,
    )

    operand = (0.0, 5.0, 0.0, 0.0)  # element 1 is 5.0: max(5, 2) = 5, a set gives 2
    box = _t_scatter(stripped, dict(stripped.params_dict()), [
        IntervalArray(shape=(4,), los=operand, his=operand),
        IntervalArray(shape=(1,), los=(1.0,), his=(1.0,)),
        IntervalArray(shape=(), los=(2.0,), his=(2.0,)),
    ])
    assert box is not None, "the stripped form is the blessed hand-built one"

    truth = np.asarray(build(jnp.asarray(operand, jnp.float64)))
    excluded = [i for i in range(4)
                if not box[0].los[i] <= truth[i] <= box[0].his[i]]
    assert excluded == [1], (
        "this is the unsoundness the load-path refusal exists to make "
        "unreachable; if it has gone away, the reason has changed and the "
        "refusal's justification needs rewriting rather than deleting"
    )


def test_a_traced_equation_carries_every_required_param():
    """The refusal is only correct if jax really does always supply them."""
    def build(x, v):
        return x.at[1].add(v), x.at[2].set(v), jnp.sqrt(jnp.abs(x) + 1.0)

    cj = transcribe(jax.make_jaxpr(build)(jnp.zeros(4, jnp.float64),
                                          jnp.float64(1.0)))
    seen = 0
    for e in cj.jaxpr.eqns:
        required = ir._REQUIRED_PARAMS.get(str(e.primitive))
        if required is None:
            continue
        seen += 1
        assert not required - {k for k, _ in e.params}, str(e.primitive)
    assert seen >= 3, "fixture should exercise several constrained primitives"


def test_a_param_only_the_NEWEST_release_supplies_gets_no_row():
    """The other side of the refusal, and jax 0.11.1 is why it is a test.

    A row in `_REQUIRED_PARAMS` is a refusal aimed at a STORED document, and
    a document is loaded by a different jax from the one that traced it —
    that is the whole reason documents exist. So a key may only be required
    once EVERY release a document could have come from supplies it.

    jax 0.11.1 added `out_sharding` to `reduce_max` and `reduce_min`, which
    0.11.0 does not emit. Re-driving the census on both releases makes the
    diff visible and the obvious next move is to add the rows. Measured, that
    move refuses an honest `jnp.max` document traced on 0.11.0 — on BOTH
    releases, because the refusal reads the document and not the running jax.
    This test is the fence against making it: it fails the day someone adds
    a row for either primitive, and says why in the message.

    It is not a claim that the two are unmodelled or unimportant. It is a
    claim about WHEN a key becomes requirable, and the answer is "when no
    supported release omits it", not "when the newest release supplies it".
    """
    for name in ("reduce_max", "reduce_min"):
        assert name not in ir._REQUIRED_PARAMS, (
            f"`{name}` gained a row in _REQUIRED_PARAMS. jax 0.11.1 added "
            f"`out_sharding` to it and jax 0.11.0 does not emit it, so a row "
            f"here refuses every `{name}` document written on 0.11.0 or "
            "earlier — measured: TranscriptionError at load, on both "
            "releases. Requiring a key is only sound once no supported "
            "release omits it. See the comment above _REQUIRED_PARAMS in "
            "src/stelling/ir.py and the 2026-08-18 entry in SOUNDNESS.md."
        )
    # and the control: the table is not simply empty, and `reduce_sum` — the
    # reduction whose `out_sharding` IS present on every release read so far —
    # does carry it, so the distinction above is being drawn and not dodged.
    assert ir._REQUIRED_PARAMS["reduce_sum"] == frozenset({"axes", "out_sharding"})


def test_scatter_add_transfer_still_admits_the_traced_form():
    """A positive control: the refusal must not have made the row unreachable."""
    def build(x, v):
        return x.at[1].add(v)

    cj = transcribe(jax.make_jaxpr(build)(jnp.zeros(4, jnp.float64),
                                          jnp.float64(1.0)))
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == "scatter-add"][0]
    box = _t_scatter_add(eqn, dict(eqn.params_dict()), [
        IntervalArray(shape=(4,), los=(0.0,) * 4, his=(1.0,) * 4),
        IntervalArray(shape=(1,), los=(1.0,), his=(1.0,)),
        IntervalArray(shape=(), los=(2.0,), his=(2.0,)),
    ])
    assert box is not None
    # accumulate, outward-rounded: the true range at element 1 is [2, 3] and
    # the box must contain it without being asserted equal to it — this row is
    # tier `sound`, so one outward bump per real is expected.
    assert box[0].los[1] <= 2.0 and box[0].his[1] >= 3.0
    assert box[0].his[1] < 3.001, "outward bump, not a widened box"
