# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The control-flow hypothesis run — the four control-flow-only properties.

Each is posed as a loop-body invariant: declare the loop carry as a box
(the candidate invariant), model one loop-body step faithfully (constants
from the thread), and check the body maps the box into itself — discrete
inductiveness, the E2a edge-flux pattern for a discrete recursion. Branch
logic uses the new `cond`/`select_n`/`max` transfers. No discrete-step
model (`design/control-flow-hypothesis.md`).

Reported with the relation breakdown per the reporting rule.
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import stelling
from stelling._jax_compat import jax_version, trace
from stelling.harness import any_array, assert_, nonvacuity
from stelling.propagate import propagate
from stelling.verdict import make_verdict


def verdict(h):
    cj = trace(h)
    return make_verdict(
        cj, propagate(cj),
        stelling_version=stelling.__version__,
        jax_version=jax_version(),
        precision_config="jax_enable_x64=True",
    )


def show(name, v, note):
    print(f"==== {name}: {v.status}")
    print(f"     nonvacuity: {v.stamp.nonvacuity.split(' — ')[0]}; "
          f"coverage: {v.stamp.coverage}")
    for o in v.obligations:
        print(f"     assert #{o.index}: {o.status}")
    print(f"     relation: {note}")
    print()
    return v


# --- dfx#207: accepted dt >= dt_min through the controller loop -------------
# Body: propose dt from the error ratio, then clamp to the dtmin floor
# (diffrax PIDController.dtmin). Invariant carry: dt in [DT_MIN, DT_MAX].
DT_MIN, DT_MAX = 1e-6, 1.0


def h_207():
    dt = any_array((), "float64", (DT_MIN, DT_MAX))   # carry, candidate invariant
    ratio = any_array((), "float64", (0.0, 100.0))    # error ratio (arbitrary >= 0)
    dt_prop = dt * jnp.clip(ratio, 0.1, 5.0)          # PID proposal (bounded factor)
    dt_new = jnp.maximum(dt_prop, DT_MIN)             # the dtmin clamp -> `max`
    o = assert_(dt_new >= DT_MIN)                      # invariant preserved
    dt0 = any_array((), "float64", (1e-3, 1e-3))       # a valid initial dt
    return o, nonvacuity(dt0 >= DT_MIN), nonvacuity(dt0 <= DT_MAX)


v207 = show("dfx#207", verdict(h_207),
            "DISCHARGES the registered property (accepted dt >= dt_min) via the "
            "dtmin clamp; property-incident gap: the max_steps collapse is a "
            "step-COUNT bound the property does not reach")

# --- npy#249: step_size > 0 throughout warmup -------------------------------
# Body: step_size = exp(log_eps) after a dual-averaging update. exp is
# strictly positive, so `> 0` is a loop-body invariant regardless of the
# update. `isfinite` is the float-specific half (R-partial), not posed.
def h_249():
    log_eps = any_array((), "float64", (-20.0, 5.0))   # carry (bounded log step)
    grad = any_array((), "float64", (-10.0, 10.0))
    log_eps_new = log_eps - 0.05 * grad                # dual-averaging step
    step_size = jnp.exp(log_eps_new)
    o = assert_(step_size > 0.0)                        # exp is positive
    le0 = any_array((), "float64", (-1.0, -1.0))
    return o, nonvacuity(le0 >= -20.0), nonvacuity(le0 <= 5.0)


v249 = show("npy#249", verdict(h_249),
            "PRECONDITION with the gap named: discharges step_size > 0 (exp is "
            "positive); the isfinite conjunct is float-specific (R-partial) and "
            "out of the registered semantics")

# --- bjx#969: non-finite proposal => step_size_max shrinks ------------------
# Body: step_size_max_new = where(non_finite, step_size_max * 0.8, step_size_max).
# Pose the implication by declaring the antecedent true (non_finite) and
# checking the consequent (strict shrink) via the select_n branch.
def h_969():
    s_max = any_array((), "float64", (0.01, 100.0))
    nf = any_array((), "bool", (1.0, 1.0))             # antecedent asserted TRUE
    s_new = jnp.where(nf, s_max * 0.8, s_max)          # select_n
    o = assert_(s_new < s_max)                         # shrinks (0.8 < 1)
    s0 = any_array((), "float64", (1.0, 1.0))
    return o, nonvacuity(s0 >= 0.01), nonvacuity(s0 <= 100.0)


v969 = show("bjx#969", verdict(h_969),
            "does NOT mechanize: the shrink `0.8*s_max < s_max` is "
            "DEPENDENCY-SHAPED (s_new derived from s_max; interval loses the "
            "correlation, same shape as the dfx#632 exhibit) -> UNKNOWN, an "
            "affine-forms case, NOT the search-shaped the prior predicted")

# --- bjx#D416: adapted step_size >= epsilon (NO clamp) ----------------------
# Body: dual-averaging update with no floor. step_size >= eps is NOT
# preserved by the body (this is why the incident happened). Posed to show
# it does not mechanize.
EPS = 1e-3


LOG_EPS = math.log(EPS)


def h_D416():
    log_eps = any_array((), "float64", (LOG_EPS, 5.0))       # carry >= log eps
    h_stat = any_array((), "float64", (-5.0, 5.0))           # adaptation signal
    log_eps_new = log_eps - 0.1 * h_stat                     # no clamp
    step_size = jnp.exp(log_eps_new)
    o = assert_(step_size >= EPS)                             # NOT a true invariant
    le0 = any_array((), "float64", (LOG_EPS, LOG_EPS))
    return o, nonvacuity(le0 >= LOG_EPS)


vD416 = show("bjx#D416", verdict(h_D416),
             "does NOT mechanize: step_size >= eps is not preserved without a "
             "floor clamp (the incident's own cause); counts 0 (also the incident "
             "mechanism is RNG key reuse, a killed category out of scope)")

mech = sum(v.status == "VERIFIED" for v in (v207, v249, v969, vD416))
print(f"control-flow-only mechanized: {mech} of 4")
