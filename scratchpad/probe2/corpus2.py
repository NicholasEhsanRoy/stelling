# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""probe2 — an independent corpus for the FLOAT half of the member gap.

Built from scratch for the repair of `fix/probe-membership`, and
deliberately NOT `scratchpad/probe/corpus.py`: that corpus declares no
`float32`, `float16` or `bfloat16` at all (its `f_pos`/`f_str`/`f_wide`
are all `float64`), so it is structurally incapable of observing the
residual it reports as zero. Every declaration below that is not a
control is a narrow float, including the one both prior corpora lack —
**a `float32` box wider than the dtype**.

Three rules, each because of how a previous measurement failed here:

1. **The oracle samples MEMBERS, in the declared dtype.** `float32
   (-1e308, 1e308)` declares the `float32` values of that interval,
   i.e. `[-3.4e38, 3.4e38]` on `float32`'s own grid — not the binary64
   interval. Members are produced by numpy/ml_dtypes casts and
   `nextafter` steps in the TARGET dtype, never by `uniform(lo, hi)`.
2. **The obligation universe comes from the SOURCE.** Every `S(...)`
   call site in the generated file is a row keyed by its line number,
   whether or not any sample evaluated it and whether or not stelling
   reported it. An obligation dropped inside an unreachable branch is
   then visible as `ABSENT`, not as an absence.
3. **The ledger is PER OBLIGATION.** The per-query roll-up is reported
   alongside, never instead — a corpus in this project once scored per
   query and turned a measured 24:168 trade into a fake 216:216.

Usage (from a worktree, with `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PYTHONPATH=<wt>/src`):

    python scratchpad/probe2/corpus2.py gen                # writes cases2.py
    python scratchpad/probe2/corpus2.py run OUT.json       # stelling + oracle
    python scratchpad/probe2/corpus2.py ledger A.json B.json

`gen` imports nothing but the stdlib and numpy/ml_dtypes; `run` needs jax
and stelling; `ledger` needs neither.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases2.py")


# ---------------------------------------------------------------------------
# the dtype grids — numpy/ml_dtypes only, never stelling
# ---------------------------------------------------------------------------


def _casters():
    import numpy as np
    import ml_dtypes

    return {
        "float16": np.float16,
        "bfloat16": ml_dtypes.bfloat16,
        "float32": np.float32,
        "float64": np.float64,
        "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
        "int2": ml_dtypes.int2,
        "int8": np.int8,
        "int32": np.int32,
        "bool": np.bool_,
    }


def _finfo(dt):
    import numpy as np
    import ml_dtypes

    c = _casters()[dt]
    try:
        return np.finfo(c)
    except (TypeError, ValueError):
        return ml_dtypes.finfo(c)


def _is_int(dt):
    return dt == "bool" or dt.startswith(("int", "uint"))


def member_bounds(dt, lo, hi):
    """(smallest, largest) VALUE of `dt` inside `[lo, hi]`, or (None, None)
    when there is none. Computed by casting and stepping in `dt`'s own
    grid — an algorithm with nothing in common with the one under test."""
    import numpy as np
    import ml_dtypes

    c = _casters()[dt]
    if _is_int(dt):
        info = (0, 1) if dt == "bool" else (
            int(np.iinfo(c).min), int(np.iinfo(c).max)
        ) if dt not in ("int2", "uint2", "int4", "uint4") else (
            int(ml_dtypes.iinfo(c).min), int(ml_dtypes.iinfo(c).max)
        )
        d_lo, d_hi = info
        a = d_lo if lo == -math.inf else max(math.ceil(lo), d_lo)
        b = d_hi if hi == math.inf else min(math.floor(hi), d_hi)
        return (float(a), float(b)) if a <= b else (None, None)
    big = float(_finfo(dt).max)
    with np.errstate(over="ignore"):
        a = c(max(min(lo, big), -big) if lo != -math.inf else -big)
        b = c(max(min(hi, big), -big) if hi != math.inf else big)
    while float(a) < lo:
        nxt = np.nextafter(a, c(np.inf))
        if not np.isfinite(nxt):
            return None, None
        a = nxt
    while float(b) > hi:
        nxt = np.nextafter(b, c(-np.inf))
        if not np.isfinite(nxt):
            return None, None
        b = nxt
    if float(a) > float(b):
        return None, None
    return float(a), float(b)


def members(dt, lo, hi, cap=9):
    """A sample of VALUES of `dt` inside `[lo, hi]`, plus whether the
    sample is exhaustive. Always includes both endpoints of the member
    set, because the corners are where the guards under test live."""
    import numpy as np

    c = _casters()[dt]
    m_lo, m_hi = member_bounds(dt, lo, hi)
    if m_lo is None:
        return [], True
    if _is_int(dt):
        n = int(m_hi) - int(m_lo) + 1
        if n <= cap:
            return [c(v) for v in range(int(m_lo), int(m_hi) + 1)], True
        step = n // (cap - 1)
        vals = sorted({int(m_lo) + i * step for i in range(cap - 1)} | {int(m_hi)})
        return [c(v) for v in vals], False
    # floats: the two corners, then a spread of interior values placed by
    # convex combination and snapped DOWN onto the dtype's grid
    seen = [c(m_lo), c(m_hi)]
    for f in (0.5, 0.25, 0.75, 0.1, 0.9, 1e-3, 1 - 1e-3):
        v = m_lo * (1.0 - f) + m_hi * f
        with np.errstate(over="ignore"):
            t = c(min(max(v, m_lo), m_hi))
        if float(t) < m_lo:
            t = np.nextafter(t, c(np.inf))
        if float(t) > m_hi:
            t = np.nextafter(t, c(-np.inf))
        seen.append(t)
    # dedupe, preserving order
    out, got = [], set()
    for t in seen:
        k = float(t)
        if k not in got:
            got.add(k)
            out.append(t)
    return out, len(out) <= 2


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

_V0 = None  # float32(0.1), filled at gen time


def _decls():
    import numpy as np

    v0 = float(np.float32(0.1))
    v1 = float(np.nextafter(np.float32(0.1), np.float32(np.inf)))
    return {
        # THE declaration both prior corpora lack: a float32 box wider
        # than float32
        "f32_wide": ((), "float32", (-1e308, 1e308)),
        "f32_ovf": ((), "float32", (-1e40, 1e40)),
        # a box whose interior holds no float32 at all
        "f32_subulp": ((), "float32", (v0, (v0 + v1) / 2.0)),
        # endpoints that are not float32 values
        "f32_offgrid": ((), "float32", (0.1, 0.3)),
        "f32_arr": ((2,), "float32", (-1.0, 1.0)),
        "f16_wide": ((), "float16", (-1e308, 1e308)),
        "f16_box": ((), "float16", (0.1, 0.2)),
        "bf16_wide": ((), "bfloat16", (-1e308, 1e308)),
        # the neighbours that must NOT move
        "f64_wide": ((), "float64", (-1e308, 1e308)),
        "f64_str": ((), "float64", (-1.0, 1.0)),
        "i8_wide": ((), "int8", (-1e9, 1e9)),
        "i32_frac": ((), "int32", (0.2, 2.8)),
        # the latent ones: dtypes neither table names
        "i2_wide": ((), "int2", (-1e9, 1e9)),
        "f8_box": ((), "float8_e4m3fn", (-1.0, 1.0)),
    }


def _guard_src(key, dtype, lo, hi, scalar):
    """The guard expression, as source. Narrow floats are compared after
    an explicit widening cast, because a `float16` LITERAL has no zero-dep
    decoder (measured: "no zero-dep decoder for array dtype '<f2'") and
    the comparison would go ⊤ before any of this is exercised."""
    m_lo, m_hi = member_bounds(dtype, lo, hi)
    w = "w" if dtype == "float64" else "w.astype(M.float64)"
    if not scalar:
        w = f"({w})[0]"
        # for the array declaration the member bounds are per element
    big = None if _is_int(dtype) else float(_finfo(dtype).max)
    if key == "below_min":  # FALSE at every member
        return None if m_lo is None else f"{w} < {m_lo!r}"
    if key == "at_min":  # TRUE at exactly the smallest member
        return None if m_lo is None else f"{w} <= {m_lo!r}"
    if key == "above_max":  # FALSE at every member
        return None if m_hi is None else f"{w} > {m_hi!r}"
    if key == "at_max":
        return None if m_hi is None else f"{w} >= {m_hi!r}"
    if key == "outside_dtype":  # FALSE at every member of a narrow dtype
        # `float64`'s own doubled maximum is `inf`, which is not a
        # threshold; that declaration simply has no such guard
        if big is None or not math.isfinite(big * 2.0):
            return None
        return f"{w} > {big * 2.0!r}"
    if key == "mid":
        if m_lo is None:
            return None
        return f"{w} > {(m_lo + m_hi) / 2.0!r}"
    if key == "pos":
        return f"{w} > 0.0"
    if key == "cancel":  # FALSE everywhere; the vacuity classic
        return f"{w} - {w} > 0.0"
    if key == "mixed":  # independent of w: pins the probe ladder/grid
        return "x[1] - x[0] > 1.0"
    raise KeyError(key)


GUARDS = (
    "below_min", "at_min", "above_max", "at_max",
    "outside_dtype", "mid", "pos", "cancel", "mixed",
)
SHAPES = ("cond_f", "cond_t", "nest", "switch", "assume")


def gen():
    decls = _decls()
    # REUSE-IgnoreStart -- these two strings are the header the GENERATED
    # file carries; they are not this file's own licensing statement, and
    # `reuse lint` reads them as an unparsable expression if left bare.
    out = [
        "# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy",
        "# SPDX-License-Identifier: Apache-2.0",
        '"""GENERATED by corpus2.py — do not edit."""',
        "",
    ]
    # REUSE-IgnoreEnd
    lines_of = {}
    names = []

    def w_(s):
        out.append(s)

    for dname, (dims, dtype, bounds) in decls.items():
        scalar = dims == ()
        # the guard-free shape: two top-level obligations, one true one false
        name = f"{dname}__none__top"
        w_(f"def case_{name}(A, S, C, M, AS):")
        w_(f"    x = A('x', (3,), 'float64', (-1.0, 1.0))")
        w_(f"    w = A('w', {dims!r}, {dtype!r}, {bounds!r})")
        w_("    del w")
        w_("    a = S(x > 5.0)")
        w_("    b = S(x > -9.0)")
        w_("    return a, b")
        w_("")
        names.append(name)

        for gkey in GUARDS:
            g = _guard_src(gkey, dtype, bounds[0], bounds[1], scalar)
            if g is None:
                continue
            for shape in SHAPES:
                name = f"{dname}__{gkey}__{shape}"
                w_(f"def case_{name}(A, S, C, M, AS):")
                w_(f"    x = A('x', (3,), 'float64', (-1.0, 1.0))")
                w_(f"    w = A('w', {dims!r}, {dtype!r}, {bounds!r})")
                if shape == "cond_f":
                    w_(f"    return C.cond({g},")
                    w_("                  lambda v: S(v > 5.0),")
                    w_("                  lambda v: S(v > -9.0), x)")
                elif shape == "cond_t":
                    w_(f"    return C.cond({g},")
                    w_("                  lambda v: S(v > -9.0),")
                    w_("                  lambda v: S(v > 5.0), x)")
                elif shape == "nest":
                    w_(f"    return C.cond({g},")
                    w_("                  lambda v: C.cond(v[0] < 2.0,")
                    w_("                                   lambda u: S(u > 5.0),")
                    w_("                                   lambda u: S(u > -9.0), v),")
                    w_("                  lambda v: S(v > -9.0), x)")
                elif shape == "switch":
                    w_(f"    k = ({g}).astype(M.int32)")
                    w_("    return C.switch(k, [lambda v: S(v > -9.0),")
                    w_("                        lambda v: S(v > 5.0)], x)")
                elif shape == "assume":
                    w_("    AS(x[0] > -0.5)")
                    w_(f"    return C.cond({g},")
                    w_("                  lambda v: S(v > 5.0),")
                    w_("                  lambda v: S(v > -9.0), x)")
                w_("")
                names.append(name)

    # line number of every S( call site, per case
    text = "\n".join(out) + "\n"
    cur = None
    for i, line in enumerate(text.splitlines(), start=1):
        if line.startswith("def case_"):
            cur = line[len("def case_"):].split("(")[0]
            lines_of[cur] = []
        elif cur is not None and re.search(r"(?<![A-Za-z_])S\(", line):
            # NOT `"S(" in line`: that also matches the `AS(` assume call,
            # which is not an obligation and would sit in the universe as a
            # permanently-ABSENT row (measured — 137 phantom rows)
            lines_of[cur].append(i)
    body = text + "\nCASE_NAMES = " + repr(names) + "\nLINES = " + repr(lines_of) + "\n"
    with open(CASES, "w") as fh:
        fh.write(body)
    n_obl = sum(len(v) for v in lines_of.values())
    print(f"wrote {CASES}: {len(names)} cases, {n_obl} obligation rows")


# ---------------------------------------------------------------------------
# the oracle: numpy only, never stelling
# ---------------------------------------------------------------------------


class _NumpyControl:
    """`jax.lax`'s cond/switch, executed for real on concrete values."""

    @staticmethod
    def cond(pred, f, g, *ops):
        return f(*ops) if bool(pred) else g(*ops)

    @staticmethod
    def switch(k, branches, *ops):
        i = int(k)
        i = 0 if i < 0 else (len(branches) - 1 if i >= len(branches) else i)
        return branches[i](*ops)


def _x_samples(rng):
    import numpy as np

    base = [
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 1.0, 0.0]),   # makes x[1]-x[0] > 1 true
        np.array([1.0, -1.0, 0.0]),
        np.array([-0.6, 0.6, 0.2]),
        np.array([-0.4, 0.9, -0.3]),
    ]
    for _ in range(5):
        base.append(rng.uniform(-1.0, 1.0, size=3))
    return base


def oracle(mod, rng):
    """For every obligation row: how many sampled MEMBERS evaluated it,
    and at how many it was false."""
    import numpy as np

    seen = collections.defaultdict(lambda: [0, 0, 0])  # line -> [eval, false, skip]
    exhaustive = {}
    for name in mod.CASE_NAMES:
        fn = getattr(mod, f"case_{name}")
        decl = {}

        def A(nm, shape, dtype, bounds, _d=decl):
            _d[nm] = (tuple(shape), dtype, bounds)
            return None

        try:
            fn(A, lambda p: p, _NumpyControl, np, lambda p: None)
        except Exception:  # noqa: BLE001 — the shape probe only needs `decl`
            pass
        shape_w, dt_w, b_w = decl["w"]
        w_vals, ex = members(dt_w, b_w[0], b_w[1])
        exhaustive[name] = ex
        xs = _x_samples(rng)
        rows = mod.LINES[name]
        for ln in rows:
            seen[(name, ln)]
        if not w_vals:
            continue
        for wv in w_vals:
            wa = np.full(shape_w, wv, dtype=wv.dtype) if shape_w else wv
            for xv in xs:
                hits = {}
                blocked = [False]

                def A2(nm, shape, dtype, bounds, _w=wa, _x=xv):
                    return _x if nm == "x" else _w

                def S(pred, _h=hits):
                    # the CALLER's line in cases2.py is the obligation's
                    # identity; `sys._getframe` rather than `inspect.stack`
                    # because the latter rebuilds the whole frame record and
                    # this runs ~200k times
                    ln = sys._getframe(1).f_lineno
                    ok = bool(np.all(np.asarray(pred)))
                    _h[ln] = _h.get(ln, True) and ok
                    return np.asarray(pred)

                def AS(pred, _b=blocked):
                    if not bool(np.all(np.asarray(pred))):
                        _b[0] = True

                try:
                    fn(A2, S, _NumpyControl, np, AS)
                except Exception:  # noqa: BLE001 — an unevaluable sample counts nothing
                    continue
                if blocked[0]:
                    for ln in rows:
                        seen[(name, ln)][2] += 1
                    continue
                for ln, ok in hits.items():
                    seen[(name, ln)][0] += 1
                    if not ok:
                        seen[(name, ln)][1] += 1
    return seen, exhaustive


def run(path):
    import random

    import numpy as np

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import stelling
    from stelling.harness import any_array, assert_, assume, trace
    from stelling.propagate import propagate

    sys.path.insert(0, HERE)
    import cases2 as mod  # noqa: E402

    print(f"stelling: {stelling.__file__}")
    print(f"jax:      {jax.__version__}  x64={jax.config.jax_enable_x64}")

    rng = np.random.default_rng(20260808)
    seen, exhaustive = oracle(mod, rng)

    rows = {}
    for name in mod.CASE_NAMES:
        fn = getattr(mod, f"case_{name}")

        def h(_f=fn):
            return _f(
                lambda nm, shape, dtype, bounds: any_array(shape, dtype, bounds),
                assert_,
                jax.lax,
                jnp,
                assume,
            )

        status_by_line = {}
        try:
            p = propagate(trace(h))
            for o in p.obligations:
                # NOT `source_info[-1]`: the frames run inner-to-outer, so
                # the last one is the DRIVER, and a top-level assert would
                # then be keyed on the runner instead of on its call site
                # (measured — 28 phantom ABSENT rows). The first cases2.py
                # frame is the innermost, which is the `S(` call site.
                ln = None
                for fr in o.source_info or ():
                    if "cases2.py:" in fr:
                        ln = int(fr.split("cases2.py:")[1].split(" ")[0])
                        break
                if ln is not None:
                    prev = status_by_line.get(ln)
                    # two dynamic occurrences of one call site: the worse
                    # (most claim-y) status wins, so nothing is hidden
                    order = {"discharged": 0, "unknown": 1, "violated-over-set": 2}
                    if prev is None or order[o.status] > order[prev]:
                        status_by_line[ln] = o.status
            err = None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        for ln in mod.LINES[name]:
            ev, viol, skip = seen[(name, ln)]
            rows[f"{name}:{ln}"] = {
                "case": name,
                "line": ln,
                "status": status_by_line.get(ln, "ABSENT"),
                "err": err,
                "oracle_evaluated": ev,
                "oracle_violated": viol,
                "oracle_blocked": skip,
                "oracle_exhaustive": exhaustive.get(name, False),
            }
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=0, sort_keys=True)
    print(f"wrote {path}: {len(rows)} obligation rows")
    classify(rows, "this run")


def classify(rows, label):
    c = collections.Counter()
    unsound_ref, unsound_ver = [], []
    for k, r in rows.items():
        c[r["status"]] += 1
        if r["status"] == "violated-over-set":
            if r["oracle_evaluated"] == 0:
                c["UNSOUND-REFUTED (never evaluated at any member)"] += 1
                unsound_ref.append(k)
            elif r["oracle_violated"] == 0:
                c["UNSOUND-REFUTED (evaluated, never false)"] += 1
                unsound_ref.append(k)
            else:
                c["sound REFUTED"] += 1
        if r["status"] == "discharged" and r["oracle_violated"] > 0:
            c["UNSOUND-DISCHARGED"] += 1
            unsound_ver.append(k)
    print(f"--- {label} ---")
    for k, v in sorted(c.items()):
        print(f"  {v:5d}  {k}")
    if unsound_ref:
        print("  unsound refutation rows (first 12):")
        for k in unsound_ref[:12]:
            print(f"     {k}")
    if unsound_ver:
        print("  UNSOUND DISCHARGED rows:")
        for k in unsound_ver[:12]:
            print(f"     {k}")
    return c


def ledger(a_path, b_path):
    with open(a_path) as fh:
        a = json.load(fh)
    with open(b_path) as fh:
        b = json.load(fh)
    assert set(a) == set(b), "obligation universes differ"
    classify(a, f"BEFORE {a_path}")
    classify(b, f"AFTER  {b_path}")
    moves = collections.Counter()
    detail = collections.defaultdict(list)
    for k in sorted(a):
        s0, s1 = a[k]["status"], b[k]["status"]
        if s0 == s1:
            continue
        sound = "sound" if b[k]["oracle_violated"] > 0 else (
            "vacuous" if b[k]["oracle_evaluated"] == 0 else "true-at-all-members"
        )
        moves[f"{s0} -> {s1}  [oracle: {sound}]"] += 1
        detail[f"{s0} -> {s1}"].append(k)
    print("--- PER-OBLIGATION MOVES ---")
    if not moves:
        print("  (none)")
    for k, v in sorted(moves.items()):
        print(f"  {v:5d}  {k}")
    for k, v in sorted(detail.items()):
        print(f"  {k}: {len(v)} rows, e.g. {v[:6]}")
    # the per-QUERY roll-up, reported alongside and never instead
    def verdicts(rows):
        by = collections.defaultdict(list)
        for r in rows.values():
            by[r["case"]].append(r["status"])
        out = {}
        for case, st in by.items():
            if "violated-over-set" in st:
                out[case] = "REFUTED"
            elif all(s == "discharged" for s in st):
                out[case] = "VERIFIED"
            else:
                out[case] = "UNKNOWN"
        return out

    va, vb = verdicts(a), verdicts(b)
    qmoves = collections.Counter(
        f"{va[c]} -> {vb[c]}" for c in va if va[c] != vb[c]
    )
    print("--- PER-QUERY roll-up (alongside, never instead) ---")
    for k, v in sorted(qmoves.items()) or [("(none)", 0)]:
        print(f"  {v:5d}  {k}")
    into_verified = sum(v for k, v in qmoves.items() if k.endswith("-> VERIFIED"))
    print(f"  queries moving INTO VERIFIED: {into_verified}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "ledger":
        ledger(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command {cmd!r}")
