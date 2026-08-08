# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""probe — the corpus behind `scratchpad/PREREG_PROBE.md`.

Independent of `scratchpad/reach/` and of the audit's corpus. Two design
decisions are deliberate, and both exist because of how `PREREG_REACH.md`'s
clause C5 failed:

1. **The oracle samples MEMBERS.** For an integer or boolean declaration it
   enumerates the integers of `[ceil(lo), floor(hi)]` intersected with the
   dtype's own range — never `uniform(lo, hi)`. An oracle that samples
   `0.2` for an `int32` cannot see the defect this branch fixes at all; the
   previous corpus sampled exactly that way, which is why a 736-case run
   missed it.
2. **The obligation universe comes from the SOURCE.** Every `S(...)` call
   site in the generated case file is a row, whether or not any sampled
   point evaluated it — `LINES` is emitted by the generator, not derived
   from what the oracle saw run. A falsifier that ranges only over executed
   obligations cannot see an obligation dropped in an unreachable branch.

The ledger is PER OBLIGATION. The per-query roll-up is reported alongside,
never instead: a corpus in this project once scored per query and turned a
measured 24:168 trade into 216:216.

Usage (from a worktree, with `JAX_PLATFORMS=cpu PYTHONPATH=<wt>/src`):

    python scratchpad/probe/corpus.py gen                 # writes cases.py
    python scratchpad/probe/corpus.py run OUT.json [N]    # stelling + oracle
    python scratchpad/probe/corpus.py analyze OUT.json
    python scratchpad/probe/corpus.py ledger BEFORE.json AFTER.json

`gen` imports nothing; `run` needs jax and stelling; `analyze` and `ledger`
need neither.
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases.py")

# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

# name -> (var, shape, dtype, (lo, hi), the declared SET in words)
DECLS = {
    # the defect's own shape: the integers of a fractional interval
    "i_frac": ("i", (), "int32", (0.2, 2.8), "{1, 2}"),
    "i_frac2": ("i", (), "int32", (-0.5, 2.5), "{0, 1, 2}"),
    "i_int": ("i", (), "int32", (0.0, 3.0), "{0, 1, 2, 3}"),
    "i_one": ("i", (), "int32", (0.9, 1.1), "{1}"),
    # bounded twice: the interval AND the dtype
    "i8_wide": ("i", (), "int8", (-1e9, 1e9), "[-128, 127]"),
    "u8_neg": ("i", (), "uint8", (-5.5, 3.5), "{0, 1, 2, 3}"),
    "b_bool": ("i", (), "bool", (0.0, 1.0), "{0, 1}"),
    # floats: the neighbours that must not move, and the wide box
    "f_pos": ("c", (), "float64", (1.0, 2.0), "[1, 2]"),
    "f_str": ("c", (), "float64", (-1.0, 1.0), "[-1, 1]"),
    "f_wide": ("c", (), "float64", (-1e308, 1e308), "the float range"),
}

# guard-key -> (expression, which declarations it may be applied to)
_INT = ("i_frac", "i_frac2", "i_int", "i_one", "u8_neg", "b_bool")
_I8 = ("i8_wide",)
_FLT = ("f_pos", "f_str", "f_wide")
GUARDS = {
    "lt1": ("i < 1", _INT),
    "gt2": ("i > 2", _INT),
    "eq1": ("i == 1", _INT),
    "between": ("(i - 1) * (i - 2) < 0", ("i_frac", "i_frac2", "i_int")),
    # numpy refuses boolean subtraction where jax allows it, so the
    # cancellation guard skips the bool declaration rather than lose the
    # oracle for it
    "cancel": ("i - i > 0", tuple(k for k in _INT if k != "b_bool")),
    "ge0": ("i >= 0", _INT),
    "castlo": ("i.astype(jnp.float64) < -200.0", _I8),
    "castmid": ("i.astype(jnp.float64) < -100.0", _I8),
    "castcancel": ("i.astype(jnp.float64) - i.astype(jnp.float64) > 0.0", _I8),
    "pos": ("c > 0.0", _FLT),
    "lowend": ("c < -1e307", ("f_wide",)),
    "highend": ("c > 1e307", ("f_wide",)),
    "fcancel": ("c - c > 0.0", _FLT),
    "sq": ("c * c > 0.25", ("f_str",)),
    "opaque": ("jnp.sin(c * 0.0) > 0.0", ("f_str",)),
    "mixed": ("x[1] - x[0] > 1.0", tuple(DECLS)),
}

# how the obligation sits relative to the guard
SHAPES = ("top", "cond_f", "cond_t", "nest", "scan", "assume", "switch")


def _emit_case(w, name, decl_key, guard_key, shape):
    var, dims, dtype, bounds, _set = DECLS[decl_key]
    guard = GUARDS[guard_key][0]
    w(f"def case_{name}(A, S, C, M):")
    w(f"    x = A('x', (3,), 'float64', (-1.0, 1.0))")
    w(f"    {var} = A('{var}', {dims!r}, '{dtype}', {bounds!r})")
    if shape == "top":
        w(f"    del {var}")
        w("    a = S(x > 5.0)")
        w("    b = S(x > -9.0)")
        w("    return a, b")
    elif shape == "cond_f":
        # the false obligation sits in the guard's TRUE leg
        w(f"    return C.cond({guard},")
        w("                  lambda v: S(v > 5.0),")
        w("                  lambda v: S(v > -9.0), x)")
    elif shape == "cond_t":
        # ... and in the FALSE leg: the other polarity
        w(f"    return C.cond({guard},")
        w("                  lambda v: S(v > -9.0),")
        w("                  lambda v: S(v > 5.0), x)")
    elif shape == "nest":
        w(f"    inner = lambda v: C.cond({guard},")
        w("                             lambda w: S(w > 5.0),")
        w("                             lambda w: S(w > -9.0), v)")
        w("    return C.cond(x[0] > 0.0, inner, lambda v: S(v > -8.0), x)")
    elif shape == "scan":
        w(f"    g = {guard}")
        w("    body = lambda cy, v: (cy, S(v > 5.0))")
        w("    out = C.scan(body, 0.0, x)")
        w("    top = S(x > -9.0)")
        w("    return out, top, g")
    elif shape == "assume":
        w(f"    M({var} == {var})")
        w(f"    return C.cond({guard},")
        w("                  lambda v: S(v > 5.0),")
        w("                  lambda v: S(v > -9.0), x)")
    elif shape == "switch":
        w(f"    k = ({guard}).astype(jnp.int32)")
        w("    return C.switch(k, [lambda v: S(v > -9.0),")
        w("                        lambda v: S(v > 5.0)], x)")
    w("")


def gen():
    lines = []
    w = lines.append
    w('"""GENERATED by corpus.py gen -- do not edit. One S() per line."""')
    w("# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy")
    w("# SPDX-License-Identifier: Apache-2.0")
    w("import jax")
    w("import jax.numpy as jnp")
    w("")
    names = []
    topped = set()
    for decl_key in DECLS:
        for guard_key, (_expr, ok) in GUARDS.items():
            if decl_key not in ok:
                continue
            for shape in SHAPES:
                if shape == "switch" and not GUARDS[guard_key][0].startswith(
                    ("i ", "c ", "(i", "x[")
                ):
                    continue
                if shape == "top":
                    # the guard is unused there: one control per
                    # declaration, not one per guard
                    if decl_key in topped:
                        continue
                    topped.add(decl_key)
                name = f"{decl_key}__{guard_key}__{shape}"
                names.append((name, decl_key, guard_key, shape))
    # record each case's S() call sites while emitting: the obligation
    # universe comes from the SOURCE, never from what the oracle ran
    line_map = {}
    for name, decl_key, guard_key, shape in names:
        start = len(lines)
        _emit_case(w, name, decl_key, guard_key, shape)
        # every `S(` call site, in source order — the obligation universe
        line_map[name] = [
            start + off + 1 for off, ln in enumerate(lines[start:]) if " S(" in ln
        ]
    w("CASES = {")
    for name, decl_key, guard_key, shape in names:
        w(
            f"    {name!r}: (case_{name}, "
            f"{{'decl': {decl_key!r}, 'guard': {guard_key!r}, "
            f"'shape': {shape!r}}}),"
        )
    w("}")
    w("LINES = {")
    for name, *_ in names:
        w(f"    {name!r}: {line_map[name]!r},")
    w("}")
    with open(CASES, "w") as f:
        f.write("\n".join(lines) + "\n")
    n_obl = sum(len(v) for v in line_map.values())
    print(f"wrote {CASES}: {len(names)} cases, {n_obl} obligation call sites")


# --------------------------------------------------------------------------
# the oracle: plain numpy, real control flow, MEMBERS of the declared set
# --------------------------------------------------------------------------


class _Reject(Exception):
    """This sampled point violates an assumed precondition."""


def _members(dtype, lo, hi, rng, cap=9):
    """Candidate values for one declaration — MEMBERS of its declared set.

    For an integer or boolean dtype that is the integers of
    `[ceil(lo), floor(hi)]` intersected with the dtype's range: nothing
    else is a value the program can hold, so nothing else is a point a
    witness may use.
    """
    import numpy as np

    ranges = {
        "bool": (0, 1),
        **{f"int{n}": (-(2 ** (n - 1)), 2 ** (n - 1) - 1) for n in (8, 16, 32, 64)},
        **{f"uint{n}": (0, 2**n - 1) for n in (8, 16, 32, 64)},
    }
    if dtype in ranges:
        d_lo, d_hi = ranges[dtype]
        m_lo = max(math.ceil(lo), d_lo)
        m_hi = min(math.floor(hi), d_hi)
        if m_lo > m_hi:
            return [], True
        n = m_hi - m_lo + 1
        if n <= cap:
            vals = list(range(m_lo, m_hi + 1))
            exhaustive = True
        else:
            vals = sorted(
                {m_lo, m_hi, m_lo + 1, m_hi - 1, (m_lo + m_hi) // 2, 0}
                & set(range(m_lo, m_hi + 1))
                | {m_lo, m_hi}
            )
            vals += [int(rng.integers(m_lo, m_hi + 1)) for _ in range(cap)]
            exhaustive = False
        return [np.array(v, dtype=np.dtype(dtype)) for v in vals], exhaustive
    mid = 0.5 * lo + 0.5 * hi
    vals = [lo, hi, mid]
    if lo < 0.0 < hi:
        vals.append(0.0)
    vals += [float(lo * (1 - f) + hi * f) for f in (0.25, 0.75, 0.9, 0.1)]
    return [np.array(v, dtype=np.float64) for v in vals], False


def _array_members(shape, dtype, lo, hi, rng, n_random):
    import numpy as np

    size = int(np.prod(shape)) if shape else 1
    if not shape:
        vals, ex = _members(dtype, lo, hi, rng)
        return vals, ex
    base, _ = _members(dtype, lo, hi, rng)
    out = [np.full(shape, b, dtype=b.dtype) for b in base]
    if size <= 3:
        for combo in itertools.product(base, repeat=size):
            out.append(np.array(combo).reshape(shape))
    for _ in range(n_random):
        out.append(rng.uniform(lo, hi, size=shape).astype(np.float64))
    return out, False


class _Ctl:
    """Real control-flow semantics, in plain Python: only the taken branch
    runs, which is what makes `n_exec == 0` mean 'unreachable'."""

    def cond(self, pred, tf, ff, *ops):
        import numpy as np

        return tf(*ops) if bool(np.asarray(pred)) else ff(*ops)

    def switch(self, idx, branches, *ops):
        import numpy as np

        k = int(np.clip(int(np.asarray(idx)), 0, len(branches) - 1))
        return branches[k](*ops)

    def scan(self, f, init, xs):
        import numpy as np

        carry, ys = init, []
        for i in range(np.asarray(xs).shape[0]):
            carry, y = f(carry, np.asarray(xs)[i])
            ys.append(y)
        return carry, ys


class _JaxShim:
    jit = staticmethod(lambda f, *a, **k: f)


def _numpy_mode():
    import numpy as np

    import cases as C

    old = (C.jnp, C.jax)
    C.jnp, C.jax = np, _JaxShim
    return C, old


def run_oracle(fn, n_random=60, seed=17):
    import numpy as np

    C, old = _numpy_mode()
    try:
        rng = np.random.default_rng(seed)
        decls = []

        def A_collect(name, shape, dtype, bounds):
            decls.append((name, tuple(shape), dtype, bounds))
            vals, _ = _members(dtype, bounds[0], bounds[1], rng)
            if not vals:
                raise _Reject("declaration has no member")
            return np.full(shape, vals[0], dtype=vals[0].dtype)

        fn(A_collect, lambda p: p, _Ctl(), lambda p: p)

        cand, exhaustive = {}, True
        for name, shape, dtype, (lo, hi) in decls:
            vals, ex = _array_members(shape, dtype, lo, hi, rng, n_random)
            cand[name] = vals
            exhaustive = exhaustive and ex
        # mixed-radix enumeration with the SMALLEST declaration varying
        # fastest, so a 3-member integer set is swept at every point of the
        # float array's grid rather than pinned to one value. (A stride
        # scheme did pin one: `(j * 3) % 3 == 0` held `i` at its first
        # member for every point, and the corpus then reported a genuinely
        # reachable branch as unreachable.)
        names = sorted((d[0] for d in decls), key=lambda n: len(cand[n]))
        sizes = [len(cand[n]) for n in names]
        pts = []
        if math.prod(sizes) <= 600:
            for combo in itertools.product(*[cand[n] for n in names]):
                pts.append(dict(zip(names, combo)))
        else:
            for j in range(max(max(sizes), 300)):
                radix, pt = 1, {}
                for n in names:
                    pt[n] = cand[n][(j // radix) % len(cand[n])]
                    radix *= len(cand[n])
                pts.append(pt)
            for _ in range(150):  # independent draws, as a second design
                pts.append({n: cand[n][rng.integers(len(cand[n]))] for n in names})

        n_exec: dict[int, int] = {}
        n_false: dict[int, int] = {}
        n_kept = 0
        for p in pts:
            rec_exec: dict[int, bool] = {}
            rec_val: dict[int, bool] = {}

            def A(name, shape, dtype, bounds, _p=p):
                return _p[name]

            def S(pred, _e=rec_exec, _v=rec_val):
                import inspect

                ln = inspect.currentframe().f_back.f_lineno
                _e[ln] = True
                _v[ln] = _v.get(ln, True) and bool(np.all(np.asarray(pred)))
                return pred

            def M(pred):
                if not bool(np.all(np.asarray(pred))):
                    raise _Reject("assumed precondition false here")
                return pred

            try:
                fn(A, S, _Ctl(), M)
            except _Reject:
                continue
            n_kept += 1
            for ln, ok in rec_val.items():
                n_exec[ln] = n_exec.get(ln, 0) + 1
                if not ok:
                    n_false[ln] = n_false.get(ln, 0) + 1
        return {
            "n_points": n_kept,
            "n_exec": n_exec,
            "n_false": n_false,
            "exhaustive": exhaustive,
        }
    finally:
        C.jnp, C.jax = old


# --------------------------------------------------------------------------
# the stelling side
# --------------------------------------------------------------------------


def run_all(out_path, n_random=60, only=None):
    import traceback

    import jax

    jax.config.update("jax_enable_x64", True)

    sys.path.insert(0, HERE)
    import cases as CASES_MOD

    import stelling
    from stelling.harness import any_array, assert_, assume
    from stelling.preconditions import check

    print("STELLING", stelling.__file__, flush=True)

    cases_path = os.path.abspath(CASES_MOD.__file__).replace(".pyc", ".py")

    class JaxCtl:
        cond = staticmethod(lambda p, t, f, *o: jax.lax.cond(p, t, f, *o))
        switch = staticmethod(lambda i, b, *o: jax.lax.switch(i, b, *o))
        scan = staticmethod(lambda f, init, xs: jax.lax.scan(f, init, xs))

    def case_line(source_info):
        for fr in source_info or ():
            path = fr.split(" (")[0]
            f, _, ln = path.rpartition(":")
            if os.path.abspath(f) == cases_path:
                try:
                    return int(ln)
                except ValueError:
                    return None
        return None

    def run_stelling(fn, **kw):
        def harness():
            return fn(
                lambda name, shape, dtype, bounds: any_array(shape, dtype, bounds),
                assert_,
                JaxCtl,
                assume,
            )

        try:
            v = check(harness, **kw)
        except Exception as e:  # noqa: BLE001
            return {
                "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-400:],
            }
        return {
            "status": v.status,
            "obl": [
                {
                    "status": o.status,
                    "line": case_line(o.source_info),
                    "detail": o.detail[:120],
                }
                for o in v.obligations
            ],
        }

    legs = [
        ("base", dict(vacuity_mode="all")),
        ("inputs-only", dict(vacuity_mode="inputs-only")),
    ]
    results = {}
    names = sorted(CASES_MOD.CASES)
    if only:
        names = [n for n in names if only in n]
    for n_done, name in enumerate(names):
        fn, meta = CASES_MOD.CASES[name]
        entry = {"meta": meta, "lines": CASES_MOD.LINES[name], "legs": {}}
        try:
            entry["oracle"] = run_oracle(fn, n_random=n_random)
        except Exception as e:  # noqa: BLE001
            entry["oracle_error"] = f"{type(e).__name__}: {e}"
        for leg, kw in legs:
            entry["legs"][leg] = run_stelling(fn, **kw)
        results[name] = entry
        if (n_done + 1) % 50 == 0:
            print(f"  ... {n_done + 1}/{len(names)}", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path}: {len(results)} cases")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

UNSOUND = (
    "REFUTE_ON_UNREACHABLE",
    "REFUTE_CONTRADICTED",
    "DISCHARGE_CONTRADICTED",
    "SWALLOWED_FALSE",
)


def classify(path):
    d = json.load(open(path))
    per_obl, per_query = [], []
    for name, e in sorted(d.items()):
        if "oracle" not in e:
            per_query.append((name, "-", "ORACLE_ERROR", 0))
            continue
        n_exec = {int(k): v for k, v in e["oracle"]["n_exec"].items()}
        n_false = {int(k): v for k, v in e["oracle"]["n_false"].items()}
        # THE UNIVERSE IS THE SOURCE, not the executed set
        all_lines = set(e["lines"]) | set(n_exec) | set(n_false)
        for leg, r in sorted(e["legs"].items()):
            if r is None or "error" in r:
                per_query.append((name, leg, "STELLING_ERROR", 0))
                continue
            seen = {o["line"]: o for o in r["obl"] if o["line"] is not None}
            for ln in sorted(all_lines):
                ex, fa = n_exec.get(ln, 0), n_false.get(ln, 0)
                o = seen.get(ln)
                if o is None:
                    klass = (
                        "SWALLOWED_FALSE"
                        if fa
                        else ("SWALLOWED_TRUE" if ex else "SWALLOWED_UNREACHED")
                    )
                elif o["status"] == "violated-over-set":
                    klass = (
                        "REFUTE_ON_UNREACHABLE"
                        if ex == 0
                        else ("REFUTE_CONTRADICTED" if fa == 0 else "REFUTE_SOUND")
                    )
                elif o["status"] == "discharged":
                    klass = (
                        "DISCHARGE_CONTRADICTED"
                        if fa
                        else ("DISCHARGE_VACUOUS" if ex == 0 else "DISCHARGE_SOUND")
                    )
                else:
                    klass = (
                        "UNKNOWN_ON_UNREACHABLE"
                        if ex == 0
                        else ("UNKNOWN_ON_FALSE" if fa else "UNKNOWN_ON_TRUE")
                    )
                per_obl.append(
                    (name, leg, ln, ex, fa, (o or {}).get("status", "<absent>"), klass)
                )
            any_false = any(n_false.get(ln, 0) > 0 for ln in all_lines)
            if r["status"] == "VERIFIED":
                q = "FALSE_VERIFIED" if any_false else "VERIFIED_SOUND"
            elif r["status"] == "REFUTED":
                q = "REFUTED_SOUND" if any_false else "FALSE_REFUTED"
            else:
                q = r["status"]
            per_query.append((name, leg, q, 0))
    return per_obl, per_query


def analyze(path):
    per_obl, per_query = classify(path)
    print("== PER-OBLIGATION ==")
    c = collections.Counter(r[6] for r in per_obl)
    for k in sorted(c):
        print(f"  {k:26s} {c[k]}")
    print("== PER-QUERY ==")
    c2 = collections.Counter(r[2] for r in per_query)
    for k in sorted(c2):
        print(f"  {k:26s} {c2[k]}")
    print("== UNSOUND ROWS (sample) ==")
    seen = collections.Counter()
    for r in per_obl:
        if r[6] in UNSOUND:
            seen[r[6]] += 1
            if seen[r[6]] <= 5:
                print("  ", r)
    print(f"  total unsound obligation rows: {sum(seen.values())}")


def ledger(before, after):
    b_obl, b_q = classify(before)
    a_obl, a_q = classify(after)
    bo = {(r[0], r[1], r[2]): r for r in b_obl}
    ao = {(r[0], r[1], r[2]): r for r in a_obl}
    assert set(bo) == set(ao), (len(bo), len(ao), list(set(bo) ^ set(ao))[:4])
    moves = collections.Counter()
    rows = []
    for k in sorted(bo):
        if bo[k][5] != ao[k][5]:
            moves[(bo[k][6], ao[k][6])] += 1
            rows.append((k, bo[k][5], ao[k][5], bo[k][6], ao[k][6]))
    print("== PER-OBLIGATION MOVES (before-class -> after-class) ==")
    for (x, y), n in sorted(moves.items(), key=lambda kv: -kv[1]):
        print(f"  {x:26s} -> {y:26s} {n}")
    print(f"  total moved obligations: {len(rows)}")
    print("== TOWARD VERIFIED (must be empty) ==")
    bad = [r for r in rows if r[2] == "discharged"]
    for r in bad[:10]:
        print("  ", r)
    print(f"  obligations that BECAME discharged: {len(bad)}")
    bq = {(r[0], r[1]): r[2] for r in b_q}
    aq = {(r[0], r[1]): r[2] for r in a_q}
    qmoves = collections.Counter(
        (bq[k], aq[k]) for k in sorted(bq) if bq[k] != aq[k]
    )
    print("== PER-QUERY MOVES ==")
    for (x, y), n in sorted(qmoves.items(), key=lambda kv: -kv[1]):
        print(f"  {x:20s} -> {y:20s} {n}")
    into_v = [k for k in bq if bq[k] != aq[k] and aq[k].startswith("VERIFIED")]
    print(f"  queries that moved INTO VERIFIED: {len(into_v)} {into_v[:5]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    elif cmd == "run":
        run_all(
            sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 60,
            sys.argv[4] if len(sys.argv) > 4 else None,
        )
    elif cmd == "analyze":
        analyze(sys.argv[2])
    elif cmd == "ledger":
        ledger(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
