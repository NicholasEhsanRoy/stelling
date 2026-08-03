# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every count quoted about the declaration layer's emptiness decision.

The rule this exists to satisfy: **a figure is quoted only with the script
that produced it.** Each number the branch `fix/widening-admitted-empty`
claims is printed here, against whichever tree is on ``PYTHONPATH`` — so the
same script run against the parent and against the branch produces the
before/after pair, and nothing has to be taken on report.

    PYTHONPATH=<tree>/src JAX_PLATFORMS=cpu \\
        python corpus/declaration_emptiness_sweep.py [--fuzz N] [--seed S]

THE ORACLE IS NOT THE TREE. Emptiness is decided here from the formats'
PUBLISHED PARAMETERS (IEEE 754 binary32/binary64 and the x87
double-extended format longdouble uses, written down below) in exact
rational arithmetic with no ``float()`` in the decision path, from
enumerated bit patterns for the formats of at most two bytes, and from
written-down integer ranges keyed by dtype NAME. A dtype the tables do not
describe raises rather than answering, so the oracle cannot rubber-stamp by
falling through — the failure mode of the instrument this replaced, which
shared `_exact_at_or_above`'s ``nextafter``-from-a-binary64-cast walk and
therefore certified `float128 (2**54+1, 2**54+3)` empty when longdouble
holds all three of its inhabitants.

Sections, each printing its own denominator:

1. THE GRID — every declaration, through four routes, with the oracle's
   verdict beside the tool's. Reports admitted-empty (the defect),
   refused-inhabited (a false refusal), and the newly-refused set.
2. ROUTES — every public entry point that declares an input, and every
   bound-pair position within it, compared byte for byte.
3. FUZZ — randomized declarations against the same oracle.
4. CORPUS — how many literal declaration bounds in ``corpus/`` are not
   exactly representable as the binary64 the IR stores (i.e. how many could
   possibly move).
5. ACCEPTANCE LOSS — the layer's PRE-EXISTING false refusals, which this
   branch does not touch: the dtype-level narrowing policy refuses every
   narrowing bound on int64/uint64/complex/longdouble whether or not the
   shaved sliver holds a value, so `uint64 (Fraction(-1,3), Fraction(1,3))`
   — which contains 0 — is refused. Counted so its size is on the record.
"""

from __future__ import annotations

import argparse
import functools
import math
import random
import sys
import warnings
from decimal import Decimal
from fractions import Fraction

import jax

jax.config.update("jax_enable_x64", True)

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402

import stelling  # noqa: E402
from stelling import contracts, preconditions  # noqa: E402
from stelling.harness import any_array, any_pytree  # noqa: E402

INF = float("inf")

# ---------------------------------------------------------------------------
# the oracle: published format parameters, exact rational arithmetic
# ---------------------------------------------------------------------------
_IEEE_FORMAT = {                 # (precision, emin, emax)
    "float32": (24, -126, 127),
    "float64": (53, -1022, 1023),
    "float128": (64, -16382, 16383),
}
_INT_DOMAIN = {
    "bool": (0, 1),
    **{f"int{n}": (-(2 ** (n - 1)), 2 ** (n - 1) - 1)
       for n in (2, 4, 8, 16, 32, 64)},
    **{f"uint{n}": (0, 2**n - 1) for n in (2, 4, 8, 16, 32, 64)},
}


def _dtype(name):
    return np.dtype(getattr(ml_dtypes, name, name))


def _binade(x):
    """The ``e`` with ``2**e <= x < 2**(e+1)``, for an exact rational x > 0."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return e


def _least_ieee_at_or_above(p, emin, emax, x):
    """Least value of the binary format (p, emin, emax) that is >= x."""
    tiny = Fraction(1, 2) ** (p - 1 - emin)
    big = (Fraction(2) - Fraction(1, 2) ** (p - 1)) * Fraction(2) ** emax
    if x > big:
        return None
    if x <= -big:
        return -big
    if x == 0:
        return Fraction(0)
    if x < 0:
        y = -x
        if y < tiny:
            return Fraction(0)
        u = max(tiny, Fraction(1, 2) ** (p - 1 - _binade(y)))
        return -((y // u) * u)
    if x <= tiny:
        return tiny
    u = max(tiny, Fraction(1, 2) ** (p - 1 - _binade(x)))
    v = -((-x) // u) * u
    return None if v > big else v


@functools.lru_cache(maxsize=None)
def _enumerated_values(name):
    """Every finite value of a <=2-byte float dtype, exactly, from its bit
    patterns. ``float()`` builds them; each is cast back and required to
    survive, so the exactness is measured rather than assumed."""
    d = _dtype(name)
    raw = np.arange(2 ** (8 * d.itemsize),
                    dtype=np.uint8 if d.itemsize == 1 else np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore"):
            vals = raw.view(d)
    out = set()
    for v in vals:
        f = float(v)
        if not math.isfinite(f):
            continue
        # the VALUE must survive the round trip, which is what makes the
        # Fraction the dtype's value rather than a rounding of it. Not the
        # bit pattern: a sub-byte format has padding bits, so several
        # patterns denote one value and bitwise equality fails for 93 of
        # float6_e2m3fn's 256 patterns while the value round-trips for all.
        assert float(np.array(f, d)) == f, (name, v, f)
        out.add(Fraction(f))
    return sorted(out)


def oracle(name, lo, hi):
    """(does [lo, hi] hold a value of `name`, an exact witness or None)."""
    dt = str(_dtype(name))
    if lo == INF or hi == -INF or (lo != -INF and hi != INF and lo > hi):
        return False, None
    if dt == "bool" or dt.startswith(("int", "uint")):
        a, b = _INT_DOMAIN[dt]
        first = a if (lo == -INF or lo <= a) else math.ceil(lo)
        if first > b or (hi != INF and first > hi):
            return False, None
        return True, first
    if dt.startswith("complex"):
        return None, None                # policy: the layer makes no claim
    if dt in _IEEE_FORMAT:
        p, emin, emax = _IEEE_FORMAT[dt]
        big = (Fraction(2) - Fraction(1, 2) ** (p - 1)) * Fraction(2) ** emax
        v = _least_ieee_at_or_above(
            p, emin, emax, -big if lo == -INF else Fraction(lo))
        if v is None or (hi != INF and v > hi):
            return False, None
        return True, v
    if _dtype(dt).itemsize <= 2:
        vals = _enumerated_values(dt)
        first = next((v for v in vals if lo == -INF or v >= lo), None)
        if first is None or (hi != INF and first > hi):
            return False, None
        return True, first
    raise AssertionError(f"no written-down description of {dt!r}")


def exact_value(v):
    """The exact number a bound spelling denotes — re-derived here, so the
    sweep does not lean on `stelling._bound_spelling` to say what it fed in."""
    if isinstance(v, bool):
        return Fraction(int(v))
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return v if math.isinf(v) else Fraction(v)
    if isinstance(v, Fraction):
        return v
    if isinstance(v, Decimal):
        if v.is_infinite():
            return INF if v > 0 else -INF
        return Fraction(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return exact_value(v[()])
    if isinstance(v, np.integer):
        return Fraction(int(v))
    if isinstance(v, np.floating):
        return (INF if v > 0 else -INF) if np.isinf(v) else Fraction(
            *v.as_integer_ratio())
    raise AssertionError(repr(type(v)))


# ---------------------------------------------------------------------------
# the dtypes and the grid
# ---------------------------------------------------------------------------
FLOATS = ["float64", "float32", "float16", "bfloat16", "float8_e4m3fn",
          "float8_e4m3fnuz", "float8_e4m3b11fnuz", "float8_e5m2",
          "float8_e5m2fnuz", "float8_e3m4", "float8_e4m3", "float8_e8m0fnu",
          "float6_e2m3fn", "float6_e3m2fn", "float4_e2m1fn"]
INTS = ["bool", "int2", "int4", "int8", "int16", "int32", "int64",
        "uint2", "uint4", "uint8", "uint16", "uint32", "uint64"]
COMPLEX = ["complex64", "complex128"]


def _known(n):
    try:
        _dtype(n)
        return True
    except TypeError:
        return False


ALL_DTYPES = [d for d in FLOATS + INTS + COMPLEX if _known(d)]

SPELLINGS = ["int", "np.int64", "np.longdouble", "Decimal", "Fraction",
             "0d-int64", "0d-longdouble", "float"]


def spell(kind, v):
    """`v` (an int) in the named spelling, or None if it cannot hold it."""
    if kind == "int":
        return int(v)
    if kind == "np.int64":
        return np.int64(v) if -(2**63) <= v < 2**63 else None
    if kind in ("np.longdouble", "0d-longdouble"):
        ld = np.longdouble(v)
        if int(ld) != int(v):
            return None
        return np.array(ld) if kind == "0d-longdouble" else ld
    if kind == "Decimal":
        return Decimal(int(v))
    if kind == "Fraction":
        return Fraction(v)
    if kind == "0d-int64":
        return np.array(v, np.int64) if -(2**63) <= v < 2**63 else None
    if kind == "float":
        f = float(v)
        return f if f == v else None
    raise AssertionError(kind)


def _extremes(name):
    d = _dtype(name)
    if d.kind == "c":
        return None, None
    if str(d) in _INT_DOMAIN:
        return _INT_DOMAIN[str(d)]
    if str(d) in _IEEE_FORMAT:
        p, _, emax = _IEEE_FORMAT[str(d)]
        big = (Fraction(2) - Fraction(1, 2) ** (p - 1)) * Fraction(2) ** emax
        return -big, big
    vals = _enumerated_values(str(d))
    return vals[0], vals[-1]


def grid():
    """(tag, dtype, lo, hi) — the declarations measured."""
    out = []
    for dt in ALL_DTYPES:
        lo_x, hi_x = _extremes(dt)
        if lo_x is None:
            pairs = [(0, 1), (2**54 + 1, 2**54 + 3), (-(2**64), -(2**63) - 1),
                     (0, 2**53 + 3), (-1, 1)]
        else:
            iL, iH = math.floor(lo_x), math.ceil(hi_x)
            pairs = [
                (iL, iH), (iL, iL), (iH, iH), (iH, iH + 1), (iH + 1, iH + 2),
                (iH + 1, iH + 3), (iL - 2, iL - 1), (iL - 1, iL),
                (iL - 3, iL - 1), (0, 1), (-1, 1), (0, 0),
                (2**54 + 1, 2**54 + 3), (2**53 + 1, 2**53 + 3),
                (-(2**64), -(2**63) - 1), (-(2**54) - 3, -(2**54) - 1),
                (0, 2**53 + 3), (0, 2**53 + 1), (2**53 + 1, 2**60),
                (0, 10**23), (10**23, 10**24),
                (-INF, iH), (iL, INF), (-INF, INF),
                (-INF, 2**53 + 1), (2**53 + 3, INF),
            ]
        out += [(f"sweep/{dt}", dt, lo, hi) for lo, hi in pairs]

    for dt, lo, hi in [
        ("float64", 2**54 + 1, 2**54 + 3), ("float64", 2**54 + 1, 2**54 + 4),
        ("float64", 2**54, 2**54 + 3), ("float64", 2**54 + 1, 2**54 + 1),
        ("float64", 2**53 + 1, 2**53 + 3), ("float64", 2**53 + 3, 2**53 + 4),
        ("int64", -(2**64), -(2**63) - 1), ("int64", -(2**64), -(2**63)),
        ("int64", -(2**63), 2**63 - 1), ("uint64", 0, 2**64 - 1),
        ("uint64", 2**64, 2**65), ("bfloat16", 2**53 + 1, 2**53 + 3),
        ("float32", 2**54 + 1, 2**54 + 3), ("float16", 2**54 + 1, 2**54 + 3),
        ("int32", 0, 10**23), ("int8", 2**54 + 1, 2**54 + 3),
        # raw-order-INVERTED, images collapsing onto equality
        ("int64", 2**53 + 1, 2**53), ("float64", 2**53 + 1, 2**53),
        ("float32", 2**53 + 1, 2**53), ("int32", 2**53 + 1, 2**53),
        ("uint64", 2**64 - 1, 2**64 - 2), ("bfloat16", 2**54 + 1, 2**54),
    ]:
        for k in SPELLINGS:
            a, b = spell(k, lo), spell(k, hi)
            if a is not None and b is not None:
                out.append((f"spell/{k}", dt, a, b))

    for dt, lo, hi in [
        ("int64", Fraction(1, 10), Fraction(1, 5)),
        ("int64", Decimal("0.1"), Decimal("0.2")),
        ("float64", Fraction(1, 3), Fraction(2, 3)),
        ("float64", Fraction(1, 10**400), Fraction(2, 10**400)),
        ("float64", Fraction(1, 2**1076), Fraction(3, 2**1076)),
        ("float64", Fraction(1, 2**1076), Fraction(7, 2**1076)),
        ("float8_e4m3", Fraction(1, 2**1076), Fraction(3, 2**1076)),
        ("uint64", -1.0, Fraction(-1, 2**1200)),
        ("uint32", -1.0, Fraction(-1, 2**1200)),
        ("bool", -1.0, Fraction(-1, 2**1200)),
        ("float16", -1e-300, Fraction(-1, 2**1200)),
        ("uint32", -1.0, Fraction(1, 2**1200)),
        ("uint32", Fraction(-3, 2**1200), Fraction(-1, 2**1200)),
        ("float64", Fraction(2**55 + 1, 2), Fraction(2**55 + 7, 2)),
        ("float64", Fraction(2**53 * 2 + 1, 2), Fraction(2**53 * 2 + 3, 2)),
        ("float64", -0.0, 0.0), ("float32", 0.0, 1e39),
        ("float32", 10**39, 10**40),
        ("float32", int(np.finfo(np.float32).max) + 1, INF),
        ("float32", -INF, -int(np.finfo(np.float32).max) - 1),
        ("float64", int(np.finfo(np.float32).max) + 1, INF),
        ("complex64", 2**54 + 1, 2**54 + 3),
        ("complex128", 2**54 + 1, 2**54 + 3),
    ]:
        out.append(("edge", dt, lo, hi))

    for dt in [d for d in FLOATS if _known(d)]:
        vals = _enumerated_values(dt) if _dtype(dt).itemsize <= 2 else None
        if vals is None:
            p, emin, _ = _IEEE_FORMAT[dt]
            t = Fraction(1, 2) ** (p - 1 - emin)
        else:
            t = min(v for v in vals if v > 0)
        L, H = _extremes(dt)
        out += [(f"sub/{dt}", dt, a, b) for a, b in [
            (t, t), (t / 4, t / 2), (t / 4, t), (Fraction(0), t / 2),
            (Fraction(H), Fraction(H) * 2),
            (Fraction(H) + Fraction(1, 7), INF),
            (Fraction(L) * 2, Fraction(L)),
            (-INF, Fraction(L) - Fraction(1, 7)),
        ]]
    return out


# ---------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------
def _decide(fn):
    """The decision a declaration reaches, TRACED — `any_array` binds a jax
    primitive and refuses to run outside `make_jaxpr`."""
    try:
        jax.make_jaxpr(fn)()
        return "admit"
    except ValueError as e:
        return "refuse:" + str(e)
    except Exception as e:                       # a crash is neither
        return f"CRASH:{type(e).__name__}:{e}"


def route_positions(dt, lo, hi, shape=(1,)):
    """Every public entry point that declares an input, keyed by
    ``function/parameter`` — nine bound-pair POSITIONS across seven public
    functions. A route is established by measurement, not by the call graph:
    two of the four route defects this layer has had were upstream
    pre-conversions the call graph did not show."""
    b, ident = (1.0, 2.0), (lambda x: x)
    triple = lambda x: (x, x, x)             # noqa: E731 — the (a,b,c) form
    return {
        "harness.any_array": lambda: (any_array(shape, dt, (lo, hi)),),
        "harness.any_pytree":
            lambda: any_pytree(np.zeros(shape, _dtype(dt)), (lo, hi)),
        "contracts.conditioning_2x2/a_range": lambda: contracts.
            conditioning_2x2(dt, (lo, hi), b, (0.0, 1.0), 10.0).harness(),
        "contracts.conditioning_2x2/c_range": lambda: contracts.
            conditioning_2x2(dt, b, (lo, hi), (0.0, 1.0), 10.0).harness(),
        "contracts.conditioning_2x2/b_range": lambda: contracts.
            conditioning_2x2(dt, b, b, (lo, hi), 10.0).harness(),
        "contracts.conditioning_2x2_field/theta_range": lambda: contracts.
            conditioning_2x2_field(shape, dt, (lo, hi), 10.0,
                                   triple).harness(),
        "contracts.coefficient_contrast/chi_range": lambda: contracts.
            coefficient_contrast(shape, dt, (lo, hi), 10.0, ident).harness(),
        "preconditions.field_positive/envelope":
            lambda: preconditions.field_positive(shape, dt, (lo, hi)),
        "preconditions.scalar_nonzero/envelope":
            lambda: preconditions.scalar_nonzero(dt, (lo, hi)),
    }


GRID_ROUTES = ("harness.any_array", "harness.any_pytree",
               "contracts.conditioning_2x2/a_range",
               "preconditions.field_positive/envelope")

# jax refuses to trace the template harnesses for these dtypes at all, and
# its message lists an unordered pair whose order differs between processes
_JAX_PROMOTION = "no available implicit dtype promotion path"


# ---------------------------------------------------------------------------
def section_grid(cases):
    print(f"\n== 1. THE GRID — {len(cases)} declarations x {len(GRID_ROUTES)} "
          f"routes = {len(cases) * len(GRID_ROUTES)} route decisions")
    tally = {k: 0 for k in ("admit-inhabited", "ADMIT-EMPTY (defect)",
                            "refuse-empty", "REFUSE-INHABITED (false)",
                            "complex (policy: no claim)",
                            "other refusal cause", "CRASH")}
    decisions, empties = {}, []
    for tag, dt, lo, hi in cases:
        row = {}
        for r in GRID_ROUTES:
            row[r] = _decide(route_positions(dt, lo, hi)[r])
        decisions[(tag, dt, repr(lo), repr(hi))] = row
        v = row["harness.any_array"]
        holds, _ = oracle(dt, exact_value(lo), exact_value(hi))
        if v.startswith("CRASH"):
            tally["CRASH"] += 1
        elif holds is None:
            tally["complex (policy: no claim)"] += 1
        elif v == "admit":
            tally["admit-inhabited" if holds else "ADMIT-EMPTY (defect)"] += 1
            if not holds:
                empties.append((dt, repr(lo), repr(hi)))
        elif "EMPTY under dtype" in v:
            tally["refuse-empty" if not holds
                  else "REFUSE-INHABITED (false)"] += 1
            if holds:
                empties.append(("FALSE-REFUSAL", dt, repr(lo), repr(hi)))
        else:
            tally["other refusal cause"] += 1
    for k, n in tally.items():
        print(f"   {n:6d}  {k}")
    for e in empties[:20]:
        print(f"           {e}")
    return decisions


def section_routes():
    print("\n== 2. ROUTES")
    cases = [("float64", 2**54 + 1, 2**54 + 3), ("float64", 2**54 + 1, 2**54 + 4),
             ("int64", -(2**64), -(2**63) - 1), ("int64", -(2**64), -(2**63))]
    total = agree = 0
    for dt, lo, hi in cases:
        res = {k: _decide(v) for k, v in route_positions(dt, lo, hi).items()}
        ref = res["harness.any_array"]
        for k, v in res.items():
            total += 1
            agree += v == ref
            if v != ref:
                print(f"   DISAGREE {dt} ({lo}, {hi}) {k}: {v[:70]}")
    n_fn = len({k.split("/")[0] for k in route_positions("float64", 0, 1)})
    print(f"   {n_fn} public functions, "
          f"{len(route_positions('float64', 0, 1))} bound-pair positions; "
          f"byte-identical to the hand route: {agree}/{total}")


def section_fuzz(n, seed):
    print(f"\n== 3. FUZZ — {n} randomized declarations, seed {seed}")
    rng = random.Random(seed)

    def rand():
        k = rng.randrange(10)
        if k == 0:
            return rng.choice([INF, -INF, 0.0, -0.0])
        if k <= 3:
            e = rng.randrange(0, 70)
            v = rng.randrange(2**e, 2 ** (e + 1)) if e else rng.randrange(0, 2)
            return v * rng.choice([1, -1]) + rng.randrange(-3, 4)
        if k <= 5:
            e = rng.randrange(0, 70)
            return Fraction(rng.randrange(2**e, 2 ** (e + 1))
                            * rng.choice([1, -1]), rng.choice([2, 4, 8]))
        if k <= 7:
            return Fraction(rng.randrange(-10**12, 10**12),
                            rng.randrange(1, 10**7))
        if k == 8:
            return Fraction(rng.randrange(1, 100) * rng.choice([1, -1]),
                            2 ** rng.randrange(1000, 1090))
        return float(np.float64(rng.uniform(-1e30, 1e30)))

    counts = {"admit": 0, "empty": 0, "other": 0, "complex": 0}
    bad = []
    for _ in range(n):
        dt = rng.choice(ALL_DTYPES)
        a, b = rand(), rand()
        if exact_value(a) > exact_value(b):
            a, b = b, a
        v = _decide(lambda: (any_array((1,), dt, (a, b)),))
        holds, _ = oracle(dt, exact_value(a), exact_value(b))
        if holds is None:
            counts["complex"] += 1
            continue
        if v.startswith("CRASH"):
            bad.append(("CRASH", dt, repr(a), repr(b), v[:90]))
        elif v == "admit":
            counts["admit"] += 1
            if not holds:
                bad.append(("ADMITTED-EMPTY", dt, repr(a)[:40], repr(b)[:40]))
        elif "EMPTY under dtype" in v:
            counts["empty"] += 1
            if holds:
                bad.append(("REFUSED-INHABITED", dt, repr(a)[:40], repr(b)[:40]))
        else:
            counts["other"] += 1
    print(f"   decisions: {counts}")
    print(f"   violations: {len(bad)}")
    for x in bad[:15]:
        print(f"      {x}")


def section_corpus():
    import ast
    import pathlib

    names = {"any_array", "any_pytree", "field_positive", "scalar_nonzero",
             "conditioning_2x2", "conditioning_2x2_field",
             "coefficient_contrast"}
    here = pathlib.Path(__file__).resolve()
    root = here.parent
    pairs, inexact, skipped = [], [], 0
    for f in sorted(root.rglob("*.py")):
        if f.resolve() == here:
            continue                 # this sweep's own params are not corpus
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = getattr(node.func, "attr", getattr(node.func, "id", None))
            if fn not in names:
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if not (isinstance(arg, ast.Tuple) and len(arg.elts) == 2):
                    continue
                try:
                    vals = [ast.literal_eval(e) for e in arg.elts]
                except Exception:
                    skipped += 1
                    continue
                if all(isinstance(v, (int, float)) for v in vals):
                    pairs.append((str(f), tuple(vals)))
                else:
                    skipped += 1
    for _, (lo, hi) in pairs:
        for v in (lo, hi):
            if isinstance(v, int) and float(v) != v:
                inexact.append(v)
    print(f"\n== 4. CORPUS — {len(pairs)} literal declaration bound PAIRS "
          f"({2 * len(pairs)} endpoint values); "
          f"{skipped} non-literal expressions skipped")
    print(f"   endpoints whose binary64 image differs from the declared "
          f"value: {len(inexact)}  (only these can move)")


def section_acceptance_loss(cases):
    """PRE-EXISTING and untouched by this branch: the dtype-level narrowing
    policy refuses every narrowing bound on int64/uint64/complex/longdouble
    whether or not the shaved sliver holds a value of the dtype."""
    n = loss = 0
    examples = []
    extra = [(dt, Fraction(-1, 3), Fraction(1, 3))
             for dt in ("int64", "uint64")]
    for dt, lo, hi in extra + [(c[1], c[2], c[3]) for c in cases]:
        v = _decide(lambda: (any_array((1,), dt, (lo, hi)),))
        if not v.startswith("refuse:") or "NARROWS" not in v:
            continue
        n += 1
        holds, w = oracle(dt, exact_value(lo), exact_value(hi))
        if holds:
            loss += 1
            if len(examples) < 3:
                examples.append((dt, repr(lo)[:40], repr(hi)[:40], w))
    print(f"\n== 5. ACCEPTANCE LOSS (pre-existing; identical on parent) — "
          f"{n} narrowing-policy refusals in this population")
    print(f"   of those, over an INHABITED declared set: {loss}")
    for dt, lo, hi, w in examples:
        print(f"   e.g. {dt} ({lo}, {hi}) contains {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    print(f"stelling: {stelling.__file__}")
    print(f"jax: {jax.__version__}   numpy: {np.__version__}")
    cases = grid()
    section_grid(cases)
    section_routes()
    section_fuzz(args.fuzz, args.seed)
    section_corpus()
    section_acceptance_loss(cases)


if __name__ == "__main__":
    sys.exit(main())
