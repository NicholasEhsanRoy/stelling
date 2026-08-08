# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A corpus that is IDENTICAL in two processes, and the ledger it produces.

The cross-series property compares one tree's verdicts under jax 0.10.2 against
the same tree's verdicts under jax 0.11.0. Two interpreters cannot share a
Hypothesis search, so the corpus has to be reproducible outside Hypothesis
entirely: this module builds it from ``random.Random(seed)`` over the same
pools ``_grammar`` draws from, which depends on the Python version's PRNG
contract and on nothing else — not on the Hypothesis version, not on jax, not
on the order the two processes happen to run in.

The comparison is keyed on the RENDERED HARNESS TEXT rather than on the index,
so a corpus that did diverge shows up as "present on one side only" — a
reportable disagreement — instead of silently comparing two different programs
and calling them equal.

**Two harnesses in ``_FIXED`` are not random.** They are there because the
cross-series property's positive control needs them: at ``8ef8f75`` a
scatter-add was VERIFIED at 6/6 equations known on jax 0.11.0 and UNKNOWN at
4/6 with a ⊤ scatter-add on 0.10.2, because the combiner was recognised by
jaxpr CONTAINER CLASS and 0.11 merged ``ClosedJaxpr`` into ``Jaxpr``. A purely
random corpus over the arithmetic grammar would never build one, and the
control could not be demonstrated. That is the honest shape of this property's
reach: it finds what the corpus contains.

Run as a script to emit the ledger as JSON on stdout — which is exactly what
``test_cross_series.py`` does to the other interpreter.
"""

from __future__ import annotations

import json
import random
import sys

import _grammar as G

CORPUS_SEED = 20260808
CORPUS_SIZE = 220


def _rand_expr(rng, decls, depth):
    names = [d.name for d in decls]
    floaty = any(d.dtype in G.FLOAT_DTYPES for d in decls)
    if depth <= 0 or rng.random() < 0.35:
        if rng.random() < 0.35:
            pool = G.FLOAT_POOL if floaty else G.INT_POOL
            return ("const", rng.choice(pool))
        return ("var", rng.choice(names))
    kind = rng.randrange(5)
    if kind == 0:
        ops = G.UNARY_ANY + (G.UNARY_FLOAT if floaty else ())
        return ("un", rng.choice(ops), _rand_expr(rng, decls, depth - 1))
    if kind == 1:
        ops = G.BINARY_ANY + (G.BINARY_FLOAT if floaty else ())
        return ("bin", rng.choice(ops), _rand_expr(rng, decls, depth - 1),
                _rand_expr(rng, decls, depth - 1))
    if kind == 2:
        return ("cancel", _rand_expr(rng, decls, depth - 1))
    if kind == 3:
        return ("sum", _rand_expr(rng, decls, depth - 1))
    targets = sorted({d.dtype for d in decls} - {"bool"}) or ["int32"]
    return ("cast", rng.choice(targets), _rand_expr(rng, decls, depth - 1))


def _rand_pred(rng, decls, depth):
    if depth <= 0 or rng.random() < 0.6:
        return ("cmp", rng.choice(G.CMP_OPS), _rand_expr(rng, decls, 2),
                _rand_expr(rng, decls, 1))
    k = rng.randrange(3)
    if k == 0:
        return ("and", _rand_pred(rng, decls, depth - 1),
                _rand_pred(rng, decls, depth - 1))
    if k == 1:
        return ("or", _rand_pred(rng, decls, depth - 1),
                _rand_pred(rng, decls, depth - 1))
    return ("not", _rand_pred(rng, decls, depth - 1))


def _rand_spec(rng):
    n = rng.randint(1, 2)
    dtype = rng.choice(G.ALL_DTYPES)
    decls = []
    for i in range(n):
        dt = dtype if i == 0 or rng.random() < 0.7 else rng.choice(G.ALL_DTYPES)
        shape = rng.choice(G.SHAPES)
        if dt == "bool":
            lo, hi = False, True
        elif dt in G.FLOAT_DTYPES:
            lo, hi = sorted((rng.choice(G.FLOAT_POOL), rng.choice(G.FLOAT_POOL)),
                            key=lambda v: (v != v, v))
        else:
            a, b = rng.choice(G.INT_POOL), rng.choice(G.INT_POOL)
            lo, hi = min(a, b), max(a, b)
        decls.append(G.Decl(f"x{i}", shape, dt, lo, hi))
    decls = tuple(decls)
    stmts = []
    for _ in range(rng.randint(1, 3)):
        kind = "assume" if rng.random() < 0.3 else "assert"
        stmts.append(G.Stmt(kind, _rand_pred(rng, decls, 1)))
    if not any(s.kind == "assert" for s in stmts):
        stmts.append(G.Stmt("assert", _rand_pred(rng, decls, 1)))
    return G.Spec(decls, tuple(stmts))


def _fixed():
    """Hand-built harnesses the random grammar cannot reach, and why each is here."""
    f64 = lambda name, shape, lo, hi: G.Decl(name, shape, "float64", lo, hi)  # noqa: E731
    out = []
    # scatter-add: the cross-series control's subject.
    for shape, idx in (((3,), 0), ((3,), 2), ((4,), 1)):
        d = f64("x0", shape, -1.0, 1.0)
        out.append(
            G.Spec(
                (d,),
                (
                    G.Stmt(
                        "assert",
                        ("cmp", ">=",
                         ("at_add", ("var", "x0"), idx, 5.0),
                         ("const", -1.0)),
                    ),
                ),
            )
        )
    # scatter-add under an assume, and with a second obligation.
    d = f64("x0", (3,), 0.0, 1.0)
    out.append(
        G.Spec(
            (d,),
            (
                G.Stmt("assume", ("cmp", ">=", ("var", "x0"), ("const", 0.25))),
                G.Stmt("assert", ("cmp", ">=",
                                  ("at_add", ("var", "x0"), 1, 2.0),
                                  ("const", 0.0))),
                G.Stmt("assert", ("cmp", "<=", ("var", "x0"), ("const", 1.0))),
            ),
        )
    )
    # a size-0 declaration beside a rank-0 one: the vacuous-conjunct shape.
    out.append(
        G.Spec(
            (f64("x0", (), -1.0, 1.0), f64("x1", (0,), -1.0, 1.0)),
            (
                G.Stmt("assume", ("and",
                                  ("cmp", ">=", ("var", "x0"), ("const", 0.5)),
                                  ("cmp", ">=", ("var", "x1"), ("const", 2.0)))),
                G.Stmt("assert", ("cmp", ">", ("var", "x0"), ("const", 0.0))),
            ),
        )
    )
    return out


def corpus(size: int = CORPUS_SIZE, seed: int = CORPUS_SEED):
    rng = random.Random(seed)
    return _fixed() + [_rand_spec(rng) for _ in range(size)]


def ledger(size: int = CORPUS_SIZE, seed: int = CORPUS_SEED, *, refine=None):
    """``{rendered harness: outcome}`` for this interpreter's jax."""
    import jax

    import _runner

    jax.config.update("jax_enable_x64", True)
    out = {}
    for spec in corpus(size, seed):
        key = spec.render()
        try:
            from stelling.preconditions import check

            v = check(G.build(spec), vacuity_mode="inputs-only", refine=refine)
            out[key] = {
                "status": v.status,
                "obligations": [o.status for o in v.obligations],
                "refusal": None,
            }
        except Exception as exc:  # noqa: BLE001 — the TYPE is the datum
            name = type(exc).__name__
            if name not in _runner.REFUSALS:
                raise
            out[key] = {"status": None, "obligations": None, "refusal": name}
    return out


def main(argv):
    import jax

    import stelling

    size = int(argv[1]) if len(argv) > 1 else CORPUS_SIZE
    print(
        json.dumps(
            {
                "jax": jax.__version__,
                "stelling_file": stelling.__file__,
                "python": sys.version.split()[0],
                "ledger": ledger(size),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
    raise SystemExit(main(sys.argv))
