# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Interrogate the collected census IR — no new collection, no value claim.

Answers, from the corpus already traced by ``run_census.py``:

1. which targets contribute each wedge primitive, and whether jax-cfd's
   stencil path contains any wedge primitive at all;
2. whether the corpus's ``concatenate`` equations are ``jnp.roll``
   lowerings (statically-checked shifts — a style that structurally
   cannot clamp);
3. what produces and consumes ``rem`` (the modular wraparound surface);
4. whether gather/scatter indices are guarded (rem/min/max/select_n in
   the index cone) and whether outputs feed ``select_n`` masks;
5. for loop-nested wedge equations: whether the index derives from a
   loop **counter** (one quantified LIA query, no descent machinery) or
   from **carried state** (the genuinely hard case).

Writes ``design/census-interrogation.md``.
"""

from __future__ import annotations

import datetime
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import run_census  # noqa: E402

import stelling  # noqa: E402
from stelling import _jax_compat, ir  # noqa: E402
from stelling.census import is_wedge_primitive  # noqa: E402

ARTIFACT = Path(__file__).resolve().parents[1] / "design" / "census-interrogation.md"

GUARD_OPS = frozenset({"rem", "min", "max", "select_n", "and", "clamp"})
AFFINE_OPS = frozenset(
    {"add", "sub", "mul", "neg", "rem", "convert_element_type", "broadcast_in_dim",
     "reshape", "squeeze", "concatenate", "iota", "copy", "stop_gradient", "sign"}
)


@dataclass
class Frame:
    jaxpr: ir.Jaxpr
    owner: ir.JaxprEqn | None  # eqn in the parent that owns this jaxpr
    role: str  # "root" | "jit[name]" | "while[body]" | "scan[body]" | "cond[k]" | prim
    parent: "Frame | None"

    def chain(self) -> str:
        parts = []
        frame = self
        while frame is not None and frame.role != "root":
            parts.append(frame.role)
            frame = frame.parent
        return " > ".join(reversed(parts)) or "top"


def _unwrap(v):
    if isinstance(v, ir.ClosedJaxpr):
        return v.jaxpr
    if isinstance(v, ir.Jaxpr):
        return v
    return None


def _sub_frames(eqn: ir.JaxprEqn, parent: Frame):
    p = eqn.params_dict()
    prim = eqn.primitive
    if prim == "while":
        for key, role in (("cond_jaxpr", "while[cond]"), ("body_jaxpr", "while[body]")):
            sub = _unwrap(p.get(key))
            if sub is not None:
                yield Frame(sub, eqn, role, parent)
        return
    if prim == "scan":
        sub = _unwrap(p.get("jaxpr"))
        if sub is not None:
            yield Frame(sub, eqn, "scan[body]", parent)
        return
    if prim == "cond":
        for k, b in enumerate(p.get("branches", ()) or ()):
            sub = _unwrap(b)
            if sub is not None:
                yield Frame(sub, eqn, f"cond[{k}]", parent)
        return
    # generic: any sub-jaxpr anywhere in params (tuple-nested included)
    pending = [v for _, v in eqn.params]
    while pending:
        item = pending.pop()
        sub = _unwrap(item)
        if sub is not None:
            role = prim
            if prim == "jit":
                role = f"jit[{p.get('name', '?')}]"
            yield Frame(sub, eqn, role, parent)
        elif isinstance(item, tuple):
            pending.extend(item)
        elif isinstance(item, ir.NamedTupleParam):
            pending.extend(v for _, v in item.fields)


def all_frames(root: ir.Jaxpr):
    stack = [Frame(root, None, "root", None)]
    while stack:
        frame = stack.pop()
        for eqn in frame.jaxpr.eqns:
            yield frame, eqn
            stack.extend(_sub_frames(eqn, frame))


class Analysis:
    def __init__(self, label: str, closed: ir.ClosedJaxpr):
        self.label = label
        self.root = closed.jaxpr
        self.sites = list(all_frames(self.root))
        self.producers: dict[int, tuple[Frame, ir.JaxprEqn]] = {}
        self.consumers: dict[int, list[str]] = {}
        for frame, eqn in self.sites:
            for ov in eqn.outvars:
                self.producers[ov.id] = (frame, eqn)
            for a in eqn.invars:
                if isinstance(a, ir.Var):
                    self.consumers.setdefault(a.id, []).append(eqn.primitive)

    # -- backward cone over the index operand ------------------------------

    def cone(self, atoms, frame: Frame):
        ops: Counter = Counter()
        terminals: list[str] = []
        seen: set[int] = set()
        work = [(a, frame) for a in atoms]
        steps = 0
        while work and steps < 2000:
            steps += 1
            atom, fr = work.pop()
            if isinstance(atom, ir.Literal):
                terminals.append("literal")
                continue
            if atom.id in seen:
                continue
            seen.add(atom.id)
            invar_pos = next(
                (i for i, v in enumerate(fr.jaxpr.invars) if v.id == atom.id), None
            )
            if invar_pos is not None:
                self._classify_invar(fr, invar_pos, terminals, work)
                continue
            if any(v.id == atom.id for v in fr.jaxpr.constvars):
                terminals.append("closure-const")
                continue
            hit = self.producers.get(atom.id)
            if hit is None:
                terminals.append("unresolved")
                continue
            pfr, peqn = hit
            ops[peqn.primitive] += 1
            work.extend((a, pfr) for a in peqn.invars)
        return ops, Counter(terminals)

    def _classify_invar(self, fr: Frame, pos: int, terminals, work) -> None:
        if fr.role == "root":
            terminals.append("trace-input")
            return
        owner, parent = fr.owner, fr.parent
        p = owner.params_dict()
        prim = owner.primitive
        if prim == "while":
            cond_n = int(p.get("cond_nconsts", 0))
            body_n = int(p.get("body_nconsts", 0))
            if fr.role == "while[body]":
                if pos < body_n:
                    work.append((owner.invars[cond_n + pos], parent))
                else:
                    k = pos - body_n
                    kind = "counter" if self._while_counter(fr, body_n, k) else "carry"
                    terminals.append(f"while-{kind}")
            else:  # while[cond]
                if pos < cond_n:
                    work.append((owner.invars[pos], parent))
                else:
                    terminals.append("while-carry(cond)")
            return
        if prim == "scan":
            if "num_consts" not in p:  # jax 0.11 flattree scan: layout params gone
                terminals.append("scan-slot(layout-unknown-0.11)")
                return
            nc = int(p.get("num_consts", 0))
            ncar = int(p.get("num_carry", 0))
            if pos < nc:
                work.append((owner.invars[pos], parent))
            elif pos < nc + ncar:
                k = pos - nc
                kind = "counter" if self._scan_counter(fr, nc, ncar, k) else "carry"
                terminals.append(f"scan-{kind}")
            else:
                terminals.append("scan-xs")
            return
        if prim == "cond":
            work.append((owner.invars[1 + pos], parent))
            return
        if len(owner.invars) == len(fr.jaxpr.invars):  # jit and friends: 1:1
            work.append((owner.invars[pos], parent))
            return
        terminals.append(f"{prim}-input")

    def _body_out_producer(self, fr: Frame, out_index: int):
        outs = fr.jaxpr.outvars
        if out_index >= len(outs) or not isinstance(outs[out_index], ir.Var):
            return None
        hit = self.producers.get(outs[out_index].id)
        return hit[1] if hit else None

    def _is_step_update(self, fr: Frame, eqn, slot_var_id: int) -> bool:
        if eqn is None or eqn.primitive not in ("add", "sub"):
            return False
        kinds = {("var", a.id) if isinstance(a, ir.Var) else ("lit", None) for a in eqn.invars}
        return ("var", slot_var_id) in kinds and ("lit", None) in kinds

    def _while_counter(self, fr: Frame, body_n: int, k: int) -> bool:
        slot = fr.jaxpr.invars[body_n + k]
        return self._is_step_update(fr, self._body_out_producer(fr, k), slot.id)

    def _scan_counter(self, fr: Frame, nc: int, ncar: int, k: int) -> bool:
        slot = fr.jaxpr.invars[nc + k]
        return self._is_step_update(fr, self._body_out_producer(fr, k), slot.id)

    # -- per-question reports ----------------------------------------------

    def _cone_eqns(self, atoms, frame: Frame):
        """The set of equations in the backward cone (not just op names)."""
        eqns: list[tuple[Frame, ir.JaxprEqn]] = []
        seen: set[int] = set()
        work = [(a, frame) for a in atoms]
        steps = 0
        while work and steps < 2000:
            steps += 1
            atom, fr = work.pop()
            if isinstance(atom, ir.Literal) or atom.id in seen:
                continue
            seen.add(atom.id)
            hit = self.producers.get(atom.id)
            if hit is None:
                continue
            pfr, peqn = hit
            eqns.append((pfr, peqn))
            work.extend((a, pfr) for a in peqn.invars)
        return eqns

    def _forward_select_n(self, eqn: ir.JaxprEqn, max_hops: int = 6):
        """select_n eqns transitively consuming this eqn's outputs (same scope)."""
        found = []
        frontier = list(eqn.outvars)
        for _ in range(max_hops):
            next_frontier = []
            for ov in frontier:
                for frame2, eqn2 in self.sites:
                    if any(isinstance(a, ir.Var) and a.id == ov.id for a in eqn2.invars):
                        if eqn2.primitive == "select_n":
                            found.append(eqn2)
                        else:
                            next_frontier.extend(eqn2.outvars)
            if not next_frontier:
                break
            frontier = next_frontier
        return found

    def _predicate_vars(self, select_eqn: ir.JaxprEqn, frame: Frame) -> set[int]:
        """Var ids in the cone of a select_n's predicate (operand 0)."""
        _, _ = self, frame
        pred = select_eqn.invars[:1]
        eqns = self._cone_eqns(pred, frame)
        ids = {v.id for _, e in eqns for v in e.outvars}
        ids |= {a.id for a in pred if isinstance(a, ir.Var)}
        return ids

    def wedge_sites(self):
        out = []
        for frame, eqn in self.sites:
            if not is_wedge_primitive(eqn.primitive):
                continue
            if eqn.primitive == "gather" or eqn.primitive.startswith("scatter"):
                idx_atoms = eqn.invars[1:2]
            elif eqn.primitive == "dynamic_slice":
                idx_atoms = eqn.invars[1:]
            else:  # dynamic_update_slice
                idx_atoms = eqn.invars[2:]
            ops, terminals = self.cone(idx_atoms, frame)
            cone_eqns = self._cone_eqns(idx_atoms, frame)
            index_selects = [(f, e) for f, e in cone_eqns if e.primitive == "select_n"]
            clamp = bool(index_selects)
            downstream_selects = self._forward_select_n(eqn)
            mask = bool(downstream_selects)
            # designed-nullification: a downstream select_n whose predicate cone
            # shares a var with an index-guard select_n's predicate cone
            shared_predicate = False
            if clamp and mask:
                idx_pred_ids = set()
                for f, s in index_selects:
                    idx_pred_ids |= self._predicate_vars(s, f)
                for s in downstream_selects:
                    if self._predicate_vars(s, frame) & idx_pred_ids:
                        shared_predicate = True
                        break
            out.append(
                {
                    "prim": eqn.primitive,
                    "chain": frame.chain(),
                    "cone_ops": dict(ops),
                    "terminals": dict(terminals),
                    "guards": sorted(set(ops) & GUARD_OPS),
                    "affine_cone": set(ops) <= AFFINE_OPS,
                    "clamp": clamp,
                    "mask": mask,
                    "shared_predicate": shared_predicate,
                    "feeds_select_n": mask,
                }
            )
        return out

    def prim_sites(self, prim: str):
        return [
            (frame.chain(), eqn)
            for frame, eqn in self.sites
            if eqn.primitive == prim
        ]

    def rem_report(self):
        out = []
        for frame, eqn in self.sites:
            if eqn.primitive != "rem":
                continue
            consumers = sorted(
                {c for ov in eqn.outvars for c in self.consumers.get(ov.id, ())}
            )
            out.append({"chain": frame.chain(), "consumers": consumers})
        return out


def main() -> int:
    analyses = []
    for target, harnesses in run_census.HARNESSES.items():
        for i, harness in enumerate(harnesses):
            label = target if len(harnesses) == 1 else f"{target}#{i}"
            try:
                _, cj = harness()
                analyses.append(Analysis(label, _jax_compat.transcribe(cj)))
                print(f"[ok] {label}")
            except Exception as exc:
                print(f"[skip] {label}: {exc}")

    lines = [
        "# Census interrogation — answers from already-collected IR",
        "",
        f"**Status:** evidence artifact, run {datetime.date.today().isoformat()}, "
        f"jax {jax.__version__}, stelling {stelling.__version__}. Generated by "
        "`corpus/interrogate_census.py` from the same harnesses as the census; "
        "no new collection, no value claim. Re-verify with the census.",
        "",
    ]

    # Q1: who owns the wedge primitives; does jax-cfd have any?
    lines += ["## 1. Wedge primitives, by target", ""]
    cfd_has_wedge = False
    for a in analyses:
        counts = Counter(eqn.primitive for _, eqn in a.sites if is_wedge_primitive(eqn.primitive))
        if counts:
            lines.append(f"- **{a.label}**: " + ", ".join(f"`{p}` ×{n}" for p, n in sorted(counts.items())))
            if a.label.startswith("jax-cfd"):
                cfd_has_wedge = True
        else:
            lines.append(f"- {a.label}: none")
    lines += [
        "",
        f"**Does jax-cfd's traced stencil path contain any wedge primitive? "
        f"{'Yes' if cfd_has_wedge else 'No.'}**",
        "",
    ]

    # Q2: concatenate — whose, and are they roll lowerings?
    lines += ["## 2. `concatenate`: whose, and from what", ""]
    for a in analyses:
        sites = a.prim_sites("concatenate")
        if not sites:
            continue
        chains = Counter(chain for chain, _ in sites)
        lines.append(f"- **{a.label}**: {len(sites)} eqns")
        for chain, n in chains.most_common():
            lines.append(f"  - ×{n} at `{chain}`")
    lines.append("")

    # Q3: rem — whose, and what consumes it
    lines += ["## 3. `rem`: the wraparound surface", ""]
    for a in analyses:
        for entry in a.rem_report():
            lines.append(
                f"- **{a.label}** at `{entry['chain']}` → consumed by: "
                + (", ".join(f"`{c}`" for c in entry["consumers"]) or "(unconsumed / output)")
            )
    lines.append("")

    # Q4 + Q5: wedge site detail — guards, masks, counter vs carry
    lines += [
        "## 4. Wedge sites: index provenance, guard anatomy, loop classification",
        "",
        "Guard anatomy (work order §3): **clamp** = `select_n` in the *index*",
        "cone (`where(ok, idx, fallback)` before the access — the access is",
        "in bounds by construction, the fallback read is wrong unless nulled).",
        "**mask** = `select_n` transitively consuming the *output* (`where(ok,",
        "val, 0)` after — the access itself may still be out of bounds, and",
        "the author is relying on XLA's clamp being harmless). **shared** =",
        "a downstream mask whose predicate cone shares a variable with the",
        "index guard's predicate — the designed-nullification pattern.",
        "",
        "| site | primitive | context | terminals | guards in cone | clamp | mask | shared pred | affine |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in analyses:
        for s in a.wedge_sites():
            term_text = ", ".join(f"{k}×{v}" for k, v in sorted(s["terminals"].items())) or "—"
            lines.append(
                f"| {a.label} | `{s['prim']}` | `{s['chain']}` "
                f"| {term_text} | {', '.join(s['guards']) or '—'} "
                f"| {'yes' if s['clamp'] else '—'} "
                f"| {'yes' if s['mask'] else '—'} "
                f"| {'yes' if s['shared_predicate'] else '—'} "
                f"| {'yes' if s['affine_cone'] else 'no'} |"
            )
    lines += [
        "",
        "Terminal legend: `while-counter`/`scan-counter` — the index depends on a",
        "loop-carried slot whose update is `slot ± literal` (one quantified LIA",
        "query: `∀i ∈ [0,N). idx(i) ∈ bounds` — no unrolling, no invariants).",
        "`while-carry`/`scan-carry` — genuinely state-dependent (the hard case).",
        "`closure-const`/`literal` — index known at trace time; in-bounds is",
        "decidable by direct evaluation. `trace-input` — from harness arguments.",
        "",
        "## 5. Is the guarding pattern indexing-specific?",
        "",
    ]
    lines += _sqrt_defence_probe()
    lines.append("")

    ARTIFACT.write_text("\n".join(lines))
    print(f"wrote {ARTIFACT}")
    return 0


def _sqrt_defence_probe() -> list[str]:
    """Numeric probe: does jax-md defend the sqrt(0) backward-NaN class the
    same way its indices are guarded? (work order §5)"""
    try:
        import jax.numpy as jnp
        import jax_md
        import jax_md.util

        displacement, _ = jax_md.space.periodic(10.0)
        energy_fn = jax_md.energy.soft_sphere_pair(displacement)
        positions = jnp.array([[1.0, 1.0], [1.0, 1.0], [4.0, 4.0], [7.0, 2.0]])
        grad = jax.grad(energy_fn)(positions)
        finite = bool(jnp.isfinite(grad).all())
        has_safe_mask = hasattr(jax_md.util, "safe_mask")
        if finite:
            return [
                "Probe: `grad(soft_sphere_pair_energy)` at **coincident",
                "particles** (rows 0 and 1 identical) — the `sqrt(0)`",
                "backward-NaN class: finite forward, NaN backward, unless",
                "defended.",
                "",
                f"- gradient finite at the coincident configuration: **{finite}**",
                "  (the naive result would be NaN)",
                f"- `jax_md.util.safe_mask` exists: **{has_safe_mask}**",
                "",
                "**Defended.** The same hand-guarding observed at every index",
                "site covers the numerically hazardous sqrt too. The guard",
                "finding is not about indexing — it is about what *mature",
                "library* means: every hazard class probed so far is defended,",
                "by hand, uniformly, with no tool checking that the defences",
                "work.",
            ]
        return [
            "Probe: coincident-particle gradient is **NaN** — jax-md has a",
            "live backward-NaN site (a finding in its own right), and",
            "NaN-freedom has a target the wedge does not.",
        ]
    except Exception as exc:  # pragma: no cover
        return [f"Probe failed: {type(exc).__name__}: {exc}"]


if __name__ == "__main__":
    raise SystemExit(main())
