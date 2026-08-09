"""Is molecular identity a coherence-dominated label? The pair screen at scale.

The ncas = 10 bridge experiment found that for ONE pair (C2H2 vs HCN, matched
kT) the class-difference aggregate carried more off-diagonal than diagonal
Frobenius weight (ratio 1.15) — suggesting "which molecule" as a label family
for the coherence program.  One pair is an anecdote.  This script computes the
same diagnostic for hundreds of pairs from the 1000-molecule CAS(8,8)
production set, in three classes:

    isomer          same formula            composition descriptors IDENTICAL
    isoelectronic   same electron count,    formula-derived features differ
                    different formula       but total N is matched
    control         electron counts         composition trivially different
                    differing by >= 4

For a pair (A, B) at matched kT the screen is that of
`train_hybrid_spin.screening_score` specialised to two states:

    ratio = ||offdiag(rho_A - rho_B)||_F / ||diag(rho_A - rho_B)||_F

computed on the SAME shared 1024-determinant register the spin-label program
uses (`keep_idx` from spin_labels_kT0p1.npz), with each state trace-normalised
first (INVARIANTS.md I8).  Reference points on this data family: the S^2 label
screens at 0.13, its coherence-only part c at 0.18, the synthetic pure-
off-diagonal control at 0.81 (8-qubit register, hybrid_spin_metrics_8q.json).

The isomer class is the one that matters: two isomers share every formula-
derived descriptor (DoU included), so whatever separates their thermal states
is not composition wearing a disguise.

Usage:
    .venv/bin/python scripts/pair_screen.py \\
        --h5 results/qh9_dense_cas8-8_kT0p1.h5 \\
        --labels results/spin_labels_kT0p1.npz \\
        --json-out results/pair_screen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger("pair_screen")

Z_OF = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def electron_count(formula: str) -> int:
    tot = 0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            tot += Z_OF[el] * (int(n) if n else 1)
    return tot


class RhoCache:
    """LRU cache of trace-normalised 1024x1024 states on the shared register.

    Consistency check on every load: the kept trace must reproduce the
    `rho_trace_kept` spin_labels.py recorded, which pins this reconstruction
    to the one the whole label program used.
    """

    def __init__(self, h5path, keep_idx, kT_tag, trace_ref, maxsize=220):
        self.f = h5py.File(h5path, "r")
        self.keep = np.asarray(keep_idx)
        self.tag = kT_tag
        self.trace_ref = trace_ref                # idx -> expected kept trace
        self.maxsize = maxsize
        self._c = OrderedDict()
        self.loads = 0

    def rho(self, idx: int) -> np.ndarray:
        if idx in self._c:
            self._c.move_to_end(idx)
            return self._c[idx]
        g = self.f[f"mol_{idx}"][self.tag]
        V = np.asarray(g["civecs"][:], np.float64)          # (m, 4900)
        p = np.asarray(g["p"][:], np.float64)
        p = p / p.sum()          # spin_labels.py convention: full trace out first
        R = V[:, self.keep]                                  # (m, 1024)
        rho = (R * p[:, None]).T @ R                         # 1024 x 1024
        tr = float(np.trace(rho))
        ref = self.trace_ref.get(idx)
        if ref is not None and abs(tr - ref) > 1e-8:
            raise ValueError(f"mol_{idx}: kept trace {tr} != recorded {ref}")
        rho /= tr
        self._c[idx] = rho
        self.loads += 1
        if len(self._c) > self.maxsize:
            self._c.popitem(last=False)
        return rho


def pair_stats(rho_a, rho_b):
    d = rho_a - rho_b
    dg = float(np.linalg.norm(np.diag(d)))
    od = float(np.sqrt(max(np.linalg.norm(d) ** 2 - dg ** 2, 0.0)))
    return {"offdiag": od, "diag": dg, "ratio": od / (dg + 1e-12)}


def sample_pairs(idx, formulas, nelec, n_per_class, rng, per_family_cap=10):
    """Pair lists per class, reproducibly sampled.

    Isomer pairs are capped per formula family so C5H8O's 48 members do not
    dominate the statistics; isoelectronic and control pairs are rejection-
    sampled from the full index set.
    """
    by_formula = {}
    for k, f in enumerate(formulas):
        by_formula.setdefault(f, []).append(k)

    iso_pairs = []
    for f, members in sorted(by_formula.items()):
        if len(members) < 2:
            continue
        fam = [(a, b) for i, a in enumerate(members) for b in members[i + 1:]]
        rng.shuffle(fam)
        iso_pairs.extend(fam[:per_family_cap])
    rng.shuffle(iso_pairs)
    iso_pairs = iso_pairs[:n_per_class]

    def rejection(cond, n):
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < 200000:
            a, b = rng.integers(0, len(idx), 2)
            tries += 1
            if a == b or (min(a, b), max(a, b)) in seen:
                continue
            if cond(int(a), int(b)):
                seen.add((min(a, b), max(a, b)))
                out.append((int(a), int(b)))
        return out

    isoel = rejection(lambda a, b: nelec[a] == nelec[b]
                      and formulas[a] != formulas[b], n_per_class)
    control = rejection(lambda a, b: abs(nelec[a] - nelec[b]) >= 4, n_per_class)
    return {"isomer": iso_pairs, "isoelectronic": isoel, "control": control}


def summarize(rows):
    r = np.array([x["ratio"] for x in rows])
    return {"n": len(rows), "median": float(np.median(r)),
            "q25": float(np.quantile(r, 0.25)), "q75": float(np.quantile(r, 0.75)),
            "frac_ratio_ge_1": float((r >= 1).mean()),
            "frac_ratio_ge_0p5": float((r >= 0.5).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="results/qh9_dense_cas8-8_kT0p1.h5")
    ap.add_argument("--labels", default="results/spin_labels_kT0p1.npz")
    ap.add_argument("--kT-tag", default="kT_0p1000")
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--per-family-cap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json-out", default="results/pair_screen.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    L = np.load(args.labels, allow_pickle=False)
    idx = L["idx"]
    formulas = [f.decode() for f in L["formula"]]
    keep_idx = L["keep_idx"]
    nelec = np.array([electron_count(f) for f in formulas])
    trace_ref = {int(i): float(t) for i, t in zip(idx, L["rho_trace_kept"])}

    rng = np.random.default_rng(args.seed)
    pairs = sample_pairs(idx, formulas, nelec, args.n_per_class, rng,
                         args.per_family_cap)
    for cls, ps in pairs.items():
        logger.info("%-14s %d pairs", cls, len(ps))

    cache = RhoCache(args.h5, keep_idx, args.kT_tag, trace_ref)
    results = {}
    t0 = time.perf_counter()
    for cls, ps in pairs.items():
        rows = []
        for k, (a, b) in enumerate(ps):
            st = pair_stats(cache.rho(int(idx[a])), cache.rho(int(idx[b])))
            st.update({"idx_a": int(idx[a]), "idx_b": int(idx[b]),
                       "formula_a": formulas[a], "formula_b": formulas[b]})
            rows.append(st)
            if (k + 1) % 50 == 0:
                logger.info("%s %d/%d (%.0fs, %d loads)", cls, k + 1, len(ps),
                            time.perf_counter() - t0, cache.loads)
        results[cls] = {"summary": summarize(rows), "pairs": rows}
        logger.info("%-14s %s", cls, results[cls]["summary"])

    results["references"] = {
        "S2_label_screen_8q": 0.1326, "c_label_screen_8q": 0.1753,
        "synthetic_offdiag_control_8q": 0.8084,
        "ncas10_C2H2_vs_HCN": 1.1526,
        "note": "8q references from results/hybrid_spin_metrics_8q.json are on "
                "the 256-determinant register; this screen uses the 1024 one."}
    results["config"] = dict(vars(args))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(results, fh, indent=1)
    logger.info("wrote %s (%.0fs, %d molecule loads)", args.json_out,
                time.perf_counter() - t0, cache.loads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
