"""Screen the second-quantization labels with the diagnostic that killed the gap.

`second_quantization_labels.py` produced, for all 1000 CAS(8,8) molecules,
labels that have **no mean-field analogue**: correlation corrections to the
ionization energy, electron affinity and singlet-triplet gap, the quasiparticle
pole strength, and the Head-Gordon unpaired-electron count.  Each is identically
zero (or exactly 1, for `Z_pole`) for any single-determinant state.

The gap audit established the right test, and it is *not* "is the label
off-diagonal per state".  It is:

  1. **Ladder.** Does the coherence channel (XX/YY, 112) beat chance, and does
     adding it to the diagonal pool (Z/ZZ, 136) buy anything?
  2. **Residual.** After an out-of-fold model of `diag(rho)`, is what remains
     predictable from coherence, or only from composition?
  3. **Confound.** How much does composition alone get?

A label passes only if the coherence channel carries signal the diagonal has not
already supplied.  The mean-field HOMO-LUMO gap is carried through as the
reference point that failed.

Writes ``results/sq_label_screen.json``.

    PYTHONPATH=scripts .venv/bin/python scripts/sq_label_screen.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import gap_diagnosis as G
from gap_diagnosis_followup import cached_load

REPO = Path(__file__).resolve().parents[1]
SQ = REPO / "results/second_quantization_labels.npz"
OUT = REPO / "results/sq_label_screen.json"
HA_EV = G.HA_EV

# label -> (human name, does it vanish for a single determinant?)
LABELS = {
    "IP_corr": ("correlation correction to the IP", True),
    "EA_corr": ("correlation correction to the EA", True),
    "ST_corr": ("correlation correction to the S-T gap", True),
    "Z_pole": ("quasiparticle pole strength Z", True),
    "N_unpaired": ("unpaired-electron count (Head-Gordon)", True),
    "w_double": ("double-excitation weight of the ground state", True),
    "E_corr_neutral": ("active-space correlation energy", True),
    "IP_cas": ("ionization energy (correlated)", False),
    "ST_cas": ("singlet-triplet gap (correlated)", False),
    "EA_cas": ("electron affinity (correlated)", False),
}


def screen(name, y_cont, X, Xd, Xo, Xdesc, seeds=25):
    """The full diagnostic for one continuous label."""
    y = (y_cont <= np.median(y_cont)).astype(int)
    res = {"n": int(len(y)),
           "spread": dict(std=float(np.std(y_cont)),
                          iqr=float(np.subtract(*np.percentile(y_cont, [75, 25]))),
                          median=float(np.median(y_cont)))}

    a_desc, _, _ = G.eval_classifier(Xdesc, y, seeds)
    a_diag, _, _ = G.eval_classifier(Xd, y, seeds)
    a_full, u_full, _ = G.eval_classifier(X, y, seeds)
    a_off, u_off, _ = G.eval_classifier(Xo, y, seeds)
    res["acc"] = {k: G.summ(v * 100) for k, v in
                  dict(composition=a_desc, diagonal=a_diag, full=a_full,
                       offdiag_only=a_off).items()}
    res["auc_offdiag_only"] = G.summ(u_off)
    d = (a_full - a_diag) * 100
    res["quantum_minus_classical"] = dict(
        **G.summ(d), p=float(stats.ttest_rel(a_full, a_diag).pvalue))

    # regression + the residual test
    r2d, _ = G.eval_regressor(Xd, y_cont, seeds)
    r2f, _ = G.eval_regressor(X, y_cont, seeds)
    r2o, _ = G.eval_regressor(Xo, y_cont, seeds)
    res["r2"] = dict(diagonal=G.summ(r2d), full=G.summ(r2f),
                     offdiag_only=G.summ(r2o),
                     delta=G.summ(r2f - r2d))

    resid = y_cont - G.oof_predictions(Xd, y_cont)
    r2_res, _ = G.eval_regressor(Xo, resid, seeds)
    res["residual"] = dict(
        r2_diag_oof=float(1 - np.var(resid) / np.var(y_cont)),
        r2_offdiag_on_residual=G.summ(r2_res))
    yr = (resid <= np.median(resid)).astype(int)
    for tag, F in (("composition", Xdesc), ("diagonal", Xd), ("offdiag", Xo)):
        a, _, _ = G.eval_classifier(F, yr, seeds)
        res["residual"].setdefault("acc", {})[tag] = G.summ(a * 100)

    print(f"  {name:44s} desc {np.mean(a_desc)*100:5.1f}  diag {np.mean(a_diag)*100:5.1f}"
          f"  full {np.mean(a_full)*100:5.1f}  off {np.mean(a_off)*100:5.1f}"
          f"  Q-C {np.mean(d):+5.2f}  R2diag {np.mean(r2d):6.3f}"
          f"  resid-off {np.mean(r2_res):+.4f}", flush=True)
    return res


def main() -> None:
    D = cached_load()
    z = np.load(SQ)
    order = {int(i): k for k, i in enumerate(z["idx"])}
    sel = np.array([order[i] for i in D["idx"]])
    assert np.array_equal(z["idx"][sel], D["idx"])
    assert bool(np.all(z["ok"][sel])), "some molecules failed to solve"

    X = D["X"]
    Xd, Xo = X[:, :G.N_DIAG], X[:, G.N_DIAG:]
    desc = D["desc"].item() if hasattr(D["desc"], "item") else D["desc"]
    B = np.column_stack([desc[k] for k in sorted(desc)])
    Xdesc = np.column_stack([B, B ** 2,
                             *[B[:, i] * B[:, j] for i in range(B.shape[1])
                               for j in range(i + 1, B.shape[1])]])

    out = {"labels": {}, "units": "IP/EA/ST and their corrections in Hartree"}
    print("label ladders (held-out %, 25 splits):")
    out["labels"]["mean-field HOMO-LUMO gap (reference)"] = screen(
        "mean-field HOMO-LUMO gap (reference)", D["gap"], X, Xd, Xo, Xdesc)
    for key, (name, vanishes) in LABELS.items():
        r = screen(name, z[key][sel], X, Xd, Xo, Xdesc)
        r["vanishes_for_single_determinant"] = vanishes
        r["raw_key"] = key
        out["labels"][name] = r

    # how big are these effects, physically?
    out["magnitudes_eV"] = {
        k: dict(median=float(np.median(z[k][sel]) * HA_EV),
                p5=float(np.percentile(z[k][sel], 5) * HA_EV),
                p95=float(np.percentile(z[k][sel], 95) * HA_EV),
                std=float(np.std(z[k][sel]) * HA_EV))
        for k in ("IP_corr", "EA_corr", "ST_corr", "IP_cas", "ST_cas",
                  "E_corr_neutral")}
    out["Z_pole"] = dict(median=float(np.median(z["Z_pole"][sel])),
                         p5=float(np.percentile(z["Z_pole"][sel], 5)),
                         min=float(np.min(z["Z_pole"][sel])))

    # how much of each label does the *mean-field* part already carry?
    for a, b in (("IP_cas", "IP_mf"), ("EA_cas", "EA_mf"), ("ST_cas", "ST_mf")):
        out.setdefault("mean_field_share", {})[a] = dict(
            pearson=float(stats.pearsonr(z[a][sel], z[b][sel]).statistic),
            r2=float(stats.pearsonr(z[a][sel], z[b][sel]).statistic ** 2))

    # and the composition confound, directly
    dou = desc["DoU"]
    out["confound_vs_DoU"] = {
        k: float(stats.spearmanr(z[k][sel], dou).statistic)
        for k in LABELS}
    out["confound_vs_gap"] = {
        k: float(stats.spearmanr(z[k][sel], D["gap"]).statistic)
        for k in LABELS}

    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
