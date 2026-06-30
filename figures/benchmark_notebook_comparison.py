"""
Compare logloss.ipynb (Phase 1) vs logloss_pennylane.ipynb (Phase 2).

For n in {2, 4, 7}, N=1000 training states, 750 epochs:
  logloss.ipynb       Phase 1 original  (per-state dense loop)
                      Phase 1+3         (vectorised, ~7x faster)
                      probed ≤60 s; speed bar uses measured ms/epoch
  logloss_pennylane   Phase 2 quantum   (aggregate R+/R-, sparse CSR)
                      Phase 2 FCIM      (exact diagonal, no eigh)
                      full 750 epochs

Figure: 3 rows × 3 columns
  Row 1  Loss vs epoch (quantum model)
  Row 2  Val accuracy vs epoch (quantum model)
  Row 3  ms/epoch bar chart + projected 750-epoch wall-clock time
"""

from __future__ import annotations
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    raise RuntimeError("pennylane not found — run with .venv/bin/python3")

FIGURES_DIR   = Path(__file__).resolve().parent
PROBE_SECONDS = 60.0
TOTAL_EPOCHS  = 750
N_TRAIN       = 1000
N_VAL         = 500
T             = 2.0
ETA           = 0.1
SEED          = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

I2 = np.eye(2, dtype=complex)
PX = np.array([[0,1],[1,0]], dtype=complex)
PY = np.array([[0,-1j],[1j,0]], dtype=complex)
PZ = np.array([[1,0],[0,-1]], dtype=complex)

def krons(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def _pauli_dense_list(n, model):
    """Exact generate_paulis() from both notebooks."""
    paulis = []
    if model == "quantum":
        base = [PX, PY, PZ]
        for op in base:
            for i in range(n - 1):
                lst = [I2]*n; lst[i] = op; lst[i+1] = op
                paulis.append(krons(lst))
    elif model == "classical":
        base = [PZ]
        for op in base:
            for i, j in itertools.combinations(range(n), 2):
                lst = [I2]*n; lst[i] = op; lst[j] = op
                paulis.append(krons(lst))
    for op in base:
        for i in range(n):
            lst = [I2]*n; lst[i] = op
            paulis.append(krons(lst))
    return paulis

def _pauli_sparse_list(n, model):
    """Sparse CSR Pauli operators (from logloss_pennylane.ipynb)."""
    syms = []
    if model == "quantum":
        for P in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            for i in range(n - 1):
                syms.append(P(i) @ P(i+1))
        for P in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            for i in range(n): syms.append(P(i))
    elif model == "classical":
        for i, j in itertools.combinations(range(n), 2):
            syms.append(qml.PauliZ(i) @ qml.PauliZ(j))
        for i in range(n): syms.append(qml.PauliZ(i))
    wo = tuple(range(n))
    return [s.sparse_matrix(wire_order=wo, format="csr") for s in syms]

def _build_H_dense(w, sp):
    H = sp[0] * float(w[0])
    for wi, op in zip(w[1:], sp[1:]):
        H = H + op * float(wi)
    return H.toarray()

def _build_fcim_features(n):
    idx = np.arange(2**n, dtype=np.uint64)[:,None]
    sh  = np.arange(n-1, -1, -1, dtype=np.uint64)
    z   = 1 - 2*((idx >> sh) & 1).astype(float)
    cols = [z[:,i]*z[:,j] for i,j in itertools.combinations(range(n),2)]
    cols += [z[:,i] for i in range(n)]
    return np.column_stack(cols)

def _fdd(ev, y):
    """Divided-difference matrix of the logistic loss."""
    l, k = ev[:,None], ev[None,:]
    d = l - k
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (T*np.log1p(np.exp(-y*l/T)) - T*np.log1p(np.exp(-y*k/T))) / d
    return np.where(np.abs(d) < 1e-10, -y/(1+np.exp(y*l/T)), F)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION  (shared by all methods via same seed)
# ═══════════════════════════════════════════════════════════════════════════════

def make_data(n, rng):
    dim = 1 << n

    def rand_vecs(m):
        v = rng.standard_normal((m, dim)) + 1j*rng.standard_normal((m, dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    train_v = rand_vecs(N_TRAIN)
    val_v   = rand_vecs(N_VAL)

    # Random quantum target (mirrors logloss.ipynb's optimize())
    t_ops = _pauli_dense_list(n, "quantum")
    t_w   = rng.uniform(-2, 2, len(t_ops))
    H_t   = sum(w*op for w, op in zip(t_w, t_ops))

    def label(vecs):
        e = np.real(np.einsum("bi,ij,bj->b", vecs.conj(), H_t, vecs))
        y = np.sign(e); y[y == 0] = 1; return y

    train_y   = label(train_v)
    val_y     = label(val_v)
    train_rho = np.einsum("bi,bj->bij", train_v, train_v.conj())
    return dict(train_v=train_v, val_v=val_v,
                train_rho=train_rho, train_y=train_y, val_y=val_y)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1  (logloss.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def _p1_step_original(w, paulis, rhos, ys):
    """Exact per-state loop (use_fast_grad=False)."""
    H = sum(wi*p for wi,p in zip(w, paulis))
    ev, ec = np.linalg.eigh(H)
    N = len(rhos)
    loss = 0.0
    F_p = _fdd(ev, +1); F_m = _fdd(ev, -1)
    for i in range(N):
        m = ec @ np.diag(T*np.log1p(np.exp(-ys[i]*ev/T))) @ ec.T.conj()
        loss += np.real(np.trace(m @ rhos[i]))
    loss /= N
    grad = np.zeros(len(w))
    for j, pj in enumerate(paulis):
        Hj_t = ec.T.conj() @ (pj/T) @ ec
        g = 0.0
        for i in range(N):
            rho_t = ec.T.conj() @ rhos[i] @ ec
            F = F_p if ys[i] > 0 else F_m
            g += np.real(np.sum(F * Hj_t * rho_t.T))
        grad[j] = g / N
    return float(loss), grad

def _p1_step_fast(w, paulis, rhos, ys):
    """Phase 1+3 vectorised (use_fast_grad=True)."""
    H = sum(wi*p for wi,p in zip(w, paulis))
    ev, ec = np.linalg.eigh(H)
    rhos_t = np.array([ec.T.conj() @ r @ ec for r in rhos])
    rhos_d = np.real(np.diagonal(rhos_t, axis1=1, axis2=2))
    diag_p = T*np.log1p(np.exp(-ev/T))
    diag_m = T*np.log1p(np.exp( ev/T))
    dloss  = np.where(ys[:,None] > 0, diag_p, diag_m)
    loss   = float(np.mean(np.sum(dloss * rhos_d, axis=1)))
    F_p = _fdd(ev, +1); F_m = _fdd(ev, -1)
    grad = np.zeros(len(w))
    for j, pj in enumerate(paulis):
        Hj_t = ec.T.conj() @ pj @ ec
        gj = 0.0
        for i in range(len(rhos)):
            F = F_p if ys[i] > 0 else F_m
            gj += np.real(np.einsum("ij,ij,ij->", F, Hj_t, rhos_t[i].T))
        grad[j] = gj / len(rhos)
    return loss, grad

def _p1_acc(w, paulis, val_v, val_y):
    H = sum(wi*p for wi,p in zip(w, paulis))
    e = np.real(np.einsum("bi,ij,bj->b", val_v.conj(), H, val_v))
    p = np.sign(e); p[p==0] = 1
    return float(np.mean(p == val_y) * 100)

def run_phase1(n, data, flavour="fast"):
    ops_q = _pauli_dense_list(n, "quantum")
    ops_c = _pauli_dense_list(n, "classical")
    w_q   = np.random.default_rng(SEED+1000).uniform(-0.5, 0.5, len(ops_q))
    w_c   = np.random.default_rng(SEED+1001).uniform(-0.5, 0.5, len(ops_c))
    step  = _p1_step_fast if flavour == "fast" else _p1_step_original

    rhos  = data["train_rho"]; ys = data["train_y"]
    val_v = data["val_v"];     val_y = data["val_y"]

    loss_q, loss_c, acc_q, ep_times = [], [], [], []
    deadline = time.perf_counter() + PROBE_SECONDS
    ep = 0

    while ep < TOTAL_EPOCHS:
        if ep > 0 and time.perf_counter() >= deadline:
            break
        t0 = time.perf_counter()
        lq, gq = step(w_q, ops_q, rhos, ys)
        lc, gc = step(w_c, ops_c, rhos, ys)
        ep_times.append(time.perf_counter() - t0)
        w_q -= ETA*gq; w_c -= ETA*gc
        loss_q.append(lq); loss_c.append(lc)
        acc_q.append(_p1_acc(w_q, ops_q, val_v, val_y))
        ep += 1

    ms_ep = float(np.median(ep_times)) * 1000
    proj  = ms_ep * TOTAL_EPOCHS / 1000
    print(f"  {ep} epochs  {ms_ep:.0f} ms/ep  "
          f"proj 750ep = {proj/60:.1f} min ({proj/3600:.1f} hr)")
    return dict(loss_q=np.array(loss_q), loss_c=np.array(loss_c),
                acc_q=np.array(acc_q), measured=ep,
                ms_ep=ms_ep, proj_total_s=proj, epochs=np.arange(ep))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2  (logloss_pennylane.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def _agg(vecs, ys):
    N = len(vecs)
    Rp = np.einsum("bi,bj->ij", vecs[ys>0], vecs[ys>0].conj()) / N
    Rm = np.einsum("bi,bj->ij", vecs[ys<0], vecs[ys<0].conj()) / N
    return Rp, Rm

def _p2_step_q(w, sp, Rp, Rm):
    H  = _build_H_dense(w, sp)
    ev, ec = np.linalg.eigh(H)
    Rpt = ec.conj().T @ Rp @ ec
    Rmt = ec.conj().T @ Rm @ ec
    loss = float(np.real(
        np.dot(T*np.logaddexp(0,-ev/T), np.diag(Rpt)) +
        np.dot(T*np.logaddexp(0, ev/T), np.diag(Rmt))
    ))
    dE  = _fdd(ev,+1)*Rpt.T + _fdd(ev,-1)*Rmt.T
    dM  = ec.conj() @ dE @ ec.T
    grad = np.array([float(np.real(op.multiply(dM).sum())) for op in sp])
    return loss, grad

def _p2_step_c(w, feat, pp, pm):
    e    = feat @ w
    loss = float(np.real(pp @ (T*np.logaddexp(0,-e/T)) +
                          pm @ (T*np.logaddexp(0, e/T))))
    dp   = -np.exp(-np.logaddexp(0,  e/T))
    dm   =  np.exp(-np.logaddexp(0, -e/T))
    return loss, np.real(feat.T @ (pp*dp + pm*dm))

def _p2_acc_q(w, sp, vv, yv):
    H = _build_H_dense(w, sp)
    e = np.real(np.einsum("bi,ij,bj->b", vv.conj(), H, vv))
    p = np.sign(e); p[p==0] = 1
    return float(np.mean(p == yv) * 100)

def _p2_acc_c(w, feat, vv, yv):
    e = np.abs(vv)**2 @ (feat @ w)
    p = np.sign(np.real(e)); p[p==0] = 1
    return float(np.mean(p == yv) * 100)

def run_phase2(n, data):
    sp_q  = _pauli_sparse_list(n, "quantum")
    feat  = _build_fcim_features(n)
    w_q   = np.random.default_rng(SEED+1000).uniform(-0.5, 0.5, len(sp_q))
    w_c   = np.random.default_rng(SEED+1001).uniform(-0.5, 0.5, feat.shape[1])

    vt = data["train_v"]; yt = data["train_y"]
    vv = data["val_v"];   yv = data["val_y"]
    Rp, Rm = _agg(vt, yt)
    pp = (np.abs(vt[yt>0])**2).sum(0) / N_TRAIN
    pm = (np.abs(vt[yt<0])**2).sum(0) / N_TRAIN

    loss_q, loss_c, acc_q, acc_c = [], [], [], []
    tq, tc = [], []
    for ep in range(TOTAL_EPOCHS):
        t0 = time.perf_counter(); lq, gq = _p2_step_q(w_q, sp_q, Rp, Rm); tq.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); lc, gc = _p2_step_c(w_c, feat, pp, pm);  tc.append(time.perf_counter()-t0)
        w_q -= ETA*gq; w_c -= ETA*gc
        loss_q.append(lq); loss_c.append(lc)
        acc_q.append(_p2_acc_q(w_q, sp_q, vv, yv))
        acc_c.append(_p2_acc_c(w_c, feat, vv, yv))
        if ep % 150 == 0 or ep == TOTAL_EPOCHS-1:
            print(f"  n={n} ep {ep:4d}  Q {lq:.4f} ({acc_q[-1]:.1f}%)"
                  f"  C {lc:.4f} ({acc_c[-1]:.1f}%)"
                  f"  {np.median(tq)*1000:.1f}/{np.median(tc)*1000:.2f} ms/ep")

    return dict(
        loss_q=np.array(loss_q), loss_c=np.array(loss_c),
        acc_q=np.array(acc_q),   acc_c=np.array(acc_c),
        ms_q=float(np.median(tq))*1000,
        ms_c=float(np.median(tc))*1000,
        epochs=np.arange(TOTAL_EPOCHS),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    n_list  = [2, 4, 7]
    results = {}

    for n in n_list:
        print(f"\n{'='*65}  n={n}  dim={1<<n}")
        rng  = np.random.default_rng(SEED + n)
        data = make_data(n, rng)

        print("[Phase 2 – logloss_pennylane.ipynb]")
        p2 = run_phase2(n, data)

        p1o = None
        if n <= 2:
            print(f"[Phase 1 original – logloss.ipynb]  probing ≤{PROBE_SECONDS:.0f}s")
            p1o = run_phase1(n, data, flavour="original")

        print(f"[Phase 1+3 – logloss.ipynb]  probing ≤{PROBE_SECONDS:.0f}s")
        p1f = run_phase1(n, data, flavour="fast")

        results[n] = dict(p2=p2, p1f=p1f, p1o=p1o)

    make_figure(results, n_list)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def _proj_label(ms_ep):
    s = ms_ep * TOTAL_EPOCHS / 1000
    if s < 60:    return f"{s:.1f} s"
    if s < 3600:  return f"{s/60:.1f} min"
    if s < 86400: return f"{s/3600:.1f} hr"
    return f"{s/86400:.1f} days"

def make_figure(results, n_list):
    fig, axes = plt.subplots(3, len(n_list),
                             figsize=(5.3*len(n_list), 12),
                             constrained_layout=True)

    ep_full = np.arange(TOTAL_EPOCHS)

    for col, n in enumerate(n_list):
        r   = results[n]
        p2  = r["p2"]; p1f = r["p1f"]; p1o = r["p1o"]
        ax_loss, ax_acc, ax_spd = axes[0,col], axes[1,col], axes[2,col]

        # ── Loss ─────────────────────────────────────────────────────────────
        ax_loss.plot(ep_full, p2["loss_q"], "#0057B8", lw=1.8,
                     label="Phase 2 quantum (logloss_pennylane)")
        ax_loss.plot(ep_full, p2["loss_c"], "#D62728", lw=1.8, ls="--",
                     label="Phase 2 FCIM (logloss_pennylane)")
        if len(p1f["loss_q"]):
            ax_loss.plot(p1f["epochs"], p1f["loss_q"], "#2CA02C", lw=2, ls="-",
                         label=f"Phase 1+3 Q (logloss.ipynb, {p1f['measured']} ep)")
            ax_loss.plot(p1f["epochs"], p1f["loss_c"], "#FF7F0E", lw=2, ls="--",
                         label=f"Phase 1+3 C (logloss.ipynb, {p1f['measured']} ep)")
        if p1o is not None and len(p1o["loss_q"]):
            ax_loss.plot(p1o["epochs"], p1o["loss_q"], "#9467BD", lw=1.5, ls=":",
                         label=f"Phase 1 orig Q ({p1o['measured']} ep)")
            ax_loss.plot(p1o["epochs"], p1o["loss_c"], "#8C564B", lw=1.5, ls=":",
                         label=f"Phase 1 orig C ({p1o['measured']} ep)")
        if p1f["measured"] < TOTAL_EPOCHS:
            for ax_ in [ax_loss, ax_acc]:
                ax_.axvline(p1f["measured"]-0.5, color="#2CA02C",
                            ls=":", lw=1.0, alpha=0.7)
                ax_.text(p1f["measured"]+5, 0.98, "← probe\n   stops",
                         transform=ax_.get_xaxis_transform(),
                         va="top", fontsize=7, color="#2CA02C", alpha=0.85)
        ax_loss.set_title(f"n = {n}   dim = {1<<n}", fontsize=11, fontweight="bold")
        ax_loss.grid(True, alpha=0.25)
        ax_loss.set_xlabel("Epoch", fontsize=9)

        # ── Accuracy ─────────────────────────────────────────────────────────
        ax_acc.plot(ep_full, p2["acc_q"],  "#0057B8", lw=1.8, label="Phase 2 quantum")
        ax_acc.plot(ep_full, p2["acc_c"],  "#D62728", lw=1.8, ls="--", label="Phase 2 FCIM")
        if len(p1f["acc_q"]):
            ax_acc.plot(p1f["epochs"], p1f["acc_q"], "#2CA02C", lw=2,
                        label="Phase 1+3 Q (probe)")
        if p1o is not None and len(p1o["acc_q"]):
            ax_acc.plot(p1o["epochs"], p1o["acc_q"], "#9467BD", lw=1.5, ls=":",
                        label="Phase 1 orig Q (probe)")
        ax_acc.axhline(50, color="gray", ls="-.", lw=0.8, alpha=0.5)
        ax_acc.set_ylim(30, 105)
        ax_acc.grid(True, alpha=0.25)
        ax_acc.set_xlabel("Epoch", fontsize=9)

        # ── Speed bar ─────────────────────────────────────────────────────────
        methods, ms_vals, bar_colors = [], [], []
        if p1o is not None:
            methods.append("Phase 1\norig"); ms_vals.append(p1o["ms_ep"]); bar_colors.append("#9467BD")
        methods.append("Phase 1+3\n(logloss.ipynb)"); ms_vals.append(p1f["ms_ep"]); bar_colors.append("#2CA02C")
        methods.append("Phase 2 Q\n(PL)");           ms_vals.append(p2["ms_q"]);    bar_colors.append("#0057B8")
        methods.append("Phase 2 C\n(PL FCIM)");      ms_vals.append(p2["ms_c"]);    bar_colors.append("#D62728")

        xs   = np.arange(len(methods))
        bars = ax_spd.bar(xs, ms_vals, color=bar_colors, edgecolor="white",
                          linewidth=0.6, alpha=0.88)
        ax_spd.set_yscale("log")
        ax_spd.set_xticks(xs); ax_spd.set_xticklabels(methods, fontsize=8)
        ax_spd.grid(True, alpha=0.25, axis="y")

        for bar, ms in zip(bars, ms_vals):
            ax_spd.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.7,
                        f"750ep:\n{_proj_label(ms)}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        fastest = ms_vals[-1]
        for bar, ms in zip(bars[:-1], ms_vals[:-1]):
            xf = ms / fastest
            if xf > 5:
                ax_spd.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() * 0.35,
                            f"{xf:.0f}×",
                            ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")

    axes[0, 0].set_ylabel("Logistic loss",        fontsize=10)
    axes[1, 0].set_ylabel("Val accuracy (%)",     fontsize=10)
    axes[2, 0].set_ylabel("Time per epoch  (ms)", fontsize=10)

    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, fontsize=8, framealpha=0.9)

    fig.suptitle(
        "logloss.ipynb (Phase 1 / 1+3)  vs  logloss_pennylane.ipynb (Phase 2)\n"
        f"N={N_TRAIN} training states · {TOTAL_EPOCHS} epochs target · T={T} · η={ETA}\n"
        "Phase 1 probed ≤60 s (speed bar extrapolates to full 750 epochs)",
        fontsize=9, y=1.06,
    )

    out = FIGURES_DIR / "notebook_comparison_2_4_7.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved → {out}")


if __name__ == "__main__":
    main()
