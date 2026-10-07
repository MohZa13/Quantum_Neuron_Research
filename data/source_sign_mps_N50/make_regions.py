"""Cut R-region states out of the full-chain source-sign MPS dataset.

Shipped next to the data as ``make_regions.py``; needs numpy and h5py only
(plus ``source_sign_mps_reader.py`` from the same directory).  The README
beside it explains the physics and every convention used below.

A region R is fixed by k (number of qubits) and d (offset of its centre from
the rotated qubit j0 = 25): the qubits 25 + d - (k-1)/2 ... 25 + d + (k-1)/2.
For every selected state (Hamiltonian, s, t) and every requested d, this
writes rho_R = Tr_{not R} |psi><psi| as one training sample.

    python make_regions.py --hamiltonian XXX --k 3 --d -5:5 --out XXX_k3.h5
    python make_regions.py --hamiltonian XX --k 7 --d all --t 0:6:0.2 --out XX_k7.h5
    python make_regions.py --selftest            # run this first on a new machine

Selectors (inclusive ranges):
    --d   all | -5:5 | 0,2,4 | -10:10:2 | mixtures like -3:3,10
    --t   all | 0:4 | 0:13:0.5 | 1.0,2.5     (times must exist in the dataset:
                                              multiples of 0.1 from 0 to 13)
    --signs  +1,-1 (default, both labels) | +1 | -1

Output (HDF5, one sample per (state, d)), sample order: t, then d, then
s = +1 before s = -1, so a cell's two labels are adjacent and share pair_id:
    rho             (M, 2^k, 2^k) complex128   little-endian, [ket, bra], unit trace
    s, t, step      label, time, Trotter step
    d, first_site, dist          region offset, its first qubit (1-based), and
                                 the number of qubits between j0 and R
    pair_id         the (t, d) cell; each value appears once per label
    in_strict_cone  dist <= v t  (v = 2 for XX, pi for XXX)
    trace_distance, helstrom     1/2 ||rho_+ - rho_-||_1 of the cell and the best
                                 single-shot accuracy 1/2 (1 + TD); only when
                                 both signs are written
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VELOCITY = {"XX": 2.0, "XXX": math.pi}

# Reference values for --selftest, computed from this dataset with the shipped
# reader (2026-10-01).  A correct reader on any machine reproduces them to
# ~1e-12; the tolerance is 1e-8.  Columns: hamiltonian, k, d, t,
#   td   = 1/2 ||rho_+ - rho_-||_1           (label-blind: catches wrong states)
#   z0   = <Z on qubit 0 of R>   for s = +1   (site 25 + d - (k-1)/2)
#   zl   = <Z on qubit k-1 of R> for s = +1   (catches a reversed qubit order)
#   j01  = <X0 Y1 - Y0 X1>       for s = +1   (odd under s AND under rho -> rho^T:
#                                              catches swapped labels and a
#                                              transposed / conjugated rho)
REFERENCE = [
    {"hamiltonian": "XX", "k": 1, "d": 0, "t": 0.0, "td": 0.0, "z0": -1.4432899320127035e-15, "zl": -1.4432899320127035e-15, "j01": 0.0},
    {"hamiltonian": "XX", "k": 3, "d": 0, "t": 0.0, "td": 0.9008600214528713, "z0": 3.1086244689504383e-15, "zl": 5.051514762044462e-15, "j01": -1.2341506668104296},
    {"hamiltonian": "XX", "k": 3, "d": 1, "t": 1.0, "td": 0.7290307829363699, "z0": -0.3542053377392841, "zl": 0.101514874515963, "j01": -0.908106061279462},
    {"hamiltonian": "XX", "k": 5, "d": -4, "t": 3.0, "td": 0.4321998323899763, "z0": 0.05967349967436114, "zl": 0.16719884256639175, "j01": -0.1009663018190034},
    {"hamiltonian": "XX", "k": 7, "d": 10, "t": 6.0, "td": 0.31737194462338425, "z0": 0.07160439925612698, "zl": 0.04166671098069552, "j01": 0.10098218604739816},
    {"hamiltonian": "XX", "k": 3, "d": 20, "t": 2.0, "td": 1.9252452236736452e-10, "z0": -2.296213219565857e-11, "zl": -8.077846724852122e-11, "j01": -1.537773727016446e-11},
    {"hamiltonian": "XXX", "k": 1, "d": 0, "t": 0.0, "td": 0.0, "z0": -7.97695243193175e-14, "zl": -7.97695243193175e-14, "j01": 0.0},
    {"hamiltonian": "XXX", "k": 3, "d": 0, "t": 0.0, "td": 0.8402453634098432, "z0": 5.928590951498336e-14, "zl": 6.500355809180292e-14, "j01": -1.0584239689959116},
    {"hamiltonian": "XXX", "k": 3, "d": 1, "t": 1.0, "td": 0.5855416989609371, "z0": -0.26882698974362956, "zl": -0.25393260809611384, "j01": -0.6212130894658714},
    {"hamiltonian": "XXX", "k": 5, "d": -4, "t": 3.0, "td": 0.3042477416185817, "z0": -0.20755701272752597, "zl": -0.07415012562056483, "j01": 0.09914068653362194},
    {"hamiltonian": "XXX", "k": 7, "d": 10, "t": 6.0, "td": 0.1770439166485077, "z0": 0.037597421905034234, "zl": 0.03772615461422796, "j01": 0.029195826278690933},
    {"hamiltonian": "XXX", "k": 3, "d": 20, "t": 2.0, "td": 7.447833986307388e-09, "z0": 6.786179368445389e-09, "zl": 4.026923239308644e-11, "j01": 1.0030090976447194e-09},
]


def _parse_ints(spec: str, lo: int, hi: int) -> list[int]:
    if spec.strip() == "all":
        return list(range(lo, hi + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            f = [int(x) for x in part.split(":")]
            a, b, st = (f + [1])[:3]
            out.extend(range(a, b + (1 if st > 0 else -1), st))
        else:
            out.append(int(part))
    bad = [d for d in out if not lo <= d <= hi]
    if bad:
        raise SystemExit(f"d values {bad} put the region off the chain; valid: {lo} .. {hi}")
    return sorted(set(out))


def _parse_times(spec: str, available) -> list[float]:
    avail = sorted(set(available))
    if spec.strip() == "all":
        return avail
    want = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            f = [float(x) for x in part.split(":")]
            a, b = f[0], f[1]
            if len(f) == 3:
                n = int(round((b - a) / f[2]))
                want.extend(a + i * f[2] for i in range(n + 1))
            else:
                want.extend(t for t in avail if a - 1e-9 <= t <= b + 1e-9)
        else:
            want.append(float(part))
    out = []
    for w in want:
        hit = [t for t in avail if abs(t - w) < 1e-6]
        if not hit:
            raise SystemExit(f"t = {w} is not in the dataset (steps of 0.1 from 0 to 13)")
        out.append(hit[0])
    return sorted(set(out))


def _work(job):
    """One state -> its requested regions.  Runs in a worker process."""
    import source_sign_mps_reader as R
    path, k, ds = job
    psi = R.load(path)
    return {d: rho for d, _, rho in R.all_regions(psi, k, ds=ds)}


def geometry(root) -> tuple[int, int]:
    """(j0, n_qubits) of the dataset, read from its first state (25, 50 here)."""
    import h5py
    import source_sign_mps_reader as R
    row = R.manifest(root)[0]
    with h5py.File(Path(root) / row["file"], "r") as f:
        return int(f.attrs["j0"]), int(f.attrs.get("n_qubits", len(f["mps"])))


def make(root, hamiltonian, k, ds, times, signs, out, workers=1, little_endian=True):
    import h5py
    import numpy as np
    import source_sign_mps_reader as R

    root = Path(root)
    J0, N = geometry(root)
    off = (k - 1) // 2
    rows = [r for r in R.manifest(root) if r["hamiltonian"] == hamiltonian and r["s"] in signs
            and any(abs(r["t"] - t) < 1e-6 for t in times)]
    if not rows:
        raise SystemExit("nothing selected")
    t_list = sorted({r["t"] for r in rows})
    t_index = {t: i for i, t in enumerate(t_list)}
    d_index = {d: i for i, d in enumerate(ds)}
    sign_order = [s for s in (1, -1) if s in signs]
    n_s, n_d = len(sign_order), len(ds)
    M, K = len(t_list) * n_d * n_s, 2 ** k

    def sample(t, d, s):
        return (t_index[t] * n_d + d_index[d]) * n_s + sign_order.index(s)

    s_arr = np.zeros(M, np.int8)
    t_arr = np.zeros(M)
    step_arr = np.zeros(M, np.int16)
    d_arr = np.zeros(M, np.int16)
    for r in rows:
        for d in ds:
            i = sample(r["t"], d, r["s"])
            s_arr[i], t_arr[i], step_arr[i], d_arr[i] = r["s"], r["t"], r["step"], d
    dist = np.maximum(0, np.abs(d_arr) - off).astype(np.int16)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        rho_ds = f.create_dataset("rho", (M, K, K), dtype=np.complex128,
                                  chunks=(1, K, K), compression="gzip", compression_opts=4)
        jobs = [(str(root / r["file"]), k, ds) for r in rows]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            ex = ProcessPoolExecutor(workers)
            results = ex.map(_work, jobs, chunksize=1)
        else:
            results = map(_work, jobs)
        for n_done, (r, regions) in enumerate(zip(rows, results), start=1):
            for d, rho in regions.items():
                rho_ds[sample(r["t"], d, r["s"])] = rho if little_endian else _to_big(rho, k)
            if n_done % 10 == 0 or n_done == len(rows):
                print(f"  {n_done}/{len(rows)} states  ({time.time() - t0:.0f} s)", flush=True)
        if workers > 1:
            ex.shutdown()

        f["s"], f["t"], f["step"], f["d"] = s_arr, t_arr, step_arr, d_arr
        f["first_site"] = (J0 + d_arr - off).astype(np.int16)
        f["dist"] = dist
        f["pair_id"] = (np.arange(M) // n_s).astype(np.int32)
        f["in_strict_cone"] = dist <= VELOCITY[hamiltonian] * t_arr + 1e-9
        if n_s == 2:
            td = np.empty(M // 2)
            for c in range(M // 2):
                td[c] = 0.5 * np.abs(np.linalg.eigvalsh(rho_ds[2 * c] - rho_ds[2 * c + 1])).sum()
            f["trace_distance"] = np.repeat(td, 2)
            f["helstrom"] = 0.5 * (1 + np.repeat(td, 2))
        a = f.attrs
        a["hamiltonian"], a["k"], a["j0"], a["n_qubits"] = hamiltonian, k, J0, N
        a["d_list"], a["t_list"], a["signs"] = np.array(ds), np.array(t_list), np.array(sign_order)
        a["basis"] = ("little-endian: qubit q of the region is site first_site + q and bit q of "
                      "the row index; bit 0 = spin up (sigma^z = +1); rho[ket, bra]"
                      if little_endian else
                      "big-endian (kron order): first_site is the MOST significant bit; "
                      "bit 0 = spin up; rho[ket, bra]")
        a["region"] = f"qubits j0 + d - (k-1)/2 .. j0 + d + (k-1)/2, 1-based, j0 = {J0}"
        a["label"] = f"s = +1 / -1: V_s = exp(-i s pi sigma^z_{J0} / 4) applied before evolving"
        a["source_dataset"] = str(root.resolve().name)
        a["generator"] = "make_regions.py"
        a["created"] = _dt.datetime.now().isoformat(timespec="seconds")
        a["n_samples"] = M
        a["schema_version"] = 1
    os.replace(tmp, out)
    print(f"wrote {out}: {M} samples of {K}x{K} ({time.time() - t0:.0f} s)")
    return out


def _to_big(rho, k):
    import source_sign_mps_reader as R
    perm = R._bit_reverse(k)
    return rho[perm][:, perm]


def _pauli(k, ops):
    """Little-endian Pauli string: ops = {qubit: 'X'|'Y'|'Z'}; kron factors reversed."""
    import numpy as np
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0])}
    out = np.array([[1.0 + 0j]])
    for q in reversed(range(k)):
        out = np.kron(out, P[ops.get(q, "I")])
    return out


def reference_values(root, cells):
    """td, z0, zl, j01 for each (hamiltonian, k, d, t) -- the selftest quantities."""
    import numpy as np
    import source_sign_mps_reader as R
    rows = R.manifest(root)
    out = []
    for h, k, d, t in cells:
        rho = {}
        for s in (1, -1):
            row = next(r for r in rows if r["hamiltonian"] == h and r["s"] == s
                       and abs(r["t"] - t) < 1e-6)
            rho[s] = R.region_rho(R.load(Path(root) / row["file"]), k, d)
        p, m = rho[1], rho[-1]
        td = 0.5 * np.abs(np.linalg.eigvalsh(p - m)).sum()
        z0 = np.trace(p @ _pauli(k, {0: "Z"})).real
        zl = np.trace(p @ _pauli(k, {k - 1: "Z"})).real
        j01 = (np.trace(p @ (_pauli(k, {0: "X", 1: "Y"}) - _pauli(k, {0: "Y", 1: "X"}))).real
               if k > 1 else 0.0)
        out.append({"hamiltonian": h, "k": k, "d": d, "t": t, "td": float(td), "z0": float(z0),
                    "zl": float(zl), "j01": float(j01), "rho": rho})
    return out


def selftest(root, tol=1e-8) -> bool:
    import numpy as np
    import source_sign_mps_reader as R
    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name} {detail}")

    print(f"selftest on {Path(root).resolve()}")
    rows = R.manifest(root)
    report("manifest has 524 states", len(rows) == 524, f"({len(rows)})")
    for h in ("XX", "XXX"):
        for t in (0.0, 13.0):
            row = next(r for r in rows if r["hamiltonian"] == h and r["s"] == 1 and r["t"] == t)
            dev = R.check(R.load(Path(root) / row["file"]))
            report(f"{row['file']} matches its stored checks", max(dev.values()) < 1e-10,
                   f"(worst {max(dev.values()):.1e})")
    got = reference_values(root, [(c["hamiltonian"], c["k"], c["d"], c["t"]) for c in REFERENCE])
    for ref, g in zip(REFERENCE, got):
        dev = max(abs(g[q] - ref[q]) for q in ("td", "z0", "zl", "j01"))
        tag = f"{ref['hamiltonian']} k={ref['k']} d={ref['d']:+d} t={ref['t']}"
        report(f"reference values {tag}", dev < tol, f"(dev {dev:.1e}, td {g['td']:.6f})")
        p, m = g["rho"][1], g["rho"][-1]
        phys = max(abs(np.trace(p) - 1), np.abs(p - p.conj().T).max(), -np.linalg.eigvalsh(p).min())
        flip = np.abs(m - R.flip_register(p)).max()
        report(f"  rho Hermitian, unit trace, PSD; rho_- = X..X rho_+ X..X", max(phys, flip) < 1e-9,
               f"(phys {phys:.1e}, flip {flip:.1e})")
        if ref["t"] == 0.0:
            cj = np.abs(m - p.conj()).max()
            report("  t = 0: rho_- = conj(rho_+)", cj < 1e-9, f"({cj:.1e})")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="See README.md for the conventions.")
    ap.add_argument("--root", default=str(HERE), help="dataset directory (default: this one)")
    ap.add_argument("--hamiltonian", choices=["XX", "XXX"])
    ap.add_argument("--k", type=int, help="region size (odd: 1, 3, 5, 7, ...)")
    ap.add_argument("--d", default="all", help="offsets of the region centre from qubit 25")
    ap.add_argument("--t", default="all", help="times (default: all 131)")
    ap.add_argument("--signs", default="+1,-1")
    ap.add_argument("--out", help="output .h5 path")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (states)")
    ap.add_argument("--threads", type=int, default=0,
                    help="BLAS threads per process (default: cores / workers)")
    ap.add_argument("--big-endian", action="store_true",
                    help="kron order instead of little-endian (first qubit most significant)")
    ap.add_argument("--selftest", action="store_true", help="verify the dataset and this machine")
    ap.add_argument("--print-reference", action="store_true", help=argparse.SUPPRESS)
    # "--d -5:5" would otherwise be read as an unknown option: bind values that
    # start with "-" to their selector so negative offsets work without "=".
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 1):
        if argv[i] in ("--d", "--t", "--signs") and argv[i + 1].startswith("-"):
            argv[i], argv[i + 1] = f"{argv[i]}={argv[i + 1]}", ""
    a = ap.parse_args([x for x in argv if x != ""])

    threads = a.threads or max(1, (os.cpu_count() or 1) // max(1, a.workers))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))    # before numpy is first imported
    sys.path.insert(0, str(HERE))

    if a.print_reference:
        import json
        cells = [(h, k, d, t) for h in ("XX", "XXX") for k, d, t in
                 ((1, 0, 0.0), (3, 0, 0.0), (3, 1, 1.0), (5, -4, 3.0), (7, 10, 6.0), (3, 20, 2.0))]
        vals = reference_values(a.root, cells)
        print(json.dumps([{q: v[q] for q in ("hamiltonian", "k", "d", "t", "td", "z0", "zl", "j01")}
                          for v in vals], indent=1))
        return
    if a.selftest:
        sys.exit(0 if selftest(a.root) else 1)
    if not (a.hamiltonian and a.k and a.out):
        ap.error("--hamiltonian, --k and --out are required (or use --selftest)")
    if a.k < 1 or a.k % 2 == 0:
        ap.error("k must be odd: the region is centred on qubit 25 + d.  For any other qubit "
                 "set use source_sign_mps_reader.reduced_density_matrix(psi, sites)")
    off = (a.k - 1) // 2
    import source_sign_mps_reader as R
    J0, N = geometry(a.root)
    ds = _parse_ints(a.d, off - (J0 - 1), N - J0 - off)
    times = _parse_times(a.t, [r["t"] for r in R.manifest(a.root)])
    signs = [int(s) for s in a.signs.split(",")]
    make(a.root, a.hamiltonian, a.k, ds, times, signs, a.out, a.workers, not a.big_endian)


if __name__ == "__main__":
    main()
