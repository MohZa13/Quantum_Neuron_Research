# CLAUDE.md — source-sign MPS dataset (data folder, not a code repository)

This folder is a **dataset**: 524 full 50-qubit MPS states of an XX / XXX
spin chain after qubit 25 was rotated by `V_s = exp(−i s π Z/4)`, `s = ±1`,
at times t = 0.0 … 13.0 (step 0.1). The task it supports is to classify `s`
from the reduced state of a block R of k adjacent qubits centred at offset d
from qubit 25.

**Read `README.md` first.** It explains how to produce region states from
(k, d), with the tool, the output schema and the conventions. `DATASET.md`
is the reference for the physics, the file schema and the accuracy.

## Before doing anything else

```bash
pip install -r requirements.txt            # numpy, h5py: nothing else is needed
sha256sum -c SHA256SUMS --quiet            # data intact after transfer (macOS: shasum -a 256 -c)
python make_regions.py --selftest          # must print SELFTEST PASSED
```

If the self-test fails, stop and report it. Do not work around it: a failure
means a corrupted transfer or a broken environment, and every region state
computed afterwards would be wrong.

## How to get region states

- Prefer the tool:
  `python make_regions.py --hamiltonian XXX --k 3 --d -5:5 --t 0:6 --out regions/XXX_k3.h5`.
- In Python, use `source_sign_mps_reader`: `region_rho(psi, k, d)`,
  `all_regions(psi, k, ds=...)`, `training_set(...)`.
- Write new code only if those cannot do what is needed. Then validate it
  against `region_rho` and the `REFERENCE` table in `make_regions.py`.

## Rules that fail silently if broken

1. **Region qubits:** `R(k, d) = {25 + d − (k−1)/2, …, 25 + d + (k−1)/2}`,
   1-based, k odd. Valid d for k = 1/3/5/7: −24…25 / −23…24 / −22…23 / −21…22.
2. **ρ is little-endian by default.** Qubit q of R is bit q of the index,
   the row is the ket, and bit 0 = spin up. A `np.kron`-built operator must
   list its factors in REVERSED qubit order. The `--big-endian` flag gives
   the kron order instead. Never mix the two.
3. **Keep ρ complex and never transpose it.** At t = 0 the labels differ
   only in Im ρ (`ρ₋ = conj ρ₊`), and `ρᵀ = conj ρ` silently swaps them.
4. **Keep both labels of a (t, d) cell (`pair_id`) on the same side of any
   train/test split.** `ρ₋ = X^{⊗k} ρ₊ X^{⊗k}` exactly.
5. **Do not modify or move anything in `XX/`, `XXX/` or `manifest.csv`.**
   Write outputs to a new directory (e.g. `regions/`).
6. **Resources:** one 7-qubit region of a late XXX state needs ~1.6 GB of RAM
   and a few seconds. Scale `--workers` to the available memory.
7. **Physics guardrails:**
   - Regions with `dist > v t` (v = 2 for XX, π for XXX) carry almost no
     signal. Use the `trace_distance` column to see this per cell.
   - XXX states after t ≈ 7.6 and XX states after t ≈ 12 contain reflections
     from the chain ends.
   - A pool of only real operators cannot classify the t = 0 states (README
     §7).
