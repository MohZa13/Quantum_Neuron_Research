# QH9 Hamiltonian Slater Weight Dataset Builder

This builder creates compact Slater determinant weight matrices from QH9
Hamiltonian data.  It does not load QM9 directly and it does not run
Hartree-Fock or any SCF kernel.

The current primary entry point is:

```bash
python3 qh9_slater_data/build_qh9_slater.py
```

The old `build_qm9_slater_hf.py` file remains as a compatibility wrapper for
older commands, but new runs should use `build_qh9_slater.py`.

## Pipeline

For each selected QH9 molecule, the script:

1. Reads atomic numbers, positions, and a stored QH9 Hamiltonian matrix.
2. Rebuilds a neutral closed-shell PySCF molecule with the matching basis.
3. Computes the AO overlap matrix \(S\).
4. Solves the generalized eigenproblem:

   \[
   H C = S C \epsilon
   \]

5. Selects occupied Kohn-Sham orbitals by electron count.
6. Builds:

   \[
   D_{\mathrm{occ}} = S^{1/2} C_{\mathrm{occ}}
   \]

7. Builds the spin-orbital Slater matrix \(W\) with alpha AO rows first and beta
   AO rows second.
8. Saves padded `W`, `D_occ`, orbital energies, HOMO/LUMO values, gap labels,
   indices, and metadata to HDF5.

## Important Notes

QH9 Hamiltonians are DFT/Kohn-Sham Hamiltonians, commonly B3LYP/def2-SVP.  The
default basis is therefore:

```bash
--basis def2-svp
```

The Hamiltonian matrix and PySCF overlap matrix must be in the same AO ordering.
The safe default is:

```bash
--ham-ordering pyscf
```

`--ham-ordering qh9` currently raises a clear `NotImplementedError` until a
verified QH9-to-PySCF AO ordering transform is added.

## Dependencies

Core packages:

```bash
python3 -m pip install numpy scipy h5py tqdm pyscf
```

For the official QH9/QHBench loader and raw LMDB fallback:

```bash
python3 -m pip install torch torch-geometric lmdb apsw gdown
```

The QH9 loader must provide `QH9Stable` and/or `QH9Dynamic`.  If it is not
installed as a package, add the loader folder to `PYTHONPATH` or pass
`--loader-module`.

By default, the script uses `--source raw-db`, which streams
`./qh9_data/QH9Stable/raw/QH9Stable.db` directly and avoids the official
loader's heavy `Processing...` step.  Use `--source loader` only if you
specifically need the official `QH9Stable` / `QH9Dynamic` classes.

## Basic Usage

Scan molecule-size groups without diagonalizing Hamiltonians:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --max-qubits 80 \
  --scan-only
```

Write scan signatures to disk while scanning:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --max-qubits 80 \
  --scan-only \
  --scan-out qh9_slater_data/qh9_scan.jsonl
```

Scan in small sequential chunks by running the same command repeatedly:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --max-qubits 80 \
  --scan-only \
  --scan-limit 1000 \
  --scan-out qh9_slater_data/qh9_scan.jsonl
```

The first run writes dataset indices `0:1000`, the next run appends
`1000:2000`, then `2000:3000`, and so on.  After each chunk, the group summary
is computed by streaming the accumulated JSONL file.

Reuse the scan file later without rescanning or loading QH9:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --scan-only \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --max-qubits 80
```

Build the most common group under a qubit limit:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --out qh9_slater_data/qh9_slater_weights.h5 \
  --max-qubits 80 \
  --max-samples 5000
```

Build using a previously written scan file for group selection:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --out qh9_slater_data/qh9_slater_weights.h5 \
  --max-qubits 80 \
  --max-samples 5000
```

Build a specific fixed-qubit group:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --out qh9_slater_data/qh9_slater_76q.h5 \
  --target-nao 38 \
  --max-samples 5000
```

Build a fixed-qubit and fixed-electron-count group:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --out qh9_slater_data/qh9_slater_76q_46e.h5 \
  --target-nao 38 \
  --target-nelec 46 \
  --max-samples 5000
```

Build every eligible qubit-count group in one streaming pass:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --all-groups \
  --out-dir qh9_slater_data/groups \
  --out-prefix qh9_slater \
  --max-qubits 400 \
  --max-samples 0
```

This writes one appendable HDF5 file per fixed spin-orbital/qubit count:

```text
qh9_slater_data/groups/qh9_slater_256q.h5
qh9_slater_data/groups/qh9_slater_302q.h5
qh9_slater_data/groups/qh9_slater_352q.h5
...
```

Split by both qubit count and electron count:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --all-groups \
  --group-by-nelec \
  --out-dir qh9_slater_data/groups_by_nelec \
  --out-prefix qh9_slater \
  --max-qubits 400 \
  --max-samples 0
```

Run all-groups mode in resumable chunks:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --all-groups \
  --out-dir qh9_slater_data/groups \
  --out-prefix qh9_slater \
  --max-qubits 400 \
  --max-samples 0 \
  --start-index 0 \
  --stop-index 10000
```

Then continue later:

```bash
python3 qh9_slater_data/build_qh9_slater.py \
  --root ./qh9_data \
  --scan-from qh9_slater_data/qh9_scan.jsonl \
  --all-groups \
  --out-dir qh9_slater_data/groups \
  --out-prefix qh9_slater \
  --max-qubits 400 \
  --max-samples 0 \
  --start-index 10000 \
  --stop-index 20000
```

Chunks append into the same files. Existing `original_index` values are skipped,
so rerunning an overlapping chunk is safe. Binary labels are recomputed from the
current accumulated file after each chunk.

## Key Arguments

`--root`

QH9 dataset root directory.

`--out`

Output HDF5 path.  Parent directories are created automatically.

`--out-dir`

Output directory used by `--all-groups`.

`--out-prefix`

Filename prefix used by `--all-groups`.

`--dataset`

`stable` or `dynamic`.  Default: `stable`.

`--split`

Split name passed to the QH9 loader.  Defaults are `random` for stable and
`geometry` for dynamic.

`--subset`

Optional split subset: `all`, `train`, `val`, or `test`.

`--label-source`

`qh9` uses the reconstructed HOMO-LUMO gap from the stored Hamiltonian.  `qm9`
is allowed only when the QH9 data object includes a QM9 gap label.

`--source`

`raw-db` streams the raw QH9Stable SQLite database directly.  This is the
recommended low-RAM path for scanning and building from `QH9Stable.db`.
`loader` uses the official QH9 loader classes and may trigger a memory-heavy
processing step.

`--scan-out`

Streams scan metadata to a JSONL file.  Each line is one small JSON record for a
valid signature or skip reason, so the scan table does not need to live in RAM.
When the file already exists, the next scan appends from the first unscanned
dataset index.  Use this with `--scan-limit` to build the scan file in small
sequential chunks.

`--scan-from`

Streams a previously written JSONL scan file for group selection.  With
`--scan-only`, this path avoids loading the QH9 dataset at all.

`--max-samples`

Maximum number of molecules to write for a single-group build.  In
`--all-groups` mode, this is a per-group cap.  Use `--max-samples 0` for no cap.

`--all-groups`

Stream the dataset once and write one appendable HDF5 file per fixed qubit-count
group.

`--group-by-nelec`

With `--all-groups`, split files by both qubit count and electron count.

`--start-index`, `--stop-index`

Chunk bounds for resumable `--all-groups` runs.

`--min-group-samples`

When `--all-groups` is combined with `--scan-from`, skip scan groups smaller
than this count.

`--flush-every`

Flush open HDF5 files every N appended records in `--all-groups` mode.

`--compression`

Optional HDF5 compression for `--all-groups` outputs: `none`, `lzf`, or `gzip`.
Compression can save disk but slows the build.

## HDF5 Outputs

Primary datasets:

```text
W
D_occ
nelec
n_occ_spatial
y_gap_binary
gap_qh9_ev
gap_qm9_ev
homo_qh9_ev
lumo_qh9_ev
orbital_energy_hartree
qh9_index
original_index
metadata_json
```

Compatibility aliases are also written for older readers:

```text
gap_hf_ev
homo_hf_ev
lumo_hf_ev
hf_energy_hartree
mo_energy_hartree
qm9_index
```

These aliases are marked with HDF5 attributes explaining that they are not
Hartree-Fock outputs.  `hf_energy_hartree` is filled with `NaN`.

## Output Shape

For each sample, the physical Slater matrix is:

```python
W_i = W[i, :, :nelec[i]]
```

Columns after `nelec[i]` are zero padding.  `D_occ` is similarly padded in its
rightmost columns.
