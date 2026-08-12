# tests/ — 337 tests

```bash
.venv/bin/python -m pytest tests/                              # all (~110 s)
.venv/bin/python -m pytest tests/qthermal/ -q                  # pipeline only
.venv/bin/python -m pytest tests/qnn/ -q                       # the network (~2 s)
.venv/bin/python -m pytest tests/qthermal/test_thermal.py      # one module
.venv/bin/python -m pytest tests/test_notebook_equivalence.py -q   # slow
```

**Run the full suite before committing anything in `qthermal/` or `qnn/`.**

---

## The philosophy: check against something independent

These tests are not smoke tests. Nearly every stage is verified against a
reference constructed by a *different* route, so that a shared misconception
cannot make both sides agree.

| Test | Independent reference |
|---|---|
| `test_diagonalize.py` | **The correctness gate.** Lowest dense eigenvalue + `ecore` must reproduce **PySCF's own CASCI energy** to 1e-8 Ha, and the second eigenvalue must match `fcisolver(nroots=2)` — at **both ncas = 6 and ncas = 8**, proving the active space is genuinely parametric |
| `test_hamiltonian.py` | A **fully manual** frozen-core construction implemented inside the test: `h_core` + HF-style core Coulomb/exchange from the frozen density, transformed to the active window, plus explicitly transformed ERIs |
| `test_encode.py` | JW-encoded eigenvectors must be exact eigenvectors of an **independently constructed PennyLane** Jordan–Wigner Hamiltonian. Fermionic signs, integral conventions, and bit ordering must all be simultaneously right for this to hold |
| `test_mps.py` | Purification MPS traced back to the density matrix built independently from JW-encoded vectors — exact untruncated, within the certified bound when capped |
| `test_thermal.py` | Brute-force expectation values; Gaussian audit against the dense g = 0 solve |
| `test_notebook_equivalence.py` | Analytic gradients vs central finite differences; four Pauli representations (dense, symbolic, CSR, fused) against each other; complex64 vs complex128 |
| `qnn/test_activations.py` | `scipy.integrate.quad` of the paper's own integral identity Eq. (A6), an independently derived closed form for `tanh`, and finite differences of the matrix function itself |
| `qnn/test_paper_equivalence.py` | Two references at once: a dense reimplementation written from the paper's equations, and the already-validated `scripts/train_spin_comparison.py` |

That discipline is why the AO-ordering bug ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I1)
was caught by comparison against fresh B3LYP rather than by a validation gate —
the gates passed on corrupt data.

---

## Layout

| File | n | Covers |
|---|---:|---|
| `qthermal/conftest.py` | — | Synthetic **B3LYP** H₂O record, built end-to-end with PySCF |
| `qthermal/test_loader.py` | 11 | Record round-trip, SQLite adapter, unit detection, `--indices` selection incl. SQLite variable-limit chunking |
| `qthermal/test_orbitals.py` | 9 | `build_mol`, overlap, MO recovery, sign canonicalization, validation gates |
| `qthermal/test_active_space.py` | 4 | Frontier window and derived dimensions |
| `qthermal/test_hamiltonian.py` | 3 | CASCI vs manual frozen-core, at (3,3) and (4,4) |
| `qthermal/test_diagonalize.py` | 29 | Both solvers, guardrails, Krylov certification, non-interacting closed form |
| `qthermal/test_thermal.py` | 16 | Weights, truncation, diagnostics, Gaussian audit |
| `qthermal/test_io_hdf5.py` | 6 | Layout, dtypes, resume safety, kT tags |
| `qthermal/test_encode.py` | 10 | JW encodings, Pauli components, tapering |
| `qthermal/test_encode_run.py` | 3 | Run file → mapping file, taper round-trip |
| `qthermal/test_mps.py` | 6 | Purification MPS |
| `qthermal/test_run.py` | 14 | End-to-end CLI on a synthetic single-molecule DB |
| `test_notebook_equivalence.py` | 12 | Original vs optimized classifier parity |
| `qnn/test_activations.py` | 102 | Every quantized activation: derivatives, divided differences, **Fréchet derivative vs finite differences of the matrix function**, self-adjointness, degenerate-eigenbasis gauge invariance |
| `qnn/test_pools.py` | 23 | Structured vs dense operators, Pauli conventions, spectral-scale init, and **the classical-reduction theorem** — a commuting pool's outputs and gradients are bit-identical on a dephased batch |
| `qnn/test_states.py` | 8 | The two data contractions vs the loops they replace |
| `qnn/test_gradients.py` | 44 | **The decisive test**: central finite differences of the composite loss across every activation, pool, depth, loss, temperature and classical activation |
| `qnn/test_paper_equivalence.py` | 4 | The single-neuron model recovered as the shallow special case, vs both an independent dense reference and `scripts/train_spin_comparison.py` |

**Why the hybrid-network gradient is finite-differenced everywhere.** Its
backward pass is a *derivation*
([`../docs/HYBRID_BACKPROP.md`](../docs/HYBRID_BACKPROP.md)), not a library
call: the chain rule terminates in the Fréchet derivative of a matrix function,
which is not `φ'(B)`. Every structurally wrong version of that rule — using
`φ'(B)`, dropping a transpose, aggregating before the nonlinearity instead of
after — still produces a plausible number and a loss curve that descends. Only
finite differences distinguish them.

**Why the fixture is B3LYP, not RHF.** RHF H₂O/def2-SVP has a 0.67 Ha
HOMO–LUMO gap and would fail `detect_units`' own physicality window
(gap ∈ [0.02, 0.6] Ha) *even in the correct unit*. QH9 is B3LYP anyway, so the
fixture matches the data it stands in for. This is a documented deviation, not
an accident (`qthermal/README.md`, Deviations 1).

---

## Adding tests

- Verify against something **independently constructed**, not against a stored
  golden value. Golden values freeze bugs.
- Parametrize over at least two active-space sizes when the code touches
  dimensions — that is what proves nothing hardcodes ncas = 8
  ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I11).
- Do not add `sys.path` manipulation. `qthermal` and `qnn` are installed editable and
  `notebook_test_utils` is a root-level module listed in `pyproject.toml`
  ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I13).
- New invariants should get a test *and* an entry in
  [`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) with a verification command.
