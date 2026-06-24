"""Build compact QH9 Slater determinant weight matrices from QH9 Hamiltonians.

This script consumes QH9 molecular geometries and stored DFT/Kohn-Sham
Hamiltonian matrices.  It rebuilds the matching PySCF molecule only to obtain
the AO overlap matrix S, then solves the generalized eigenproblem

    H C = S C eps

The occupied Kohn-Sham orbitals define a Slater determinant compactly through
W.  No Hartree-Fock or SCF calculation is run here.

Important: the QH9 Hamiltonian and the PySCF overlap matrix must use the same
AO ordering and basis.  The default basis is def2-svp because QH9 Hamiltonians
are provided in that basis.
"""

import argparse
import importlib
import inspect
import json
import os
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.linalg as la


def ensure_usable_tempdir():
    """Provide a local temp directory before PySCF imports tempfile."""
    candidates = [
        os.environ.get("PYSCF_TMPDIR"),
        os.environ.get("TMPDIR"),
        "/private/tmp",
        os.path.join(os.path.dirname(__file__), ".tmp"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            test_path = os.path.join(candidate, ".qh9_tmp_test")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(test_path)
        except OSError:
            continue

        os.environ.setdefault("TMPDIR", candidate)
        os.environ.setdefault("TEMP", candidate)
        os.environ.setdefault("TMP", candidate)
        os.environ.setdefault("PYSCF_TMPDIR", candidate)
        return candidate

    raise RuntimeError("No usable temporary directory found for PySCF.")


ensure_usable_tempdir()
from pyscf import gto
from tqdm import tqdm


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 1.8897259886

PYSCF_DEF2SVP_CONVENTION = SimpleNamespace(
    atom_to_orbitals_map={1: "ssp", 6: "sssppd", 7: "sssppd", 8: "sssppd", 9: "sssppd"},
    orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
    orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
    orbital_order_map={
        1: [0, 1, 2],
        6: [0, 1, 2, 3, 4, 5],
        7: [0, 1, 2, 3, 4, 5],
        8: [0, 1, 2, 3, 4, 5],
        9: [0, 1, 2, 3, 4, 5],
    },
)


class IndexedDataset:
    """Small index-view wrapper for dataset subsets."""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        return self.base_dataset[int(self.indices[idx])]

    def source_index(self, idx):
        return int(self.indices[idx])


class RawQH9SQLiteDataset:
    """Streaming reader for raw QH9 SQLite files, avoiding PyG processing."""

    def __init__(self, db_path, dataset="stable"):
        self.db_path = db_path
        self.dataset = dataset
        self._length = None

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Raw QH9 SQLite DB not found: {self.db_path}")

    def __len__(self):
        if self._length is None:
            with sqlite3.connect(self.db_path) as connection:
                self._length = int(connection.execute("select count(*) from data").fetchone()[0])
        return self._length

    def __getitem__(self, idx):
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute("select * from data where id = ?", (int(idx),)).fetchone()

        if row is None:
            raise IndexError(f"No QH9 raw DB row with id={idx}")

        if self.dataset == "stable":
            _, num_nodes, atoms_blob, pos_blob, ham_blob = row
            pos_scale = 1.0
        else:
            raise NotImplementedError("Raw SQLite streaming is currently implemented for QH9Stable.db")

        atoms = np.frombuffer(atoms_blob, dtype=np.int32).astype(np.int64)
        pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(num_nodes), 3) / pos_scale
        ham_flat = np.frombuffer(ham_blob, dtype=np.float64)
        num_orbitals = int(sum(qh9_def2svp_ao_count(atom) for atom in atoms))
        ham = ham_flat.reshape(num_orbitals, num_orbitals)
        ham = transform_qh9_hamiltonian_to_pyscf_order(ham, atoms)

        return SimpleNamespace(atoms=atoms, pos=pos, Ham=ham)

    def get_geometry(self, idx):
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "select N, Z, pos from data where id = ?",
                (int(idx),),
            ).fetchone()

        if row is None:
            raise IndexError(f"No QH9 raw DB row with id={idx}")

        num_nodes, atoms_blob, pos_blob = row
        atoms = np.frombuffer(atoms_blob, dtype=np.int32).astype(np.int64)
        pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(num_nodes), 3)
        return SimpleNamespace(atoms=atoms, pos=pos)


def to_numpy(value, dtype=None):
    """Convert common tensor/array values to NumPy without depending on torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)

    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def qh9_def2svp_ao_count(atomic_number):
    """AO count used by the QHBench def2-SVP Hamiltonian matrices."""
    return 5 if int(atomic_number) <= 2 else 14


def transform_qh9_hamiltonian_to_pyscf_order(matrix, atoms):
    """Transform raw QH9 def2-SVP Hamiltonians to PySCF AO ordering."""
    orbitals = ""
    orbitals_order = []

    for atomic_number in atoms:
        atomic_number = int(atomic_number)
        offset = len(orbitals_order)
        orbitals += PYSCF_DEF2SVP_CONVENTION.atom_to_orbitals_map[atomic_number]
        orbitals_order += [
            idx + offset for idx in PYSCF_DEF2SVP_CONVENTION.orbital_order_map[atomic_number]
        ]

    transform_indices = []
    transform_signs = []
    for orbital in orbitals:
        offset = sum(len(indices) for indices in transform_indices)
        transform_indices.append(
            np.array(PYSCF_DEF2SVP_CONVENTION.orbital_idx_map[orbital]) + offset
        )
        transform_signs.append(np.array(PYSCF_DEF2SVP_CONVENTION.orbital_sign_map[orbital]))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    transformed = matrix[transform_indices, :]
    transformed = transformed[:, transform_indices]
    transformed = transformed * transform_signs[:, None]
    transformed = transformed * transform_signs[None, :]
    return transformed


def qhbench_raw_lmdb_path(dataset):
    """Return an official QHBench LMDB path if this dataset exposes one."""
    processed_dir = getattr(dataset, "processed_dir", None)
    if processed_dir is None:
        return None

    for lmdb_name in ["QH9Stable.lmdb", "QH9Dynamic.lmdb"]:
        candidate = os.path.join(processed_dir, lmdb_name)
        if os.path.isdir(candidate):
            return candidate

    return None


def load_qhbench_raw_sample(dataset, source_idx):
    """
    Load the raw full Hamiltonian from an official QHBench LMDB, if available.

    Upstream QHBench Data objects often expose block Hamiltonian tensors after
    their model-order transform.  The LMDB payload keeps atoms, positions, and
    the full Hamiltonian matrix.  This fallback lets the script use the official
    loader while still diagonalizing a full matrix.
    """
    lmdb_path = qhbench_raw_lmdb_path(dataset)
    if lmdb_path is None:
        return None

    try:
        lmdb = importlib.import_module("lmdb")
    except ImportError:
        return None

    db_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    try:
        with db_env.begin() as txn:
            payload = txn.get(int(source_idx).to_bytes(length=4, byteorder="big"))
        if payload is None:
            return None

        data_dict = pickle.loads(payload)
        num_nodes = int(data_dict["num_nodes"])
        atoms = np.frombuffer(data_dict["atoms"], dtype=np.int32).astype(np.int64)
        pos = np.frombuffer(data_dict["pos"], dtype=np.float64).reshape(num_nodes, 3)
        ham_flat = np.frombuffer(data_dict["Ham"], dtype=np.float64)

        if "QH9Dynamic" in os.path.basename(lmdb_path):
            pos = pos / BOHR_TO_ANGSTROM

        num_orbitals = int(sum(qh9_def2svp_ao_count(atom) for atom in atoms))
        ham = ham_flat.reshape(num_orbitals, num_orbitals)

        return SimpleNamespace(atoms=atoms, pos=pos, Ham=ham)
    finally:
        db_env.close()


def dataset_base_and_source_index(dataset, idx):
    if isinstance(dataset, IndexedDataset):
        return dataset.base_dataset, dataset.source_index(idx)
    return dataset, int(idx)


def get_dataset_sample(dataset, idx):
    """Return a sample, preferring raw full QHBench Hamiltonians when present."""
    base_dataset, source_idx = dataset_base_and_source_index(dataset, idx)
    raw_sample = load_qhbench_raw_sample(base_dataset, source_idx)
    if raw_sample is not None:
        return raw_sample
    return dataset[idx]


def get_dataset_geometry_sample(dataset, idx):
    """Return only atoms/positions when a dataset can provide lightweight geometry."""
    base_dataset, source_idx = dataset_base_and_source_index(dataset, idx)
    if hasattr(base_dataset, "get_geometry"):
        return base_dataset.get_geometry(source_idx)
    return get_dataset_sample(dataset, idx)


def get_field(data, names, required=True):
    """Read the first available field from a PyG-style data object."""
    for name in names:
        if hasattr(data, name):
            value = getattr(data, name)
            if value is not None:
                return value, name

        try:
            value = data[name]
        except (AttributeError, KeyError, TypeError):
            continue
        if value is not None:
            return value, name

    if required:
        joined = ", ".join(names)
        raise ValueError(f"Missing required QH9 field; tried: {joined}")
    return None, None


def has_any_field(data, names):
    for name in names:
        try:
            value, _ = get_field(data, [name], required=False)
        except Exception:
            value = None
        if value is not None:
            return True
    return False


def extract_qh9_geometry(data):
    """Extract atomic numbers and Angstrom positions from a QH9 data object."""
    atoms_value, atoms_name = get_field(data, ["atoms", "z", "atomic_numbers"])
    pos_value, _ = get_field(data, ["pos", "positions", "positions_angstrom"])

    z = to_numpy(atoms_value, dtype=np.int64).reshape(-1)
    pos = to_numpy(pos_value, dtype=float)

    if pos.ndim == 1:
        if pos.size % 3 != 0:
            raise ValueError("Position array is flat but its length is not divisible by 3")
        pos = pos.reshape(-1, 3)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Position array must have shape (n_atoms, 3), got {pos.shape}")

    if z.shape[0] != pos.shape[0]:
        raise ValueError(
            f"Atomic numbers from {atoms_name} have length {z.shape[0]}, "
            f"but positions contain {pos.shape[0]} atoms"
        )

    unsupported = sorted(set(int(atomic_number) for atomic_number in z) - set(Z_TO_SYMBOL))
    if unsupported:
        raise ValueError(f"Unsupported atomic numbers for this script: {unsupported}")

    return z, pos


def extract_full_hamiltonian(data):
    """Extract a full Hamiltonian matrix from common QH9 field names."""
    ham_value, ham_name = get_field(
        data,
        ["Ham", "ham", "hamiltonian", "Hamiltonian", "H"],
        required=False,
    )

    if ham_value is None:
        block_fields = [
            "diagonal_hamiltonian",
            "non_diagonal_hamiltonian",
            "edge_index_full",
        ]
        if has_any_field(data, block_fields):
            raise ValueError(
                "This QH9 data object exposes block Hamiltonian fields instead "
                "of a full PySCF-ordered Ham/ham/hamiltonian matrix. Upstream "
                "QHBench block fields are commonly in QH9 AO ordering; add a "
                "verified qh9-to-pyscf AO conversion before diagonalizing them."
            )
        raise ValueError("No Hamiltonian field found; expected Ham, ham, or hamiltonian")

    ham = np.squeeze(to_numpy(ham_value, dtype=float))

    if ham.ndim == 1:
        side = int(round(np.sqrt(ham.size)))
        if side * side != ham.size:
            raise ValueError(f"Flat Hamiltonian from {ham_name} has non-square length {ham.size}")
        ham = ham.reshape(side, side)

    if ham.ndim != 2:
        raise ValueError(f"Hamiltonian from {ham_name} must be 2D, got shape {ham.shape}")

    return ham


def extract_qh9_fields(data):
    """Return atomic numbers, Angstrom positions, and full Hamiltonian matrix."""
    z, pos = extract_qh9_geometry(data)
    ham = extract_full_hamiltonian(data)
    return z, pos, ham


def extract_qm9_gap_ev(data):
    """Return an optional QM9 gap label from a QH9 data object, if present."""
    for name in ["gap_qm9_ev", "qm9_gap_ev"]:
        value, _ = get_field(data, [name], required=False)
        if value is not None:
            flat = to_numpy(value, dtype=float).reshape(-1)
            if flat.size:
                return float(flat[0])

    y_value, _ = get_field(data, ["y"], required=False)
    if y_value is None:
        return None

    y = to_numpy(y_value, dtype=float).reshape(-1)
    if y.shape[0] <= 4:
        return None
    return float(y[4])


def build_pyscf_mol_from_arrays(z, pos, basis):
    """Build a neutral closed-shell PySCF molecule from arrays."""
    nelec = int(np.sum(z))
    if nelec % 2 != 0:
        return None

    atoms = []
    for atomic_number, xyz in zip(z, pos):
        symbol = Z_TO_SYMBOL[int(atomic_number)]
        atoms.append((symbol, tuple(float(x) for x in xyz)))

    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    return mol


def molecule_signature(data, basis):
    """
    Signature used for pruning.

    Only geometry and basis construction are used; no SCF is run.
    Returns (nao, n_spin_orbitals, nelec).
    """
    z, pos = extract_qh9_geometry(data)
    mol = build_pyscf_mol_from_arrays(z, pos, basis)
    if mol is None:
        return None

    nao = mol.nao_nr()
    nelec = mol.nelectron
    n_spin_orbitals = 2 * nao

    return nao, n_spin_orbitals, nelec


def symmetric_sqrt(matrix):
    """Compute the symmetric square root of a real symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)

    tol_neg = -1e-8
    min_eig = float(np.min(eigvals))
    if min_eig < tol_neg:
        raise ValueError(f"Overlap matrix has negative eigenvalues: {min_eig}")

    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T


def validate_hamiltonian_and_overlap(ham, overlap, nao):
    """Validate matrix dimensions and basic Hermiticity before eigensolving."""
    if ham.ndim != 2:
        raise ValueError(f"Hamiltonian must be a 2D matrix, got shape {ham.shape}")
    if ham.shape[0] != ham.shape[1]:
        raise ValueError(f"Hamiltonian must be square, got shape {ham.shape}")
    if ham.shape != (nao, nao):
        raise ValueError(
            f"Hamiltonian shape {ham.shape} does not match PySCF nao {(nao, nao)}"
        )
    if overlap.shape != (nao, nao):
        raise ValueError(
            f"Overlap shape {overlap.shape} does not match PySCF nao {(nao, nao)}"
        )

    asym = np.linalg.norm(ham - ham.T)
    scale = max(float(np.linalg.norm(ham)), 1.0)
    if asym / scale > 1e-8:
        raise ValueError(f"Hamiltonian is not symmetric enough for eigh: rel_err={asym / scale:.3e}")


def diagonalize_hamiltonian_and_build_W(data, basis, ordering="pyscf", label_source="qh9"):
    """
    Build the occupied spin-orbital Slater matrix W from a QH9 Hamiltonian.

    Steps:
    1. Extract QH9 atoms, Angstrom positions, and a full Hamiltonian matrix.
    2. Build the matching PySCF molecule and AO overlap S.
    3. Solve H C = S C eps.
    4. Select occupied closed-shell Kohn-Sham orbitals by electron count.
    5. Lowdin-transform to an orthonormal AO basis: D_occ = S^(1/2) C_occ.
    6. Build W with alpha AO rows first and beta AO rows second.
    """
    if ordering == "qh9":
        raise NotImplementedError(
            "--ham-ordering qh9 was requested, but no verified QH9-to-PySCF "
            "AO ordering transformation is implemented. The Hamiltonian and "
            "PySCF overlap matrix must be in the same AO ordering."
        )
    if ordering != "pyscf":
        raise ValueError(f"Unsupported Hamiltonian ordering: {ordering}")

    # TODO: insert a verified AO-order conversion/permutation here if QH9-order
    # Hamiltonians need to be transformed to PySCF AO order in the future.
    z, pos, ham = extract_qh9_fields(data)
    mol = build_pyscf_mol_from_arrays(z, pos, basis)
    if mol is None:
        raise ValueError("Molecule has an odd electron count; only closed-shell systems are supported")

    nao = mol.nao_nr()
    nelec = int(mol.nelectron)
    if nelec % 2 != 0:
        raise ValueError(f"Electron count must be even for closed-shell QH9 workflow, got {nelec}")

    overlap = mol.intor("int1e_ovlp")
    validate_hamiltonian_and_overlap(ham, overlap, nao)

    ham = 0.5 * (ham + ham.T)
    eps, coeff = la.eigh(ham, overlap)
    order = np.argsort(eps)
    eps = np.asarray(eps[order], dtype=float)
    coeff = np.asarray(coeff[:, order], dtype=float)

    n_occ_spatial = nelec // 2
    if n_occ_spatial <= 0:
        raise ValueError("No occupied orbitals were selected")
    if n_occ_spatial >= eps.shape[0]:
        raise ValueError(
            f"Need at least one virtual orbital for a LUMO, got n_occ={n_occ_spatial} "
            f"and n_orbitals={eps.shape[0]}"
        )

    coeff_occ = coeff[:, :n_occ_spatial]

    s_sqrt = symmetric_sqrt(overlap)
    d_occ = s_sqrt @ coeff_occ

    err = np.linalg.norm(d_occ.T @ d_occ - np.eye(n_occ_spatial))
    if err > 1e-7:
        raise ValueError(f"D_occ orthonormality failure: {err}")

    n_spatial = d_occ.shape[0]
    w = np.zeros((2 * n_spatial, nelec), dtype=np.complex128)

    for p in range(n_occ_spatial):
        w[0:n_spatial, 2 * p] = d_occ[:, p]
        w[n_spatial : 2 * n_spatial, 2 * p + 1] = d_occ[:, p]

    w_err = np.linalg.norm(w.conj().T @ w - np.eye(nelec))
    if w_err > 1e-7:
        raise ValueError(f"W orthonormality failure: {w_err}")

    homo_qh9_ev = float(eps[n_occ_spatial - 1] * HARTREE_TO_EV)
    lumo_qh9_ev = float(eps[n_occ_spatial] * HARTREE_TO_EV)
    gap_qh9_ev = float(lumo_qh9_ev - homo_qh9_ev)

    gap_qm9_ev = extract_qm9_gap_ev(data)
    if label_source == "qm9" and gap_qm9_ev is None:
        raise ValueError(
            "--label-source qm9 was requested, but this QH9 data object has no "
            "available QM9 gap label"
        )

    return {
        "W": w,
        "D_occ": d_occ.astype(np.complex128),
        "orbital_energy_hartree": eps,
        "homo_qh9_ev": homo_qh9_ev,
        "lumo_qh9_ev": lumo_qh9_ev,
        "gap_qh9_ev": gap_qh9_ev,
        "gap_qm9_ev": np.nan if gap_qm9_ev is None else float(gap_qm9_ev),
        "hf_energy_hartree": np.nan,
        "nao": int(n_spatial),
        "n_spin_orbitals": int(2 * n_spatial),
        "nelec": int(nelec),
        "n_occ_spatial": int(n_occ_spatial),
        "atomic_numbers": z.tolist(),
        "positions_angstrom": pos.tolist(),
    }


def import_qh9_classes(loader_module=None):
    """Find QH9Stable/QH9Dynamic classes from local or installed modules."""
    module_names = []
    if loader_module:
        module_names.append(loader_module)

    module_names.extend(
        [
            "datasets",
            "qh9.datasets",
            "QH9.datasets",
            "torch_geometric.datasets",
            "torch_geometric.datasets.qh9",
        ]
    )

    import_errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue

        stable_cls = getattr(module, "QH9Stable", None)
        dynamic_cls = getattr(module, "QH9Dynamic", None)
        if stable_cls is not None or dynamic_cls is not None:
            return stable_cls, dynamic_cls, module_name

    details = "\n  ".join(import_errors[-5:])
    raise ImportError(
        "Could not import QH9Stable or QH9Dynamic. Add the QH9 benchmark "
        "loader to PYTHONPATH, install a package that provides it, or pass "
        "--loader-module. Recent attempts:\n  "
        f"{details}"
    )


def instantiate_dataset(dataset_cls, kwargs):
    """Instantiate a dataset while passing only constructor-supported kwargs."""
    try:
        signature = inspect.signature(dataset_cls)
    except (TypeError, ValueError):
        return dataset_cls(**kwargs)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        call_kwargs = kwargs
    else:
        call_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters and value is not None
        }

    return dataset_cls(**call_kwargs)


def apply_subset(dataset, subset):
    """Return the requested split subset without materializing all samples."""
    if subset == "all":
        return dataset

    mask_name = {
        "train": "train_mask",
        "val": "val_mask",
        "test": "test_mask",
    }[subset]

    if not hasattr(dataset, mask_name):
        raise ValueError(f"Dataset has no {mask_name}; cannot select --subset {subset}")

    indices = to_numpy(getattr(dataset, mask_name), dtype=np.int64).reshape(-1)
    return IndexedDataset(dataset, indices)


def load_qh9_dataset(args):
    """Load a QH9 stable or dynamic dataset using an available local loader."""
    if args.source == "raw-db":
        if args.dataset != "stable":
            raise NotImplementedError("--source raw-db currently supports --dataset stable only")
        if args.subset != "all":
            raise ValueError("--source raw-db streams the raw DB and currently supports --subset all only")

        split = args.split or "raw"
        db_path = os.path.join(args.root, "QH9Stable", "raw", "QH9Stable.db")
        dataset = RawQH9SQLiteDataset(db_path=db_path, dataset=args.dataset)
        return dataset, "raw SQLite QH9Stable.db", split

    stable_cls, dynamic_cls, module_name = import_qh9_classes(args.loader_module)

    if args.dataset == "stable":
        if stable_cls is None:
            raise ImportError(f"Module {module_name} does not provide QH9Stable")
        split = args.split or "random"
        dataset = instantiate_dataset(
            stable_cls,
            {
                "root": args.root,
                "split": split,
            },
        )
    elif args.dataset == "dynamic":
        if dynamic_cls is None:
            raise ImportError(f"Module {module_name} does not provide QH9Dynamic")
        split = args.split or "geometry"
        dataset = instantiate_dataset(
            dynamic_cls,
            {
                "root": args.root,
                "split": split,
                "version": args.version,
            },
        )
    else:
        raise ValueError(f"Unknown QH9 dataset: {args.dataset}")

    dataset = apply_subset(dataset, args.subset)
    return dataset, module_name, split


def skip_reason(exc):
    message = str(exc).splitlines()[0] if str(exc) else ""
    if len(message) > 180:
        message = message[:177] + "..."
    return f"{type(exc).__name__}: {message}"


def print_skip_summary(skipped):
    if not skipped:
        return
    print("Skipped molecules by reason:")
    for reason, count in skipped.most_common(10):
        print(f"  {count:6d}  {reason}")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_scan_jsonl(handle, record):
    if handle is None:
        return
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def print_group_summary(group_counts, nelec_dist):
    print("Top groups (rank | nao | spin_orbitals | count | most common electron counts):")
    for rank, (group, count) in enumerate(group_counts.most_common(20), start=1):
        nao, n_spin_orbitals = group
        common_nelec = nelec_dist[group].most_common(5)
        common_str = ", ".join(f"{e}:{c}" for e, c in common_nelec)
        print(f"{rank:2d} | {nao:3d} | {n_spin_orbitals:3d} | {count:6d} | {common_str}")


def next_scan_dataset_index(scan_path):
    """Return the next dataset index for appending a sequential scan chunk."""
    if scan_path is None or not os.path.exists(scan_path) or os.path.getsize(scan_path) == 0:
        return 0

    next_idx = 0
    scanned_records_without_dataset_idx = 0

    with open(scan_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            if record.get("record_type") not in {"signature", "skip"}:
                continue

            if "dataset_idx" in record:
                next_idx = max(next_idx, int(record["dataset_idx"]) + 1)
            else:
                scanned_records_without_dataset_idx += 1

    if next_idx == 0 and scanned_records_without_dataset_idx:
        next_idx = scanned_records_without_dataset_idx

    return next_idx


def choose_group_from_scan_file(scan_path, max_qubits, basis=None, scan_limit=None):
    """Choose a group by streaming a previously written scan JSONL file."""
    group_counts = Counter()
    nelec_dist = defaultdict(Counter)
    skipped = Counter()
    n_seen = 0

    with open(scan_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            record_type = record.get("record_type")

            if record_type in {"metadata", "scan_chunk"}:
                scan_basis = record.get("basis")
                if basis is not None and scan_basis is not None and scan_basis != basis:
                    print(
                        "Warning: scan file basis "
                        f"{scan_basis!r} does not match current --basis {basis!r}."
                    )
                continue

            if scan_limit is not None and n_seen >= scan_limit:
                break
            n_seen += 1

            if record_type == "skip":
                skipped[str(record.get("reason", "unknown"))] += 1
                continue

            if record_type != "signature":
                continue

            nao = int(record["nao"])
            n_spin_orbitals = int(record["n_spin_orbitals"])
            nelec = int(record["nelec"])

            if n_spin_orbitals <= max_qubits:
                key = (nao, n_spin_orbitals)
                group_counts[key] += 1
                nelec_dist[key][nelec] += 1

    if not group_counts:
        print_skip_summary(skipped)
        raise RuntimeError(
            f"No molecule group found in {scan_path} with n_spin_orbitals <= {max_qubits}."
        )

    print(f"Loaded scan signatures from {scan_path}")
    print_group_summary(group_counts, nelec_dist)
    print_skip_summary(skipped)
    return group_counts.most_common(1)[0][0]


def choose_group(dataset, basis, max_qubits, scan_limit=None, scan_out=None):
    """
    Choose the most common same-size group satisfying n_spin_orbitals <= max_qubits.

    Group key is (nao, n_spin_orbitals).  Electron counts are reported but are
    allowed to vary unless --target-nelec is later supplied.
    """
    group_counts = Counter()
    nelec_dist = defaultdict(Counter)
    skipped = Counter()
    scan_path = None
    scan_mode = "w"

    if scan_out is None:
        scan_start = 0
        scan_stop = len(dataset) if scan_limit is None else min(scan_limit, len(dataset))
    else:
        scan_path = scan_out
        ensure_parent_dir(scan_path)
        scan_start = next_scan_dataset_index(scan_path)
        scan_stop = len(dataset) if scan_limit is None else min(scan_start + scan_limit, len(dataset))
        scan_mode = "a" if os.path.exists(scan_path) and os.path.getsize(scan_path) > 0 else "w"

        if scan_start >= len(dataset):
            print(f"Scan file already covers {scan_start} dataset entries: {scan_path}")
            return choose_group_from_scan_file(
                scan_path=scan_path,
                max_qubits=max_qubits,
                basis=basis,
                scan_limit=None,
            )

    n_scan = max(scan_stop - scan_start, 0)

    scan_handle = None
    if scan_path is not None:
        scan_handle = open(scan_path, scan_mode, encoding="utf-8")

    try:
        write_scan_jsonl(
            scan_handle,
            {
                "record_type": "scan_chunk",
                "basis": basis,
                "max_qubits": int(max_qubits),
                "scan_limit": None if scan_limit is None else int(scan_limit),
                "scan_start": int(scan_start),
                "scan_stop": int(scan_stop),
                "n_scan": int(n_scan),
            },
        )

        for idx in tqdm(range(scan_start, scan_stop), desc="Scanning QH9 molecule sizes"):
            original_index = source_index_for(dataset, idx)
            try:
                sig = molecule_signature(get_dataset_geometry_sample(dataset, idx), basis)
            except Exception as exc:
                reason = skip_reason(exc)
                skipped[reason] += 1
                write_scan_jsonl(
                    scan_handle,
                    {
                        "record_type": "skip",
                        "dataset_idx": int(idx),
                        "index": int(original_index),
                        "reason": reason,
                    },
                )
                continue

            if sig is None:
                reason = "odd electron count"
                skipped[reason] += 1
                write_scan_jsonl(
                    scan_handle,
                    {
                        "record_type": "skip",
                        "dataset_idx": int(idx),
                        "index": int(original_index),
                        "reason": reason,
                    },
                )
                continue

            nao, n_spin_orbitals, nelec = sig
            write_scan_jsonl(
                scan_handle,
                {
                    "record_type": "signature",
                    "dataset_idx": int(idx),
                    "index": int(original_index),
                    "nao": int(nao),
                    "n_spin_orbitals": int(n_spin_orbitals),
                    "nelec": int(nelec),
                },
            )

            if n_spin_orbitals <= max_qubits:
                key = (nao, n_spin_orbitals)
                group_counts[key] += 1
                nelec_dist[key][nelec] += 1
    finally:
        if scan_handle is not None:
            scan_handle.close()

    if scan_path is not None:
        print(
            f"Wrote scan chunk {scan_start}:{scan_stop} "
            f"({n_scan} entries) to {scan_path}"
        )
        return choose_group_from_scan_file(
            scan_path=scan_path,
            max_qubits=max_qubits,
            basis=basis,
            scan_limit=None,
        )

    if not group_counts:
        print_skip_summary(skipped)
        raise RuntimeError(
            f"No molecule group found with n_spin_orbitals <= {max_qubits}. "
            "Increase --max-qubits or check basis/dataset compatibility."
        )

    print_group_summary(group_counts, nelec_dist)
    print_skip_summary(skipped)
    return group_counts.most_common(1)[0][0]


def source_index_for(dataset, idx):
    if hasattr(dataset, "source_index"):
        return int(dataset.source_index(idx))
    return int(idx)


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="./qh9_data")
    parser.add_argument("--out", type=str, default="qh9_slater_data/qh9_slater_weights.h5")
    parser.add_argument(
        "--basis",
        type=str,
        default="def2-svp",
        help="Basis used to rebuild PySCF overlap matrix; must match QH9 Hamiltonian basis.",
    )

    parser.add_argument(
        "--source",
        choices=["raw-db", "loader"],
        default="raw-db",
        help="raw-db streams QH9Stable.db directly; loader uses QH9Stable/QH9Dynamic classes.",
    )
    parser.add_argument("--dataset", choices=["stable", "dynamic"], default="stable")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Stable: random or size_ood. Dynamic: geometry or mol. Defaults by dataset.",
    )
    parser.add_argument("--subset", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--version", choices=["100k", "300k"], default="300k")
    parser.add_argument(
        "--loader-module",
        type=str,
        default=None,
        help="Optional module path containing QH9Stable/QH9Dynamic classes.",
    )

    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--target-nao", type=int, default=None)
    parser.add_argument("--target-nelec", type=int, default=None)

    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument(
        "--scan-out",
        type=str,
        default=None,
        help="Append per-molecule scan signatures/skips to a JSONL file.",
    )
    parser.add_argument(
        "--scan-from",
        type=str,
        default=None,
        help="Reuse a JSONL scan file for group selection instead of rescanning.",
    )

    parser.add_argument(
        "--label-source",
        choices=["qh9", "qm9"],
        default="qh9",
        help="Use reconstructed QH9 gap or an available QM9 gap field to create the binary label.",
    )
    parser.add_argument(
        "--ham-ordering",
        choices=["pyscf", "qh9"],
        default="pyscf",
        help="AO ordering of the full Hamiltonian matrix. Only pyscf is implemented safely.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan QH9 for group counts and exit without building Slater determinants.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.ham_ordering == "qh9":
        raise NotImplementedError(
            "--ham-ordering qh9 requires a verified AO-order transformation before "
            "diagonalization. Use --ham-ordering pyscf only when Ham already matches "
            "PySCF def2-SVP AO ordering."
        )

    if args.scan_only and args.scan_from is not None:
        chosen = choose_group_from_scan_file(
            scan_path=args.scan_from,
            max_qubits=args.max_qubits,
            basis=args.basis,
            scan_limit=args.scan_limit,
        )
        target_nao, target_n_spin_orbitals = chosen
        print("Auto-selected group (scan-only from file):")
        print(f"  nao               = {target_nao}")
        print(f"  n_spin_orbitals   = {target_n_spin_orbitals}")
        print("Exiting per --scan-only.")
        sys.exit(0)

    dataset, loader_module, resolved_split = load_qh9_dataset(args)

    print("Loaded QH9 dataset:")
    print(f"  loader module      = {loader_module}")
    print(f"  dataset            = {args.dataset}")
    print(f"  split              = {resolved_split}")
    print(f"  subset             = {args.subset}")
    print(f"  samples            = {len(dataset)}")
    print(f"  basis              = {args.basis}")
    print(f"  ham_ordering       = {args.ham_ordering}")

    if args.scan_only:
        chosen = choose_group(
            dataset=dataset,
            basis=args.basis,
            max_qubits=args.max_qubits,
            scan_limit=args.scan_limit,
            scan_out=args.scan_out,
        )
        target_nao, target_n_spin_orbitals = chosen
        print("Auto-selected group (scan-only):")
        print(f"  nao               = {target_nao}")
        print(f"  n_spin_orbitals   = {target_n_spin_orbitals}")
        print("Exiting per --scan-only.")
        sys.exit(0)

    if args.target_nao is None:
        if args.scan_from is not None:
            chosen_key = choose_group_from_scan_file(
                scan_path=args.scan_from,
                max_qubits=args.max_qubits,
                basis=args.basis,
                scan_limit=args.scan_limit,
            )
        else:
            chosen_key = choose_group(
                dataset=dataset,
                basis=args.basis,
                max_qubits=args.max_qubits,
                scan_limit=args.scan_limit,
                scan_out=args.scan_out,
            )

        target_nao, target_n_spin_orbitals = chosen_key
        target_nelec = None

        print("Auto-selected group:")
        print(f"  nao               = {target_nao}")
        print(f"  n_spin_orbitals   = {target_n_spin_orbitals}")
        print("  nelec             = variable")

    else:
        target_nao = args.target_nao
        target_nelec = args.target_nelec
        target_n_spin_orbitals = 2 * target_nao

        print("Using requested group:")
        print(f"  nao               = {target_nao}")
        print(f"  n_spin_orbitals   = {target_n_spin_orbitals}")
        if target_nelec is None:
            print("  nelec             = variable")
        else:
            print(f"  nelec             = {target_nelec}")

    records = []
    skipped = Counter()

    for idx in tqdm(range(len(dataset)), desc="Diagonalizing QH9 Hamiltonians and building W"):
        data = get_dataset_sample(dataset, idx)

        try:
            sig = molecule_signature(data, args.basis)
        except Exception as exc:
            skipped[skip_reason(exc)] += 1
            continue

        if sig is None:
            skipped["odd electron count"] += 1
            continue

        nao, _, nelec = sig

        if nao != target_nao:
            continue

        if target_nelec is not None and nelec != target_nelec:
            continue

        try:
            result = diagonalize_hamiltonian_and_build_W(
                data,
                args.basis,
                ordering=args.ham_ordering,
                label_source=args.label_source,
            )
        except Exception as exc:
            skipped[skip_reason(exc)] += 1
            continue

        original_index = source_index_for(dataset, idx)
        result["original_index"] = original_index
        result["qh9_index"] = original_index
        records.append(result)

        if len(records) >= args.max_samples:
            break

    print_skip_summary(skipped)

    if len(records) == 0:
        raise RuntimeError("No records survived filtering and QH9 Hamiltonian diagonalization.")

    if args.label_source == "qh9":
        gaps = np.array([float(r["gap_qh9_ev"]) for r in records], dtype=float)
    elif args.label_source == "qm9":
        gaps = np.array([float(r["gap_qm9_ev"]) for r in records], dtype=float)
        if not np.all(np.isfinite(gaps)):
            raise RuntimeError("--label-source qm9 requested, but at least one QM9 gap is unavailable")
    else:
        raise ValueError(f"Unknown label source: {args.label_source}")

    threshold = float(np.median(gaps))
    labels = (gaps >= threshold).astype(np.int64)

    n_samples = len(records)
    n_qubits = int(target_n_spin_orbitals)
    nao_val = int(target_nao)

    w_all = np.zeros((n_samples, n_qubits, n_qubits), dtype=np.complex128)
    d_all = np.zeros((n_samples, nao_val, nao_val), dtype=np.complex128)
    orbital_energy_all = np.full((n_samples, nao_val), np.nan, dtype=float)

    nelec_arr = np.zeros((n_samples,), dtype=np.int64)
    n_occ_spatial_arr = np.zeros((n_samples,), dtype=np.int64)

    gap_qh9_list = []
    gap_qm9_list = []
    homo_qh9_list = []
    lumo_qh9_list = []
    qh9_index_list = []

    for i, record in enumerate(records):
        w = record["W"]
        d_occ = record["D_occ"]
        orbital_energy = record["orbital_energy_hartree"]
        nelec_i = int(record["nelec"])
        n_occ_i = int(record["n_occ_spatial"])

        w_all[i, :, :nelec_i] = w
        d_all[i, :, :n_occ_i] = d_occ
        orbital_energy_all[i, : orbital_energy.shape[0]] = orbital_energy

        nelec_arr[i] = nelec_i
        n_occ_spatial_arr[i] = n_occ_i

        gap_qh9_list.append(float(record["gap_qh9_ev"]))
        gap_qm9_list.append(float(record["gap_qm9_ev"]))
        homo_qh9_list.append(float(record["homo_qh9_ev"]))
        lumo_qh9_list.append(float(record["lumo_qh9_ev"]))
        qh9_index_list.append(int(record["qh9_index"]))

    gap_qh9_ev = np.array(gap_qh9_list, dtype=float)
    gap_qm9_ev = np.array(gap_qm9_list, dtype=float)
    homo_qh9_ev = np.array(homo_qh9_list, dtype=float)
    lumo_qh9_ev = np.array(lumo_qh9_list, dtype=float)
    hf_energy = np.full((n_samples,), np.nan, dtype=float)
    qh9_indices = np.array(qh9_index_list, dtype=np.int64)

    metadata_json = [
        json.dumps(
            {
                "original_index": record["original_index"],
                "qh9_index": record["qh9_index"],
                "atomic_numbers": record["atomic_numbers"],
                "positions_angstrom": record["positions_angstrom"],
                "nao": record["nao"],
                "n_spin_orbitals": record["n_spin_orbitals"],
                "nelec": record["nelec"],
                "n_occ_spatial": record["n_occ_spatial"],
                "gap_qh9_ev": record["gap_qh9_ev"],
                "homo_qh9_ev": record["homo_qh9_ev"],
                "lumo_qh9_ev": record["lumo_qh9_ev"],
            }
        )
        for record in records
    ]

    string_dtype = h5py.string_dtype(encoding="utf-8")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.out, "w") as h5:
        h5.attrs["basis"] = args.basis
        h5.attrs["method"] = "QH9 generalized diagonalization"
        h5.attrs["source"] = "QH9"
        h5.attrs["hamiltonian_type"] = "DFT/Kohn-Sham Hamiltonian from QH9"
        h5.attrs["scf_run"] = False
        h5.attrs["dataset"] = args.dataset
        h5.attrs["split"] = resolved_split
        h5.attrs["subset"] = args.subset
        h5.attrs["loader_module"] = loader_module
        h5.attrs["target_nao"] = target_nao
        h5.attrs["target_n_spin_orbitals"] = target_n_spin_orbitals
        h5.attrs["target_nelec"] = "variable" if target_nelec is None else int(target_nelec)
        h5.attrs["label_source"] = args.label_source
        h5.attrs["gap_threshold_ev"] = threshold
        h5.attrs["spin_orbital_ordering"] = "alpha block first, beta block second"
        h5.attrs["ham_ordering"] = args.ham_ordering
        h5.attrs["basis_note"] = "basis must match QH9 Hamiltonian basis; default is def2-svp"
        h5.attrs["W_padding"] = "columns after nelec[i] are zero padding"

        h5.create_dataset("W", data=w_all)
        h5.create_dataset("D_occ", data=d_all)
        h5.create_dataset("nelec", data=nelec_arr)
        h5.create_dataset("n_occ_spatial", data=n_occ_spatial_arr)

        h5.create_dataset("y_gap_binary", data=labels)
        h5.create_dataset("gap_qh9_ev", data=gap_qh9_ev)
        h5.create_dataset("gap_qm9_ev", data=gap_qm9_ev)
        h5.create_dataset("homo_qh9_ev", data=homo_qh9_ev)
        h5.create_dataset("lumo_qh9_ev", data=lumo_qh9_ev)
        h5.create_dataset("orbital_energy_hartree", data=orbital_energy_all)

        gap_hf_alias = h5.create_dataset("gap_hf_ev", data=gap_qh9_ev)
        gap_hf_alias.attrs["compatibility_alias"] = "same as gap_qh9_ev; not Hartree-Fock"

        homo_hf_alias = h5.create_dataset("homo_hf_ev", data=homo_qh9_ev)
        homo_hf_alias.attrs["compatibility_alias"] = "same as homo_qh9_ev; not Hartree-Fock"

        lumo_hf_alias = h5.create_dataset("lumo_hf_ev", data=lumo_qh9_ev)
        lumo_hf_alias.attrs["compatibility_alias"] = "same as lumo_qh9_ev; not Hartree-Fock"

        hf_energy_ds = h5.create_dataset("hf_energy_hartree", data=hf_energy)
        hf_energy_ds.attrs["unavailable"] = "No HF/SCF calculation is run in the QH9 workflow"

        mo_energy_alias = h5.create_dataset("mo_energy_hartree", data=orbital_energy_all)
        mo_energy_alias.attrs["compatibility_alias"] = (
            "same as orbital_energy_hartree; Kohn-Sham/QH9 eigenvalues, not HF"
        )

        h5.create_dataset("qh9_index", data=qh9_indices)
        qm9_index_alias = h5.create_dataset("qm9_index", data=qh9_indices)
        qm9_index_alias.attrs["compatibility_alias"] = "same as qh9_index/original_index"
        h5.create_dataset("original_index", data=qh9_indices)

        meta_arr = np.array(metadata_json, dtype=string_dtype)
        h5.create_dataset("metadata_json", data=meta_arr)

    print()
    print(f"Saved {len(records)} molecules to {args.out}")
    print(f"W shape: {w_all.shape}")
    print(f"D_occ shape: {d_all.shape}")
    print(f"Binary label threshold: {threshold:.6f} eV")
    print(f"Positive labels: {int(labels.sum())}")
    print(f"Negative labels: {int(len(labels) - labels.sum())}")


if __name__ == "__main__":
    main()
