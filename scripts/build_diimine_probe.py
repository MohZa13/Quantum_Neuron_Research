"""Build hexa-2,4-diene-1,6-diimine at QH9's level of theory and export
integrals-only Module-G-layout files for CAS(10,10)..CAS(20,20).

The molecule is NOT in QH9 Stable (verified: no unbranched C6H8N2 exists
there), so it is constructed here the way QH9 built its records: B3LYP with
def2-SVP. Deviation from QH9: the geometry is RDKit ETKDG + MMFF94 (no DFT
optimizer in this environment) with a B3LYP single point on top — adequate
for a timing/entanglement probe, stated in the file meta.

No solver runs: the CI sectors (up to 3.4e10 at CAS(20,20)) are exactly what
the MPS pipeline exists to bypass. Files carry only (ecore, h1eff, g, Z) +
meta, which is all `QThermalMPS.read_case` needs.
"""
import numpy as np, h5py, os
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from qthermal.active_space import select_active
from qthermal.hamiltonian import build_cas_hamiltonian
from qthermal.orbitals import _canonicalize_signs

SCRATCH = os.path.dirname(os.path.abspath(__file__))

# --- geometry: all-trans conjugated chain, MMFF-relaxed ------------------
mol_rd = Chem.AddHs(Chem.MolFromSmiles("N=C/C=C/C=C/C=N"))
params = AllChem.ETKDGv3()
params.randomSeed = 7
cids = AllChem.EmbedMultipleConfs(mol_rd, numConfs=12, params=params)
res = AllChem.MMFFOptimizeMoleculeConfs(mol_rd, maxIters=2000)
energies = [e for (conv, e) in res]
best = int(np.argmin(energies))
print(f"conformers: {len(cids)}, MMFF energies kcal/mol: "
      f"min {min(energies):.2f} max {max(energies):.2f}, using conf {best}")
conf = mol_rd.GetConformer(best)
Z = np.array([a.GetAtomicNum() for a in mol_rd.GetAtoms()])
R = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
               conf.GetAtomPosition(i).z] for i in range(mol_rd.GetNumAtoms())])
with open(os.path.join(SCRATCH, "diimine.xyz"), "w") as f:
    f.write(f"{len(Z)}\nhexa-2,4-diene-1,6-diimine MMFF geometry\n")
    for z, r in zip(Z, R):
        f.write(f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} "
                f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")

# --- B3LYP/def2-SVP single point (QH9's level of theory) -----------------
mol = gto.M(atom=[(int(z), r) for z, r in zip(Z, R)], unit="Angstrom",
            basis="def2-svp", charge=0, spin=0, verbose=0)
mf = dft.RKS(mol)
mf.xc = "b3lyp"
E = mf.kernel()
assert mf.converged, "SCF did not converge"
nocc = mol.nelectron // 2
eps = mf.mo_energy
gap = eps[nocc] - eps[nocc - 1]
print(f"nao={mol.nao}  nelec={mol.nelectron}  nocc={nocc}  "
      f"E(B3LYP)={E:.6f}  gap={gap:.5f} Ha")
C = _canonicalize_signs(np.asarray(mf.mo_coeff))

# --- active spaces -> Module-G-layout files ------------------------------
for nact in (5, 6, 7, 8, 9, 10):
    aspace = select_active(eps, nocc, n_act_occ=nact, n_act_virt=nact)
    ham = build_cas_hamiltonian(mol, C, nocc, aspace)
    path = os.path.join(SCRATCH, f"diimine_cas{aspace.ncas}.h5")
    with h5py.File(path, "w") as f:
        m = f.create_group("meta")
        m.attrs["ncas"] = aspace.ncas
        m.attrs["nelecas"] = aspace.nelecas
        m.attrs["basis"] = "def2-svp"
        m.attrs["xc"] = "b3lyp"
        m.attrs["source"] = ("custom molecule, NOT QH9: hexa-2,4-diene-1,6-"
                             "diimine; RDKit MMFF geometry + B3LYP/def2-SVP "
                             "single point (no DFT optimizer available)")
        m.attrs["scf_energy"] = E
        m.attrs["gap_Ha"] = gap
        g = f.create_group("mol_0")
        g.create_dataset("h1eff", data=ham.h1eff, compression="gzip")
        g.create_dataset("g", data=ham.g, compression="gzip")
        g.create_dataset("ecore", data=ham.ecore)
        g.create_dataset("Z", data=Z)
        g.attrs["complete"] = True
    print(f"  wrote {os.path.basename(path)}: ncas={aspace.ncas} "
          f"nelecas={aspace.nelecas} dim={aspace.dim:,} ecore={ham.ecore:.6f}")
print("done")
