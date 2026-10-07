# 4_source_sign_grelu_minimal: reading a hidden bit from three qubits

A 50-qubit chain starts in its ground state. Qubit 25 is turned a quarter turn about z, one way (s = +1) or the other (s = −1), and the chain evolves under the XXX Hamiltonian. A single quantized GReLU neuron (He, Liu & Wilde, arXiv:2605.24386), trained with **Algorithm 8**, reads the hidden bit s from the 3-qubit state of sites 24–26. The data are in [`../data/source_sign_mps_N50/`](../data/source_sign_mps_N50/) (see its `README.md`).

**Start with [pipeline.ipynb](pipeline.ipynb).** It holds all the code, with plain-language explanations.

```
pipeline.ipynb  the whole pipeline with explanations (Julia 1.12 kernel)
neuron.jl       the neuron on k qubits: GReLU, H(θ), exact Alg 8 loss and gradient, firing (Alg 5)
data.jl         load region states, fix their axis and bit order, check them, split by time
run.jl          cut regions -> load/check -> split -> train -> test -> fire -> figures
plots.jl        the six figures, redrawn from results/ (run.jl calls it)
regions/        region states cut by the dataset's make_regions.py (generated, not in git)
results/        CSVs (generated, not in git)
figures/        PNGs from the latest run
```

The `.jl` files hold exactly the notebook's code cells.

## Running

Julia 1.12 with HDF5, Optimisers, Plots and SpecialFunctions, plus Python 3 with numpy and h5py for the first run, which cuts the region states with the dataset's own tool (~30 s). The dataset's `.h5` state files (3.8 GB) are not in git and must be placed in `data/source_sign_mps_N50/XX/` and `XXX/`. From the repository root:

```bash
julia 4_source_sign_grelu_minimal/run.jl      # ~15 s after the regions exist
```

## Setup

- **Samples:** the 8 × 8 density matrix of sites 24, 25, 26 (k = 3, centred on the turned qubit, d = 0), XXX chain, t = 0.1 to 7.5, both labels: 150 states. t = 0 is left out because the neuron's real-valued terms give both labels the same output there. Later times are left out because the signal reflects off the chain ends at t ≈ 7.6.
- **Index order:** the stored matrices are read with their axes swapped back and their bits reversed into `kron` order. Four checks confirm this against values the simulation stored itself.
- **Split:** whole times are held out (both labels together, since they are exact spin flips of each other): 60 training times and 15 test times.
- **Neuron:** terms I, Z, and neighbouring ZZ, XX, YY (10 weights). Squared loss with targets 1 (s = +1) and 0 (s = −1), and 300 Adam steps at learning rate 0.05 with weight decay 10⁻⁴.
- **Outputs:** training history, θ and its gradient at every step, final weights, the output of every state, and test accuracy from firing the neuron with 1 to 4096 copies of each state.
