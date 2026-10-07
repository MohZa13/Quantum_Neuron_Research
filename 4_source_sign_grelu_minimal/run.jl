# run.jl — the whole pipeline: cut region states -> load and check -> split -> train -> test -> fire
# -> figures.  (plots.jl redraws the figures from results/ alone.)
#
#   julia 4_source_sign_grelu_minimal/run.jl          (from the repository root; ~1 min)
#
# pipeline.ipynb holds the same code, cell by cell, with explanations.

const HERE = @__DIR__
include(joinpath(HERE, "data.jl"))

using Optimisers, Printf

# ---- which data ----
const DATA_DIR = joinpath(HERE, "..", "data", "source_sign_mps_N50")   # the full-chain dataset
const HAM = "XXX"              # Hamiltonian whose states we classify
const K, D = 3, 0              # region: K qubits centred D sites from the turned qubit 25 -> sites 24, 25, 26
const TMIN, TMAX = 0.1, 7.5    # times used. t = 0 is left out (see below), and XXX's signal reaches the chain
                               # ends at t ≈ 7.6, after which reflections mix in.
const REGION_FILE = joinpath(HERE, "regions", "$(HAM)_k$(K)_d$(D).h5")
const RESULTS = mkpath(joinpath(HERE, "results"))

# ---- training settings (the same as the XX vs XXX minimal pipeline) ----
const T = 1.0                                  # width of the GReLU smoothing
const L2 = 1e-4                                # weight decay
const STEPS, LR, SEED = 300, 0.05, 20261007    # Adam steps, learning rate, random seed

"Write rows of values to a CSV file with the given header."
writecsv(path, header, rows) = open(io -> (println(io, join(header, ","));
                                           foreach(r -> println(io, join(r, ",")), rows)), path, "w")

if !isfile(REGION_FILE)
  mkpath(dirname(REGION_FILE))
  run(Cmd(`python3 make_regions.py --hamiltonian $HAM --k $K --d $D --workers 4 --out $REGION_FILE`; dir=DATA_DIR))
end

R = load_regions(REGION_FILE; tmin=TMIN, tmax=TMAX)
@printf("%d states: %s, sites %s, t = %.1f to %.1f, %d times × 2 labels\n",
        length(R.s), HAM, R.sites, minimum(R.t), maximum(R.t), length(unique(R.cell)))
check_regions(R, DATA_DIR, HAM)

R0 = load_regions(REGION_FILE; tmin=0.0, tmax=0.0)
ρp, ρm = R0.rho[findfirst(==(1), R0.s)], R0.rho[findfirst(==(-1), R0.s)]
term_names, terms = neuron_terms(R.sites)
θ_random = randn(MersenneTwister(1), length(terms))
o = outputs(terms, θ_random, [ρp, ρm], T)
@printf("t = 0:  ρ₋ = conj(ρ₊) to %.1e;  trace distance between them %.2f;  outputs with random θ: %.6f vs %.6f\n",
        maximum(abs.(ρm .- conj(ρp))), R0.trace_distance[1], o...)

train, test = split_by_cell(R; seed=SEED)
@printf("train %d states (%d times),  test %d states (%d times)\n",
        length(train), length(unique(R.cell[train])), length(test), length(unique(R.cell[test])))

target(s) = s == 1 ? 1.0 : 0.0                          # Algorithm 8 targets: 1 for s = +1, 0 for s = −1
accuracy(out, idx) = mean((out .> 0.5) .== (R.s[idx] .== 1))

"""
Adam on the squared loss (+ weight decay) with the exact gradient.
Returns the final weights, θ and the loss gradient at every step (column s+1 = step s), and a history.
"""
function train_neuron(terms)
  rhos, targets = R.rho[train], target.(R.s[train])
  theta = 0.1 .* randn(MersenneTwister(SEED), length(terms))      # small random starting weights
  opt = Optimisers.setup(Optimisers.Adam(LR), theta)
  thetas, grads = zeros(length(terms), STEPS + 1), zeros(length(terms), STEPS + 1)
  history = []
  for step in 0:STEPS
    L, g = loss_grad(terms, theta, rhos, targets, T)               # loss and exact gradient at the current θ
    thetas[:, step + 1], grads[:, step + 1] = theta, g             # record where θ is and where it is pushed
    if step % 20 == 0
      acc_train = accuracy(outputs(terms, theta, R.rho[train], T), train)
      acc_test = accuracy(outputs(terms, theta, R.rho[test], T), test)
      push!(history, (step, L, acc_train, acc_test))
      @printf("  step %3d   train loss %.4f   train accuracy %.3f   test accuracy %.3f   ||θ||_1 %.2f\n",
              step, L, acc_train, acc_test, norm(theta, 1))
    end
    step == STEPS && break
    opt, theta = Optimisers.update!(opt, theta, g .+ L2 .* theta)  # one Adam step
  end
  return (; theta=copy(theta), thetas, grads, history)
end

model = train_neuron(terms)

writecsv(joinpath(RESULTS, "history.csv"), ["step", "train_loss", "train_accuracy", "test_accuracy"], model.history)
writecsv(joinpath(RESULTS, "weights.csv"), ["term", "theta"], zip(term_names, model.theta))
writecsv(joinpath(RESULTS, "theta_trajectory.csv"), ["step", "term", "theta", "grad"],
         [(st, term_names[j], model.thetas[j, st + 1], model.grads[j, st + 1]) for st in 0:STEPS for j in eachindex(term_names)])

out = outputs(terms, model.theta, R.rho, T)
@printf("exact outputs:  train accuracy %.3f   test accuracy %.3f   ||θ||_1 = %.2f\n",
        accuracy(out[train], train), accuracy(out[test], test), norm(model.theta, 1))

set_of(m) = m in test ? "test" : "train"
writecsv(joinpath(RESULTS, "outputs.csv"), ["t", "s", "set", "output", "trace_distance"],
         [(R.t[m], R.s[m], set_of(m), out[m], R.trace_distance[m]) for m in eachindex(R.s)])

const COPIES = [1, 4, 16, 64, 256, 1024, 4096]
const REPS = 20
rng = MersenneTwister(SEED + 1)
E, V = eigen(hamiltonian(terms, model.theta))
pops = [real(diag(V' * R.rho[m] * V)) for m in test]      # each test state's weight on each eigenvector of H
firing = map(COPIES) do N
  mean(mean(classify_by_firing(E, pops[j], T, N, rng) == R.s[m] for (j, m) in enumerate(test)) for _ in 1:REPS)
end
println(join((@sprintf("%d copies: %.3f", N, a) for (N, a) in zip(COPIES, firing)), "   "))
writecsv(joinpath(RESULTS, "firing.csv"), ["copies", "accuracy"], zip(COPIES, firing))

Rall = load_regions(REGION_FILE)
Zq = [pauli(K, [(q, 'Z')]) for q in 1:K]
writecsv(joinpath(RESULTS, "overview.csv"), ["t", "s", "trace_distance", ["z$(x)" for x in Rall.sites]...],
         [(Rall.t[m], Rall.s[m], Rall.trace_distance[m], [real(tr(Z * Rall.rho[m])) for Z in Zq]...)
          for m in eachindex(Rall.s)])

include(joinpath(HERE, "plots.jl"))   # figures, from results/
