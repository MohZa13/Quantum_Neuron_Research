# run.jl — the whole pipeline:  data -> split -> train -> circuit check -> test -> fire -> figures.
# (plots.jl redraws the figures from results/ alone, without retraining.)
#
#   julia 3_xx_xxx_grelu_minimal/run.jl          (from the repository root; ~12 min)
#
# Writes results/*.csv and figures/*.png next to this file.  pipeline.ipynb holds
# the same code, cell by cell, with explanations.

const HERE = @__DIR__
include(joinpath(HERE, "data.jl"))

using Optimisers, Printf

const DATA = joinpath(HERE, "..", "data", "xx_xxx_thermal_states", "xx_xxx_n10.h5")
const RESULTS = mkpath(joinpath(HERE, "results"))

const T = 1.0                                # activation width of GReLU_T
const L2 = 1e-4                              # weight decay
const STEPS, LR, SEED = 200, 0.05, 20260923  # Adam steps, learning rate, seed
const CHECKPOINTS = [0, 10, 20, 50, 100, 200]          # steps at which the circuit is simulated
const VALUE_RUNS = [10^2, 10^3, 10^4, 10^5, 10^6]     # value-block runs (cheap)
const GRAD_RUNS = [10, 30, 100, 300]                  # full Algorithm 8 runs (~30 ms each)

writecsv(path, header, rows) = open(io -> (println(io, join(header, ","));
                                           foreach(r -> println(io, join(r, ",")), rows)), path, "w")

println("loading data ...")
ds = load_dataset(DATA)
train, test = split_by_state(ds)
@printf("  %d states from %d chains;  train %d states (from %d chains),  test %d states (from %d chains)\n",
        length(ds.y), length(ds.chains), length(train), length(unique(ds.chain[train])),
        length(test), length(unique(ds.chain[test])))

accuracy(score, idx) = mean((score .>= 0) .== (ds.y[idx] .> 0))
term_names, terms = neuron_terms(ds.n)
println("  neuron: ", length(terms), " terms: ", join(term_names, " "))

"""
Adam on loss + L2/2 |theta|^2 with the exact gradient.  Returns the final
weights, theta and the loss gradient at every step (column s+1 = step s), and
a history of loss and test accuracy every 10 steps.
"""
function train_neuron(terms)
  theta = 0.1 .* randn(MersenneTwister(SEED), length(terms))
  opt = Optimisers.setup(Optimisers.Adam(LR), theta)
  thetas, grads = zeros(length(terms), STEPS + 1), zeros(length(terms), STEPS + 1)
  history = []
  for step in 0:STEPS
    L, g = square_loss_grad(ds, train, terms, theta, T)
    thetas[:, step + 1], grads[:, step + 1] = theta, g
    if step % 10 == 0
      acc = accuracy(scores(ds, test, terms, theta, T), test)
      push!(history, (step, L, acc))
      @printf("  step %3d   train loss %.4f   test accuracy %.3f   ||theta||_1 %.2f\n",
              step, L, acc, norm(theta, 1))
    end
    step == STEPS && break
    opt, theta = Optimisers.update!(opt, theta, g .+ L2 .* theta)
  end
  return (; theta=copy(theta), thetas, grads, history)
end

model = train_neuron(terms)

writecsv(joinpath(RESULTS, "history.csv"), ["step", "train_loss", "test_accuracy"], model.history)
writecsv(joinpath(RESULTS, "weights.csv"), ["term", "theta"], zip(term_names, model.theta))
writecsv(joinpath(RESULTS, "theta_trajectory.csv"), ["step", "term", "theta", "grad"],
         [(s, term_names[j], model.thetas[j, s + 1], model.grads[j, s + 1]) for s in 0:STEPS for j in eachindex(term_names)])

C = pauli_expectations(ds, train, terms)
i0 = findfirst(i -> ds.y[train[i]] > 0, eachindex(train))     # one XXX training state for the value check
@printf("value check on training state %s at kT = %.3g\n", ds.group[train[i0]], ds.kT[train[i0]])
rng = MersenneTwister(SEED + 2)
vrows, grows, nrows = [], [], []
for step in CHECKPOINTS
  th, g = model.thetas[:, step + 1], model.grads[:, step + 1]
  nt1 = norm(th, 1)
  # 1. the value block alone
  est, exact = value_block_estimates(ds, train[i0], C[i0, :], terms, th, T; runs=VALUE_RUNS, rng)
  append!(vrows, [(step, nt1, N, v, exact) for (N, v) in zip(VALUE_RUNS, est)])
  # 2. the full Algorithm 8 gradient
  out = alg8_circuit_runs(ds, train, C, terms, th, T; nruns=maximum(GRAD_RUNS), rng)
  append!(grows, [(step, nt1, N, norm(vec(mean(out[:, 1:N]; dims=2)) - g) / norm(g)) for N in GRAD_RUNS])
  noise = sqrt(sum(var(out; dims=2)))          # spread of one run's output (norm over j)
  n10 = (noise / (0.1 * norm(g)))^2            # runs needed for a 10% relative error
  push!(nrows, (step, nt1, norm(g), noise, n10))
  @printf("  step %3d  ||theta||_1 %5.2f   value: exact %.3f, after 10^6 runs %.3f   gradient: |g| %.1e, one-run spread %.1e  ->  %.1e runs for 10%% error\n",
          step, nt1, exact, est[end], norm(g), noise, n10)
end
writecsv(joinpath(RESULTS, "circuit_value.csv"), ["step", "theta_l1", "runs", "estimate", "exact"], vrows)
writecsv(joinpath(RESULTS, "circuit_gradient.csv"), ["step", "theta_l1", "runs", "rel_error"], grows)
writecsv(joinpath(RESULTS, "circuit_cost.csv"), ["step", "theta_l1", "grad_norm", "run_spread", "runs_for_10pct"], nrows)

sc = scores(ds, test, terms, model.theta, T)
@printf("test accuracy %.3f   ||theta||_1 = %.1f   bias theta_I = %.2f\n",
        accuracy(sc, test), norm(model.theta, 1), model.theta[1])
writecsv(joinpath(RESULTS, "test_scores.csv"), ["chain", "kT", "y", "score"],
         [(ds.group[i], ds.kT[i], Int(ds.y[i]), sc[j]) for (j, i) in enumerate(test)])

const COPIES = [2, 8, 32, 128, 512, 2048]
const REPS = 10
rng = MersenneTwister(SEED + 1)
E, V = eigen(Symmetric(hamiltonian(terms, model.theta)))
pops = populations(ds, test, V)
firing = map(COPIES) do N
  mean(mean(classify_by_firing(E, pops[:, j], T, N, rng) == ds.y[i]
            for (j, i) in enumerate(test)) for _ in 1:REPS)
end
println(join((@sprintf("%d copies: %.3f", N, a) for (N, a) in zip(COPIES, firing)), "   "))
writecsv(joinpath(RESULTS, "firing.csv"), ["copies", "accuracy"], zip(COPIES, firing))

include(joinpath(HERE, "plots.jl"))   # figures, from results/
