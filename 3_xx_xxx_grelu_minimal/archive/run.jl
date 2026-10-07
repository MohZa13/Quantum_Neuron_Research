# run.jl — the whole minimal pipeline:  data -> split -> train -> test -> fire -> figures.
# (plots.jl redraws the figures from results/ alone, without retraining.)
#
#   julia 3_xx_xxx_grelu_minimal/run.jl          (from the repository root; ~15 min)
#
# Writes results/*.csv and figures/*.png next to this file.

include(joinpath(@__DIR__, "data.jl"))
using Optimisers, Printf

const DATA = joinpath(@__DIR__, "..", "data", "xx_xxx_thermal_states", "xx_xxx_n10.h5")
const RESULTS = mkpath(joinpath(@__DIR__, "results"))
const FIGURES = mkpath(joinpath(@__DIR__, "figures"))

# Settings (chosen by cross-validation in folder 2)
const T = 1.0                                       # activation width of GReLU_T
const L2 = Dict(:margin => 1e-3, :square => 1e-4)   # weight decay
const STEPS, LR, SEED = 200, 0.05, 20260923
const NAME = Dict(:margin => "Alg 9 (margin loss)", :square => "Alg 8 (squared loss)")

writecsv(path, header, rows) = open(io -> (println(io, join(header, ","));
                                           foreach(r -> println(io, join(r, ",")), rows)), path, "w")

# ------------------------------------------------------------------ 1. data ---

println("loading data ...")
ds = load_dataset(DATA)
train, test = split_by_chain(ds)
@printf("  %d states from %d chains;  train %d states (%d chains),  test %d states (%d chains)\n",
        length(ds.y), length(ds.chains), length(train), length(unique(ds.chain[train])),
        length(test), length(unique(ds.chain[test])))

accuracy(score, idx) = mean((score .>= 0) .== (ds.y[idx] .> 0))

# ----------------------------------------------------------------- 2. train ---

"""
Adam on loss + L2/2 |theta|^2 with the exact gradient.  `kind` is :margin
(Algorithm 9) or :square (Algorithm 8).  Returns the weights and a history.
"""
function train_neuron(kind)
  names, terms = neuron_terms(ds.n; bias=(kind == :square))
  theta = 0.1 .* randn(MersenneTwister(SEED), length(terms))
  opt = Optimisers.setup(Optimisers.Adam(LR), theta)
  kind == :margin && ((Rp, Rm) = class_averages(ds, train))
  history = []
  for step in 0:STEPS
    L, g = kind == :margin ? margin_loss_grad(terms, theta, T, Rp, Rm) :
                             square_loss_grad(ds, train, terms, theta, T)
    if step % 10 == 0
      acc = accuracy(scores(kind, ds, test, terms, theta, T), test)
      push!(history, (step, L, acc))
      @printf("  %-22s step %3d   train loss %.4f   test accuracy %.3f\n", NAME[kind], step, L, acc)
    end
    step == STEPS && break
    opt, theta = Optimisers.update!(opt, theta, g .+ L2[kind] .* theta)
  end
  return (; names, terms, theta, history)
end

models = Dict(kind => train_neuron(kind) for kind in (:margin, :square))

writecsv(joinpath(RESULTS, "history.csv"), ["model", "step", "train_loss", "test_accuracy"],
         [(k, h...) for k in (:margin, :square) for h in models[k].history])
writecsv(joinpath(RESULTS, "weights.csv"), ["model", "term", "theta"],
         [(k, n, t) for k in (:margin, :square) for (n, t) in zip(models[k].names, models[k].theta)])

# ------------------------------------------------------------------ 3. test ---

println("\ntest set (exact outputs = infinitely many copies of each state):")
rows = []
for kind in (:margin, :square)
  m = models[kind]
  s = scores(kind, ds, test, m.terms, m.theta, T)
  @printf("  %-22s accuracy %.3f   ||theta||_1 = %.1f\n", NAME[kind], accuracy(s, test), norm(m.theta, 1))
  append!(rows, [(kind, ds.group[i], ds.kT[i], Int(ds.y[i]), s[j]) for (j, i) in enumerate(test)])
end
writecsv(joinpath(RESULTS, "test_scores.csv"), ["model", "chain", "kT", "y", "score"], rows)

# ------------------------------------------------- 4. firing (Algorithm 5) ---

println("\nclassifying by firing the neuron (finite copies of each test state):")
const COPIES = [2, 8, 32, 128, 512, 2048]
const REPS = 10
rng = MersenneTwister(SEED + 1)
firing = Dict()
for kind in (:margin, :square)
  m = models[kind]
  E, V = eigen(Symmetric(hamiltonian(m.terms, m.theta)))
  pops = populations(ds, test, V)
  firing[kind] = map(COPIES) do N
    mean(mean(classify_by_firing(kind, E, pops[:, j], T, N, rng) == ds.y[i]
              for (j, i) in enumerate(test)) for _ in 1:REPS)
  end
  @printf("  %-22s %s\n", NAME[kind],
          join((@sprintf("%d copies: %.3f", N, a) for (N, a) in zip(COPIES, firing[kind])), "   "))
end
writecsv(joinpath(RESULTS, "firing.csv"), ["copies", "accuracy_alg9_model", "accuracy_alg8_model"],
         [(N, firing[:margin][i], firing[:square][i]) for (i, N) in enumerate(COPIES)])

include(joinpath(@__DIR__, "plots.jl"))   # figures, from results/
