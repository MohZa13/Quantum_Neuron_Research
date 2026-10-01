# ffnn_xx_xxx.jl
#
# Classical control for the GReLU neuron: the simplest feed-forward network,
#
#     x (37) --dense--> 10 ReLU --dense--> 1 logit,     391 parameters,
#
# on the SAME chain-level train/test split (../results/split.csv).
#
# Input x_m = the 37 local expectation values Tr[H_j rho_m] (Z_i, Z_iZ_{i+1},
# X_iX_{i+1}, Y_iY_{i+1}) -- exactly what the quantum neuron's H(theta) reads,
# standardised with training-set statistics only.  (<Z_i> = 0 for every state
# by symmetry, so those 10 inputs carry nothing; they are kept so the input
# matches the quantum model's term list.)  The 10 hidden units are a dense layer
# -- "one per site" in count only, not wired to sites.
#
# Trained full-batch with Adam on binary cross-entropy + l2, 5 seeds.  A
# label-shuffle control retrains 50 times with the labels of the 68 TRAINING
# chains permuted (whole chains, balanced) and tests on the true labels: if the
# pipeline leaked test information this would still score well; if it is sound
# it averages chance.  A single permutation still agrees with the truth on
# ~50 +- 6% of chains, so individual shuffles scatter around 0.5 -- hence 50.
#
#   julia experiments/xx_xxx_grelu/src/ffnn_xx_xxx.jl                 # everything
#   julia experiments/xx_xxx_grelu/src/ffnn_xx_xxx.jl main transfer measure
#
# `transfer`: train on hot (kT >= 0.5) training states, test on cold test
# states, and the reverse -- the same split as the quantum neuron's figure F1.
#
# `measure`: the network's inputs must themselves be MEASURED on hardware.  All
# 37 are covered by three measurement settings: a Z-basis measurement of every
# qubit gives every Z_i and Z_iZ_{i+1} at once, an X-basis one every X_iX_{i+1},
# a Y-basis one every Y_iY_{i+1}.  With S copies of rho per state, S/3 go to
# each basis; each copy yields one bitstring drawn from the EXACT joint outcome
# distribution p(b) = <b|U rho U^dag|b> of that basis.  The networks are
# trained on exact inputs (as the quantum neurons were trained with exact
# gradients) and tested on the estimated ones -- the same footing as firing.
#
# Output: ../results/ffnn_{summary,history,predictions,shuffle50}.csv,
#         ffnn_kT_transfer_predictions.csv, ffnn_measure.csv

isdefined(Main, :CFG) || include(joinpath(@__DIR__, "train_xx_xxx_grelu.jl"))    # loader, split, auc, write_rows

const FF = (hidden=10, lr=1e-2, l2=1e-4, epochs=1000, seeds=1:5, log_every=10)

relu(x) = max(x, zero(x))
sigmoid(z) = 1 / (1 + exp(-z))

function init_mlp(rng, d, h)
  return (W1=randn(rng, h, d) .* sqrt(2 / d), b1=zeros(h),
          W2=randn(rng, 1, h) .* sqrt(2 / h), b2=zeros(1))
end

"Logits for every column of X (d x M)."
logits(p, X) = vec(p.W2 * relu.(p.W1 * X .+ p.b1) .+ p.b2)

"Mean binary cross-entropy + l2/2 |W|^2 and its gradient (manual backprop)."
function mlp_loss_grad(p, X, y; l2=FF.l2)
  A = p.W1 * X .+ p.b1                  # h x M pre-activations
  Hd = relu.(A)
  z = vec(p.W2 * Hd .+ p.b2)
  q = sigmoid.(z)
  M = length(y)
  L = -mean(y .* log.(q .+ 1e-12) .+ (1 .- y) .* log.(1 .- q .+ 1e-12)) +
      l2 / 2 * (sum(abs2, p.W1) + sum(abs2, p.W2))
  dz = permutedims((q .- y) ./ M)       # 1 x M
  dH = (p.W2' * dz) .* (A .> 0)          # h x M
  g = (W1=dH * X' .+ l2 .* p.W1, b1=vec(sum(dH; dims=2)),
       W2=dz * Hd' .+ l2 .* p.W2, b2=[sum(dz)])
  return L, g
end

function metrics(p, X, y)
  z = logits(p, X)
  q = sigmoid.(z)
  L = -mean(y .* log.(q .+ 1e-12) .+ (1 .- y) .* log.(1 .- q .+ 1e-12))
  return (; loss=L, acc=mean((z .>= 0) .== (y .> 0.5)), auc=auc(z, 2 .* y .- 1), z)
end

function fit(Xtr, ytr, Xte, yte; seed, tag)
  p = init_mlp(MersenneTwister(seed), size(Xtr, 1), FF.hidden)
  opt = Optimisers.setup(Optimisers.Adam(FF.lr), p)
  hist = NamedTuple[]
  for ep in 0:FF.epochs
    L, g = mlp_loss_grad(p, Xtr, ytr)
    if ep % FF.log_every == 0 || ep == FF.epochs
      tr, te = metrics(p, Xtr, ytr), metrics(p, Xte, yte)
      push!(hist, (; run=tag, seed, epoch=ep, train_obj=L, train_loss=tr.loss, train_acc=tr.acc,
                   train_auc=tr.auc, test_loss=te.loss, test_acc=te.acc, test_auc=te.auc))
    end
    ep == FF.epochs && break
    opt, p = Optimisers.update!(opt, p, g)
  end
  return p, hist
end

"Standardise with the statistics of the training columns only."
function standardise(X, tr)
  mu = mean(X[:, tr]; dims=2)
  sd = std(X[:, tr]; dims=2)
  sd[sd .< 1e-8] .= 1.0
  return mu, sd
end

"Temperature transfer: hot (kT >= 0.5) <-> cold, same split as figure F1."
function exp_transfer(ds, split, X, y)
  println("\n   temperature transfer (5 seeds)")
  hot = ds.kT .>= 0.5
  rows = NamedTuple[]
  for (tag, trm, tem) in (("hot_to_cold", hot, .!hot), ("cold_to_hot", .!hot, hot))
    tr = findall(m -> split[m] == "train" && trm[m], 1:length(ds))
    te = findall(m -> split[m] == "test" && tem[m], 1:length(ds))
    mu, sd = standardise(X, tr)
    Xs = (X .- mu) ./ sd
    for seed in FF.seeds
      p, _ = fit(Xs[:, tr], y[tr], Xs[:, te], y[te]; seed, tag)
      ev = metrics(p, Xs[:, te], y[te])
      for (i, m) in enumerate(te)
        push!(rows, (; run=tag, seed, sample=ds.names[m], model=ds.model[m], group=ds.group[m],
                     kT=ds.kT[m], y=Int(ds.y[m]), logit=ev.z[i],
                     correct=Int((ev.z[i] >= 0) == (y[m] > 0.5))))
      end
      @printf("   %-12s seed %d: test acc %.3f  AUC %.3f\n", tag, seed, ev.acc, ev.auc)
    end
  end
  write_rows(joinpath(OUT, "ffnn_kT_transfer_predictions.csv"), rows)
end

"The 2^n outcome probabilities of measuring every qubit in `basis` (Z, X or Y)."
function basis_probs(R::Matrix{Float64}, U::Matrix{ComplexF64})
  return max.(real(vec(sum((U * R) .* conj(U); dims=2))), 0.0)   # diag(U R U^dag)
end

"Single-qubit rotation taking the measured Pauli to Z."
const ROT = Dict('Z' => ComplexF64[1 0; 0 1],
                 'X' => ComplexF64[1 1; 1 -1] ./ sqrt(2),                 # H
                 'Y' => (ComplexF64[1 1; 1 -1] ./ sqrt(2)) * ComplexF64[1 0; 0 -im])   # H S^dag

"Finite-measurement inputs: S copies per test state, S/3 in each of the Z, X, Y bases."
function exp_measure(ds, split, X, y, Xs, tr, te)
  println("\n   finite-measurement inputs")
  n = ds.n
  terms = model_terms(n)
  basis_of(t) = t[1] in (:z, :zz) ? 'Z' : t[1] === :xx ? 'X' : 'Y'
  sites(t) = t[1] === :z ? (t[2],) : (t[2], t[2] + 1)
  bit(b, s) = (b >> (n - s)) & 1                                   # site 1 = most significant
  U = Dict(c => reduce(kron, fill(ROT[c], n)) for c in ('Z', 'X', 'Y'))
  # value of each term's Pauli product on each outcome bitstring of its basis
  cols = Dict(c => findall(t -> basis_of(t) == c, terms) for c in ('Z', 'X', 'Y'))
  vals = Dict(c => [(-1.0)^sum(bit(b, s) for s in sites(terms[j])) for b in 0:(2^n - 1), j in cols[c]]
              for c in ('Z', 'X', 'Y'))
  # exact outcome distributions per test state, checked against the exact inputs
  P = Dict{Tuple{Int,Char},Vector{Float64}}()
  err = 0.0
  for m in te
    R = GN.to_full(ds.B, GN.rho(ds.B, ds.fams[ds.fam[m]], ds.beta[m]))
    for c in ('Z', 'X', 'Y')
      p = basis_probs(R, U[c])
      P[(m, c)] = p ./ sum(p)
      err = max(err, maximum(abs, vals[c]' * P[(m, c)] .- X[cols[c], m]))
    end
  end
  @printf("   outcome distributions reproduce the exact inputs: max error %.1e\n", err)
  err < 1e-10 || error("measurement model is wrong")

  mu, sd = standardise(X, tr)
  nets = [fit(Xs[:, tr], y[tr], Xs[:, te], y[te]; seed, tag="ffnn")[1] for seed in FF.seeds]
  rows = NamedTuple[]
  for S in (3, 12, 48, 192, 768, 3072, 12288), r in 1:10
    rng = MersenneTwister(4099r + S)
    k = S ÷ 3
    Xe = zeros(length(terms), length(te))
    for (i, m) in enumerate(te), c in ('Z', 'X', 'Y')
      cdf = cumsum(P[(m, c)])
      counts = zeros(2^n)
      for _ in 1:k
        counts[min(searchsortedfirst(cdf, rand(rng)), 2^n)] += 1
      end
      Xe[cols[c], i] .= vals[c]' * counts ./ k
    end
    Xes = (Xe .- mu) ./ sd
    for (seed, p) in zip(FF.seeds, nets)
      z = logits(p, Xes)
      for (i, m) in enumerate(te)
        push!(rows, (; copies=S, rep=r, seed, sample=ds.names[m], model=ds.model[m],
                     group=ds.group[m], kT=ds.kT[m], logit=z[i],
                     correct=Int((z[i] >= 0) == (y[m] > 0.5))))
      end
    end
    r == 10 && @printf("   S = %5d copies/state (%4d per basis): test acc %.3f (mean over 5 nets x 10 repeats)\n",
                       S, k, mean(x.correct for x in rows if x.copies == S))
  end
  write_rows(joinpath(OUT, "ffnn_measure.csv"), rows)
end

function main(parts=("main", "transfer", "measure"))
  ds = XXXData.load(DATA)
  split, _ = load_split(ds)
  terms = model_terms(ds.n)
  X = permutedims(subproblem(ds, 1:length(ds), [Pauli(ds.B, t) for t in terms], 1.0).C)  # 37 x 840
  y = Float64.(ds.label)                                   # 1 = XXX
  tr, te = findall(==("train"), split), findall(==("test"), split)
  mu = mean(X[:, tr]; dims=2)
  sd = std(X[:, tr]; dims=2)
  sd[sd .< 1e-8] .= 1.0                                    # the <Z_i> = 0 columns
  Xs = (X .- mu) ./ sd
  # backprop vs central finite differences on a random network
  pc = init_mlp(MersenneTwister(7), size(Xs, 1), FF.hidden)
  _, gc = mlp_loss_grad(pc, Xs[:, tr], y[tr])
  err = maximum(keys(pc)) do k
    maximum(eachindex(pc[k])) do i
      a, b = deepcopy(pc), deepcopy(pc)
      a[k][i] += 1e-6; b[k][i] -= 1e-6
      abs((mlp_loss_grad(a, Xs[:, tr], y[tr])[1] - mlp_loss_grad(b, Xs[:, tr], y[tr])[1]) / 2e-6 - gc[k][i])
    end
  end
  @printf("backprop vs finite differences: max abs err %.1e\n", err)
  err < 1e-6 || error("backprop gradient is wrong")
  "transfer" in parts && exp_transfer(ds, split, X, y)
  "measure" in parts && exp_measure(ds, split, X, y, Xs, tr, te)
  "main" in parts || return
  nparam = FF.hidden * (length(terms) + 1) + FF.hidden + 1
  @printf("\nFFNN %d -> %d ReLU -> 1   (%d parameters)   %d train / %d test states\n",
          length(terms), FF.hidden, nparam, length(tr), length(te))

  H, S, P = NamedTuple[], NamedTuple[], NamedTuple[]
  for seed in FF.seeds
    p, h = fit(Xs[:, tr], y[tr], Xs[:, te], y[te]; seed, tag="ffnn")
    append!(H, h)
    ev = metrics(p, Xs[:, te], y[te])
    first_hit = something(findfirst(r -> r.test_acc == 1, h), 0)
    push!(S, (; run="ffnn", seed, test_acc=ev.acc, test_auc=ev.auc, test_loss=ev.loss,
              train_acc=h[end].train_acc, first_epoch_test_acc_1=first_hit == 0 ? -1 : h[first_hit].epoch))
    @printf("   seed %d: test acc %.3f  AUC %.3f   (train acc %.3f)\n", seed, ev.acc, ev.auc, h[end].train_acc)
    if seed == first(FF.seeds)
      append!(P, [(; run="ffnn", seed, sample=ds.names[m], model=ds.model[m], group=ds.group[m],
                    kT=ds.kT[m], y=Int(ds.y[m]), logit=ev.z[i], correct=Int((ev.z[i] >= 0) == (y[m] > 0.5)))
                  for (i, m) in enumerate(te)])
    end
  end

  # label-shuffle control: permute class labels across the 68 training CHAINS
  groups = unique(ds.group[tr])
  truth = [ds.label[findfirst(==(g), ds.group)] for g in groups]
  R = NamedTuple[]
  for s in 1:50
    glab = shuffle(MersenneTwister(1000 + s), truth)
    yperm = Float64[glab[findfirst(==(ds.group[m]), groups)] for m in tr]
    p, _ = fit(Xs[:, tr], yperm, Xs[:, te], y[te]; seed=s, tag="shuffled_labels")
    ev = metrics(p, Xs[:, te], y[te])
    push!(R, (; run="shuffled_labels", seed=s, agree=sum(glab .== truth) / length(groups),
              test_acc=ev.acc, test_auc=ev.auc))
  end
  acc = [r.test_acc for r in R]
  @printf("\n   label-shuffle control, 50 permutations of the training chains' labels:\n")
  @printf("   test acc %.3f +- %.3f (min %.3f, max %.3f)   AUC %.3f   corr(acc, label agreement) %.2f\n",
          mean(acc), std(acc), minimum(acc), maximum(acc), mean(r.test_auc for r in R),
          cor(acc, [r.agree for r in R]))
  write_rows(joinpath(OUT, "ffnn_shuffle50.csv"), R)

  write_rows(joinpath(OUT, "ffnn_summary.csv"), S)
  write_rows(joinpath(OUT, "ffnn_history.csv"), H)
  write_rows(joinpath(OUT, "ffnn_predictions.csv"), P)
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(isempty(ARGS) ? ("main", "transfer", "measure") : Tuple(ARGS))
end
