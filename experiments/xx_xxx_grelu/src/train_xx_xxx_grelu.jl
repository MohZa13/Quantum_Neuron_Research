# train_xx_xxx_grelu.jl
#
# Train and test the quantized GReLU neuron (grelu_neuron.jl) on the XX vs XXX
# thermal states, with the GReLU versions of Algorithm 9 (margin loss) and
# Algorithm 8 (squared loss) as gradient oracles.  Writes CSVs only; plotting
# is a separate step so figures can be restyled without re-running.
#
#   julia experiments/xx_xxx_grelu/src/train_xx_xxx_grelu.jl split          # ~40 s
#   julia experiments/xx_xxx_grelu/src/train_xx_xxx_grelu.jl cv final       # exact gradients
#   julia experiments/xx_xxx_grelu/src/train_xx_xxx_grelu.jl shots gradvar readout kT classical
#
# Experiments (each writes experiments/xx_xxx_grelu/results/<name>*.csv):
#
#   split     stratified GROUP hold-out: 8 XX + 8 XXX chains (all 10 kT rungs
#             each) -> test; the other 68 chains get stratified group 5-fold
#             ids for model selection.  Also contracts every chain's chi64 MPO
#             at its coldest and hottest rung and checks it against ED.
#   cv        5-fold grouped CV on the training chains over (T, l2), exact
#             gradients, for both losses.  Picks by mean validation AUC, then loss.
#   final     retrain on all 68 training chains with the chosen (T, l2), exact
#             gradients (the infinite-shot limit of Alg 8/9); evaluate on test.
#   shots     the same, but every step's gradient comes from the Monte-Carlo
#             algorithm with single-shot measurements at several budgets.
#   gradvar   estimator quality: relative error / cosine to the exact gradient
#             vs number of circuit runs, at the initial and trained theta.
#   readout   finite-shot inference: test accuracy vs measurements per state.
#   kT        temperature transfer: train on hot rungs, test on cold, and back.
#   classical ablation: the same neuron with only Z / ZZ terms (a classical
#             Ising neuron in the paper's sense) -- does XX/YY matter?
#   fire      inference with the neuron itself: Gaussian Algorithm 5 firings
#             (one copy of rho per firing) instead of exact outputs.
#
# Every run starts from the same seeded initial theta.  Only the chosen
# hyperparameters, never the test set, drive any choice.

include(joinpath(@__DIR__, "grelu_neuron.jl"))
include(joinpath(@__DIR__, "xx_xxx_data.jl"))

using .GReLUNeuron, .XXXData
using Optimisers
using LinearAlgebra, Printf, Random, Statistics
const GN = GReLUNeuron

const EXP = normpath(joinpath(@__DIR__, ".."))            # experiments/xx_xxx_grelu
const ROOT = normpath(joinpath(EXP, "..", ".."))           # repository root
const DATA = joinpath(ROOT, "data", "xx_xxx_thermal_states", "xx_xxx_n10.h5")
const OUT = joinpath(EXP, "results")
const SPLIT = joinpath(OUT, "split.csv")

# ------------------------------------------------------------ configuration ---

const CFG = (
  seed=20260923,
  init_scale=0.1,            # theta_0 ~ init_scale * N(0, 1)
  steps=200, lr=0.05,        # Adam
  # losses: which activation width T, whether H(theta) has an identity term.
  # Alg 9 decides by sign Tr[H rho]; with no bias the ZZ-vs-XX contrast is
  # scale free, so it keeps working as all correlators -> 0 at high kT.  Alg 8
  # regresses onto {0, 1} and needs the offset.
  bias=Dict(:margin => false, :square => true),
  cv_grid=Dict(:margin => [(T=1.0, l2=l2) for l2 in (1e-4, 1e-3, 1e-2, 1e-1)],
               :square => [(T=T, l2=l2) for T in (0.5, 1.0, 2.0) for l2 in (1e-4, 1e-3, 1e-2)]),
  cv_steps=150,
  shot_budgets=Dict(:margin => (16, 64, 256), :square => (64, 256)),
  gradvar_budgets=(8, 32, 128, 512), gradvar_reps=20,
  readout_shots=(1, 4, 16, 64, 256, 1024, 4096),
  fire_budgets=(1, 4, 16, 64, 256, 1024, 4096, 16384), fire_reps=10,
)

# ------------------------------------------------------------------ helpers ---

const LOSS_NAME = Dict(:margin => "Alg9 margin", :square => "Alg8 square")

terms_for(ds, kind; ansatz=:quantum, bias=CFG.bias[kind]) =
  ansatz === :quantum ? model_terms(ds.n; bias) :
  filter(t -> t[1] in (:id, :z, :zz), model_terms(ds.n; bias))

subproblem(ds, idx, paulis, T) =
  GN.Problem(ds.B, paulis, T, ds.fams, ds.fam[idx], ds.beta[idx], ds.y[idx])

targets(prob) = (prob.y .+ 1) ./ 2

"Mann-Whitney AUC of `score` for y = +1 vs y = -1 (ties count 1/2)."
function auc(score, y)
  pos, neg = score[y .> 0], score[y .< 0]
  return mean((p > q) + 0.5 * (p == q) for p in pos, q in neg)
end

function loss_grad(kind, prob, theta)
  kind === :margin && return margin_loss_grad(prob, theta)
  L, g, _ = square_loss_grad(prob, theta, targets(prob))
  return L, g
end

"Scores (sign = predicted label), neuron outputs and loss on one set."
function evaluate(kind, prob, theta)
  out = neuron_outputs(prob, theta)
  if kind === :margin
    score = margin_scores(prob, theta)
    L = margin_loss_grad(prob, theta)[1]
  else
    score = out .- 0.5
    L = mean(abs2, out .- targets(prob))
  end
  pred = ifelse.(score .>= 0, 1.0, -1.0)
  return (; loss=L, acc=mean(pred .== prob.y), auc=auc(score, prob.y), score, out, pred)
end

function mc_grad(kind, prob, theta, nsamples, rng)
  kind === :margin && return alg9_grelu_gradient(prob, theta; nsamples, rng)
  return alg8_grelu_gradient(prob, theta, targets(prob); nsamples, rng)
end

"""
    train(kind, ptr, pte; T, l2, steps, lr, nsamples=0, seed)

Adam on L + l2/2 |theta|^2.  `nsamples = 0` uses the exact gradient; otherwise
each step's gradient is the single-shot Monte-Carlo algorithm with that many
circuit runs.  The exact gradient is still computed every step (cheap) so the
estimate's error can be logged.  `pte` may be `nothing` (CV inner fits).
"""
function train(kind, ptr, pte; l2, steps=CFG.steps, lr=CFG.lr, nsamples::Int=0,
               seed=CFG.seed, log_test=true, monitor=true, verbose=true)
  J = length(ptr.paulis)
  theta = CFG.init_scale .* randn(MersenneTwister(seed), J)
  rng = MersenneTwister(seed + 1)
  opt = Optimisers.setup(Optimisers.Adam(lr), theta)
  hist = NamedTuple[]
  t0 = time()
  for step in 0:steps
    ts = time()
    L, gex = loss_grad(kind, ptr, theta)
    g = nsamples == 0 ? gex : mc_grad(kind, ptr, theta, nsamples, rng)
    gt, gext = g .+ l2 .* theta, gex .+ l2 .* theta
    tgrad = time() - ts
    nomon = (; loss=NaN, acc=NaN, auc=NaN)
    tr_ev = monitor || step == steps ? evaluate(kind, ptr, theta) : nomon
    te_ev = (pte === nothing || !log_test) ? nothing : evaluate(kind, pte, theta)
    push!(hist, (; step, train_obj=L + l2 / 2 * sum(abs2, theta), train_loss=L,
                 train_acc=tr_ev.acc, train_auc=tr_ev.auc,
                 test_loss=te_ev === nothing ? NaN : te_ev.loss,
                 test_acc=te_ev === nothing ? NaN : te_ev.acc,
                 test_auc=te_ev === nothing ? NaN : te_ev.auc,
                 gnorm=norm(gext), grad_relerr=norm(gt - gext) / norm(gext),
                 grad_cos=dot(gt, gext) / (norm(gt) * norm(gext)),
                 theta_l1=norm(theta, 1), seconds=tgrad, theta=copy(theta)))
    if verbose && (step % 25 == 0 || step == steps)
      @printf("   %4d  obj %.5f  train acc %.3f auc %.3f  test acc %s  |g| %.2e  %.2fs/step\n",
              step, hist[end].train_obj, tr_ev.acc, tr_ev.auc,
              te_ev === nothing ? "  -  " : @sprintf("%.3f", te_ev.acc), norm(gext), tgrad)
    end
    step == steps && break
    opt, theta = Optimisers.update!(opt, theta, gt)
  end
  return (; theta, hist, wall=time() - t0)
end

# -------------------------------------------------------------------- CSVs ---

csvval(x::AbstractString) = x
csvval(x::Integer) = string(x)
csvval(x::Real) = isnan(x) ? "" : @sprintf("%.8g", x)
csvval(x::Symbol) = String(x)

function write_rows(path, rows)
  isempty(rows) && return
  mkpath(dirname(path))
  # union of columns in first-seen order: runs with and without a bias term
  # share a file, and a row lacking a column gets an empty cell
  ks = unique(reduce(vcat, [collect(keys(r)) for r in rows]))
  open(path, "w") do io
    println(io, join(ks, ","))
    for r in rows
      println(io, join((haskey(r, k) ? csvval(r[k]) : "" for k in ks), ","))
    end
  end
  println("   wrote ", relpath(path, ROOT))
end

"History rows, with theta expanded into one column per term."
function hist_rows(hist, paulis; extra...)
  labels = [GN.term_label(P.term) for P in paulis]
  map(hist) do h
    base = (; extra..., (k => getfield(h, k) for k in keys(h) if k != :theta)...)
    merge(base, NamedTuple{Tuple(Symbol.("theta_" .* labels))}(Tuple(h.theta)))
  end
end

function prediction_rows(ds, idx, ev; extra...)
  [(; extra..., sample=ds.names[m], model=ds.model[m], draw=ds.draw[m], group=ds.group[m],
     kT=ds.kT[m], y=Int(ds.y[m]), score=ev.score[i], neuron_output=ev.out[i],
     pred=Int(ev.pred[i]), correct=Int(ev.pred[i] == ds.y[m])) for (i, m) in enumerate(idx)]
end

# -------------------------------------------------------------- experiments ---

function exp_split(ds)
  split, fold = XXXData.make_split(ds; seed=CFG.seed)
  XXXData.write_split(SPLIT, ds, split, fold)
  println("split -> ", relpath(SPLIT, ROOT))
  XXXData.describe_split(ds, split, fold)
  # MPO vs ED at each chain's coldest and hottest rung
  which = [m for m in 1:length(ds) if ds.kT[m] in (minimum(ds.kT), maximum(ds.kT))]
  res = XXXData.check_mpo(ds, which)
  write_rows(joinpath(OUT, "mpo_check.csv"),
             [(; sample=r.name, group=r.group, kT=r.kT, trace_distance=r.td, min_eig=r.mineig)
              for r in res])
  # bond-averaged correlators <ZZ>, <XX>, <YY> of every state: what H(theta) reads
  terms = model_terms(ds.n)
  pall = subproblem(ds, 1:length(ds), [Pauli(ds.B, t) for t in terms], 1.0)
  avg(kind) = vec(mean(pall.C[:, [t[1] === kind for t in terms]]; dims=2))
  zz, xx, yy, z = avg(:zz), avg(:xx), avg(:yy), avg(:z)
  write_rows(joinpath(OUT, "correlators.csv"),
             [(; sample=ds.names[m], model=ds.model[m], group=ds.group[m], kT=ds.kT[m],
                split=split[m], zz=zz[m], xx=xx[m], yy=yy[m], z=z[m]) for m in 1:length(ds)])
  return split, fold
end

load_split(ds) = isfile(SPLIT) ? XXXData.read_split(SPLIT, ds) : exp_split(ds)

function exp_cv(ds, split, fold; kinds=(:margin, :square))
  rows = NamedTuple[]
  best = Dict{Symbol,Any}()
  for kind in kinds
    println("\nCV  ", LOSS_NAME[kind])
    for hp in CFG.cv_grid[kind], k in sort(unique(fold[fold .> 0]))
      tr = findall(f -> f > 0 && f != k, fold)
      va = findall(==(k), fold)
      paulis = [Pauli(ds.B, t) for t in terms_for(ds, kind)]
      ptr, pva = subproblem(ds, tr, paulis, hp.T), subproblem(ds, va, paulis, hp.T)
      res = train(kind, ptr, nothing; l2=hp.l2, steps=CFG.cv_steps, monitor=false, verbose=false)
      ev = evaluate(kind, pva, res.theta)
      push!(rows, (; loss_type=kind, T=hp.T, l2=hp.l2, fold=k, val_loss=ev.loss,
                   val_acc=ev.acc, val_auc=ev.auc, train_acc=res.hist[end].train_acc,
                   seconds=res.wall))
      @printf("   T=%-4g l2=%-6g fold %d: val acc %.3f auc %.3f  (%.0fs)\n",
              hp.T, hp.l2, k, ev.acc, ev.auc, res.wall)
    end
    agg = map(CFG.cv_grid[kind]) do hp
      r = filter(x -> x.loss_type == kind && x.T == hp.T && x.l2 == hp.l2, rows)
      (; hp..., auc=mean(x.val_auc for x in r), loss=mean(x.val_loss for x in r))
    end
    b = sort(agg; by=a -> (-a.auc, a.loss))[1]
    best[kind] = (T=b.T, l2=b.l2)
    @printf("   chosen: T = %g, l2 = %g  (mean val AUC %.4f)\n", b.T, b.l2, b.auc)
  end
  write_rows(joinpath(OUT, "cv.csv"), rows)
  write_rows(joinpath(OUT, "cv_choice.csv"),
             [(; loss_type=k, T=v.T, l2=v.l2) for (k, v) in best])
  return best
end

"Chosen hyperparameters from cv_choice.csv, or the defaults if CV has not run."
function chosen(kind)
  path = joinpath(OUT, "cv_choice.csv")
  if isfile(path)
    for l in readlines(path)[2:end]
      c = Base.split(l, ',')
      c[1] == String(kind) && return (T=parse(Float64, c[2]), l2=parse(Float64, c[3]))
    end
  end
  @warn "no CV choice for $kind -- using T = 1, l2 = 1e-3"
  return (T=1.0, l2=1e-3)
end

function problems(ds, split, kind; ansatz=:quantum, bias=CFG.bias[kind], T=chosen(kind).T,
                  trmask=trues(length(ds)), temask=trues(length(ds)))
  paulis = [Pauli(ds.B, t) for t in terms_for(ds, kind; ansatz, bias)]
  tr = findall(m -> split[m] == "train" && trmask[m], 1:length(ds))
  te = findall(m -> split[m] == "test" && temask[m], 1:length(ds))
  return subproblem(ds, tr, paulis, T), subproblem(ds, te, paulis, T), tr, te, paulis
end

function run_and_save(ds, split, kind, tag; ansatz=:quantum, nsamples=0, kw...)
  hp = chosen(kind)
  ptr, pte, tr, te, paulis = problems(ds, split, kind; ansatz, kw...)
  @printf("\n%s  [%s]  T=%g l2=%g  %d train / %d test states  J=%d  %s\n",
          LOSS_NAME[kind], tag, hp.T, hp.l2, length(tr), length(te), length(paulis),
          nsamples == 0 ? "exact gradient" : "$nsamples circuit runs/step")
  res = train(kind, ptr, pte; l2=hp.l2, nsamples)
  meta = (; loss_type=kind, run=tag, ansatz, bias=Int(get(kw, :bias, CFG.bias[kind])),
          nsamples, T=hp.T, l2=hp.l2)
  hrows = hist_rows(res.hist, paulis; meta...)
  prows = vcat(prediction_rows(ds, tr, evaluate(kind, ptr, res.theta); meta..., split="train"),
               prediction_rows(ds, te, evaluate(kind, pte, res.theta); meta..., split="test"))
  return hrows, prows, res
end

function exp_final(ds, split)
  H, P = NamedTuple[], NamedTuple[]
  for kind in (:margin, :square)
    h, p, _ = run_and_save(ds, split, kind, "final")
    append!(H, h); append!(P, p)
  end
  write_rows(joinpath(OUT, "final_history.csv"), H)
  write_rows(joinpath(OUT, "final_predictions.csv"), P)
end

function exp_shots(ds, split)
  for kind in (:margin, :square), N in CFG.shot_budgets[kind]
    h, p, _ = run_and_save(ds, split, kind, "shots$N"; nsamples=N)
    write_rows(joinpath(OUT, "shots_$(kind)_N$(N)_history.csv"), h)
    write_rows(joinpath(OUT, "shots_$(kind)_N$(N)_predictions.csv"), p)
  end
end

function exp_classical(ds, split)
  # Z/ZZ only.  The Alg 9 neuron has no bias by design, but without XX/YY terms
  # its score sum theta_ZZ <ZZ> has one sign for every state (all <ZZ> < 0), so
  # it can rank but never threshold.  "classical_bias" adds the identity term:
  # that is the fair classical comparison; "classical" is the matched ablation.
  H, P = NamedTuple[], NamedTuple[]
  for (kind, tag, bias) in ((:margin, "classical", false), (:margin, "classical_bias", true),
                            (:square, "classical", true))
    h, p, _ = run_and_save(ds, split, kind, tag; ansatz=:classical, bias)
    append!(H, h); append!(P, p)
  end
  write_rows(joinpath(OUT, "classical_history.csv"), H)
  write_rows(joinpath(OUT, "classical_predictions.csv"), P)
end

"Train on hot (kT >= 0.5) rungs, test on cold (kT < 0.5), and the reverse."
function exp_kT(ds, split)
  hot = ds.kT .>= 0.5
  H, P = NamedTuple[], NamedTuple[]
  for kind in (:margin, :square), (tag, trm, tem) in (("hot_to_cold", hot, .!hot),
                                                       ("cold_to_hot", .!hot, hot))
    h, p, _ = run_and_save(ds, split, kind, tag; trmask=trm, temask=tem)
    append!(H, h); append!(P, p)
  end
  write_rows(joinpath(OUT, "kT_transfer_history.csv"), H)
  write_rows(joinpath(OUT, "kT_transfer_predictions.csv"), P)
end

"Gradient-estimator error vs budget at theta_0 and at the exact-trained theta."
function exp_gradvar(ds, split)
  rows = NamedTuple[]
  for kind in (:margin, :square)
    hp = chosen(kind)
    ptr, pte, _, _, _ = problems(ds, split, kind)
    thetas = [("init", CFG.init_scale .* randn(MersenneTwister(CFG.seed), length(ptr.paulis))),
              ("trained", train(kind, ptr, nothing; l2=hp.l2, monitor=false, verbose=false).theta)]
    for (where, th) in thetas
      _, gex = loss_grad(kind, ptr, th)
      for N in CFG.gradvar_budgets, sampled in (true, false), r in 1:CFG.gradvar_reps
        rng = MersenneTwister(1000r + N)
        t = @elapsed g = kind === :margin ?
            alg9_grelu_gradient(ptr, th; nsamples=N, rng, sampled) :
            alg8_grelu_gradient(ptr, th, targets(ptr); nsamples=N, rng, sampled)
        push!(rows, (; loss_type=kind, theta_at=where, nsamples=N, sampled=Int(sampled),
                     rep=r, relerr=norm(g - gex) / norm(gex), abserr=norm(g - gex),
                     gnorm=norm(gex),
                     cos=dot(g, gex) / (norm(g) * norm(gex)), theta_l1=norm(th, 1), seconds=t))
      end
      @printf("   %s %-7s done\n", LOSS_NAME[kind], where)
    end
  end
  write_rows(joinpath(OUT, "gradvar.csv"), rows)
end

"""
Finite-shot inference on the test set with the exact-trained theta.
Alg 9 model: score Tr[H rho] estimated by sampling k ~ q and measuring H_k
(one Pauli shot each: ||theta||_1 sgn(theta_k) x).  Alg 8 model: the neuron
output via the value circuit (`value_estimate`).
"""
function exp_readout(ds, split)
  rows = NamedTuple[]
  for kind in (:margin, :square)
    hp = chosen(kind)
    ptr, pte, _, te, _ = problems(ds, split, kind)
    th = train(kind, ptr, nothing; l2=hp.l2, monitor=false, verbose=false).theta
    eig = GN.diagonalise(GN.hamiltonian(pte.B, pte.paulis, th))
    cq = cumsum(abs.(th) ./ norm(th, 1))
    for S in CFG.readout_shots, r in 1:10
      rng = MersenneTwister(97r + S)
      score = map(1:GN.nstates(pte)) do m
        if kind === :margin
          mean(begin
                 k = GN.sampleq(rng, cq)
                 norm(th, 1) * sign(th[k]) * GN.shot(rng, pte.C[m, k], true)
               end for _ in 1:S)
        else
          value_estimate(pte, th, m; nshots=S, rng, eig)[1] - 0.5
        end
      end
      pred = ifelse.(score .>= 0, 1.0, -1.0)
      for (i, m) in enumerate(te)
        push!(rows, (; loss_type=kind, shots=S, rep=r, sample=ds.names[m], kT=ds.kT[m],
                     y=Int(ds.y[m]), score=score[i], correct=Int(pred[i] == pte.y[i])))
      end
    end
    @printf("   %s readout done\n", LOSS_NAME[kind])
  end
  write_rows(joinpath(OUT, "readout.csv"), rows)
end

"""
Inference with the neuron itself: Gaussian Algorithm 5 (`fire`), one copy of
rho_m per firing, on the exact-trained theta.
  Alg 8 model: N firings of GReLU(H); predict XXX if their mean > 1/2.
  Alg 9 model: N firings each of GReLU(H) and GReLU(-H); predict XXX if
               mean(H) - mean(-H) >= 0  (that difference estimates Tr[H rho]).
`total_firings` counts copies of rho used: N for Alg 8, 2N for Alg 9.
"""
function exp_fire(ds, split)
  rows = NamedTuple[]
  for kind in (:margin, :square)
    hp = chosen(kind)
    ptr, pte, _, te, _ = problems(ds, split, kind)
    th = train(kind, ptr, nothing; l2=hp.l2, monitor=false, verbose=false).theta
    eig = GN.diagonalise(GN.hamiltonian(pte.B, pte.paulis, th))
    samplers = [FiringSampler(pte, eig, m) for m in 1:GN.nstates(pte)]
    for N in CFG.fire_budgets, r in 1:CFG.fire_reps
      rng = MersenneTwister(7919r + N)
      for (i, m) in enumerate(te)
        fs = samplers[i]
        score = kind === :margin ?
                mean(fire(fs, rng) for _ in 1:N) - mean(fire(fs, rng; sign=-1) for _ in 1:N) :
                mean(fire(fs, rng) for _ in 1:N) - 0.5
        pred = score >= 0 ? 1.0 : -1.0
        push!(rows, (; loss_type=kind, firings_per_neuron=N,
                     total_firings=kind === :margin ? 2N : N, rep=r, sample=ds.names[m],
                     model=ds.model[m], group=ds.group[m], kT=ds.kT[m], y=Int(ds.y[m]),
                     score, correct=Int(pred == ds.y[m])))
      end
    end
    for N in CFG.fire_budgets
      acc = mean(r.correct for r in rows if r.loss_type == kind && r.firings_per_neuron == N)
      @printf("   %s  N = %5d firings/neuron: test accuracy %.3f\n", LOSS_NAME[kind], N, acc)
    end
  end
  write_rows(joinpath(OUT, "fire.csv"), rows)
end

# ---------------------------------------------------------------------- main ---

if abspath(PROGRAM_FILE) == @__FILE__
  isempty(ARGS) && (println("usage: julia train_xx_xxx_grelu.jl split|cv|final|shots|gradvar|readout|kT|classical|fire ..."); exit())
  ds = XXXData.load(DATA)
  split, fold = "split" in ARGS ? exp_split(ds) : load_split(ds)
  "cv" in ARGS && exp_cv(ds, split, fold)
  "final" in ARGS && exp_final(ds, split)
  "classical" in ARGS && exp_classical(ds, split)
  "kT" in ARGS && exp_kT(ds, split)
  "gradvar" in ARGS && exp_gradvar(ds, split)
  "readout" in ARGS && exp_readout(ds, split)
  "shots" in ARGS && exp_shots(ds, split)
  "fire" in ARGS && exp_fire(ds, split)
end
