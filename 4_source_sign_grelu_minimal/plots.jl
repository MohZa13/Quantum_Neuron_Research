# plots.jl — the figures, drawn from results/*.csv (run.jl calls this at the end).
#
#   julia 4_source_sign_grelu_minimal/plots.jl

if !@isdefined(HERE)
  const HERE = @__DIR__
end

using Plots, Statistics

const RES = joinpath(HERE, "results")
const FIG = mkpath(joinpath(HERE, "figures"))

"Read a CSV written above into named columns (numbers parsed where possible)."
function readcsv(name)
  lines = readlines(joinpath(RES, name))
  head = Symbol.(split(lines[1], ","))
  rows = [split(l, ",") for l in lines[2:end]]
  col(i) = (v = getindex.(rows, i); all(x -> tryparse(Float64, x) !== nothing, v) ? parse.(Float64, v) : String.(v))
  return NamedTuple{Tuple(head)}(Tuple(col(i) for i in eachindex(head)))
end

termkind(n) = n == "I" ? "I" : count(isletter, n) == 1 ? "Z" : string(n[1], n[1])   # "X24X25" -> "XX"
const KINDS = ["I", "Z", "ZZ", "XX", "YY"]
const KCOLOR = Dict("I" => :black, "Z" => :seagreen, "ZZ" => :crimson, "XX" => :royalblue, "YY" => :darkorange)
const SCOLOR = Dict(1.0 => :crimson, -1.0 => :royalblue)      # label colours: s = +1 red, s = −1 blue
default(framestyle=:box, grid=false, dpi=200)

ov = readcsv("overview.csv")
p = ov.s .== 1
p1 = plot(ov.t[p], ov.trace_distance[p], lw=2, c=:black, label="trace distance", ylim=(0, 1.02),
          xlabel="time t", ylabel="", title="distinguishability of the two labels", legend=:topright)
vspan!(p1, [0.1, 7.5], c=:gold, alpha=0.15, label="training window")
zcols = filter(n -> startswith(String(n), "z"), collect(keys(ov)))
p2 = plot(xlabel="time t", ylabel="⟨Z⟩ for s = +1", title="⟨Z⟩ on each qubit", legend=:topright)
for (i, zc) in enumerate(zcols)
  plot!(p2, ov.t[p], getproperty(ov, zc)[p], lw=2, c=get(cgrad(:viridis), (i - 1) / max(length(zcols) - 1, 1) * 0.85),
        label="site " * String(zc)[2:end])
end
hline!(p2, [0], c=:gray70, lw=0.5, label="")
vspan!(p2, [0.1, 7.5], c=:gold, alpha=0.15, label="")
fig = plot(p1, p2, layout=(1, 2), size=(1000, 360), margin=5Plots.mm)
savefig(fig, joinpath(FIG, "0_data.png"))
fig

h = readcsv("history.csv")
p1 = plot(h.step, h.train_loss, lw=2, c=:darkorange, yscale=:log10, xlabel="Adam step",
          ylabel="training loss", title="squared loss", legend=false)
p2 = plot(h.step, h.train_accuracy, lw=2, c=:darkorange, label="train", ylim=(0.4, 1.03),
          xlabel="Adam step", ylabel="accuracy (exact outputs)", title="accuracy", legend=:bottomright)
plot!(p2, h.step, h.test_accuracy, lw=2, c=:darkorange, ls=:dash, label="test")
fig = plot(p1, p2, layout=(1, 2), size=(900, 350), margin=5Plots.mm)
savefig(fig, joinpath(FIG, "1_training.png"))
fig

tr_ = readcsv("theta_trajectory.csv")
steps = sort(unique(tr_.step))
termnames = tr_.term[tr_.step .== 0]
traj(n) = tr_.theta[tr_.term .== n]               # rows are written in step order
ps = map(KINDS) do k
  ns = filter(n -> termkind(n) == k, termnames)
  q = plot(xlabel="Adam step", ylabel="θ", title=k == "I" ? "bias (identity)" : "$k terms", legend=:best,
           legendfontsize=7)
  hline!(q, [0], c=:gray70, lw=0.5, label="")
  for (i, n) in enumerate(ns)
    plot!(q, steps, traj(n), lw=1.8, label=n,
          c=k == "I" ? :black : get(cgrad(:viridis), (i - 1) / max(length(ns) - 1, 1) * 0.85))
  end
  q
end
l1 = [sum(abs, tr_.theta[tr_.step .== st]) for st in steps]
push!(ps, plot(steps, l1, lw=2, c=:black, xlabel="Adam step", ylabel="‖θ‖₁", title="‖θ‖₁", legend=false))
fig = plot(ps..., layout=(2, 3), size=(1050, 600), margin=4Plots.mm)
savefig(fig, joinpath(FIG, "2_theta_trajectories.png"))
fig

w = readcsv("weights.csv")
lim = 1.15 * maximum(abs, w.theta)
p = plot(ylabel="θ", title="trained weights", size=(700, 380), legend=:outerright, ylim=(-lim, lim),
         xticks=(1:length(w.term), w.term), xrotation=45, bottom_margin=6Plots.mm, left_margin=4Plots.mm)
for k in KINDS
  sel = termkind.(w.term) .== k
  bar!(p, findall(sel), w.theta[sel], bar_width=0.8, c=KCOLOR[k], lc=KCOLOR[k], label=k)
end
hline!(p, [0], c=:black, lw=0.5, label="")
savefig(p, joinpath(FIG, "3_weights.png"))
p

o = readcsv("outputs.csv")
p = plot(xlabel="time t", ylabel="output − ½", title="neuron output (red s = +1, blue s = −1)",
         size=(750, 380), legend=:topright)
for (sv, lab) in ((1.0, "s = +1"), (-1.0, "s = −1"))
  r1 = (o.s .== sv) .& (o.set .== "train")
  r2 = (o.s .== sv) .& (o.set .== "test")
  scatter!(p, o.t[r1], o.output[r1] .- 0.5, ms=3, msw=0, c=SCOLOR[sv], label=lab * " train")
  scatter!(p, o.t[r2], o.output[r2] .- 0.5, ms=5, mc=:white, msc=SCOLOR[sv], msw=1.5, label=lab * " test")
end
hline!(p, [0], ls=:dash, c=:black, label="")
savefig(p, joinpath(FIG, "4_outputs.png"))
p

f = readcsv("firing.csv")
p = plot(f.copies, f.accuracy, lw=2, marker=:circle, c=:darkorange, xscale=:log2, label="fired neuron",
         xlabel="copies of each test state", ylabel="test accuracy", ylim=(0.45, 1.03),
         title="classifying by firing the neuron (Algorithm 5)", size=(600, 380), legend=:bottomright)
hline!(p, [0.5], ls=:dash, c=:gray, label="chance")
savefig(p, joinpath(FIG, "5_firing.png"))
p
