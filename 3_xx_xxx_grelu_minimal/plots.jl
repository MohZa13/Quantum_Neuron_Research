# plots.jl — the figures, drawn from results/*.csv (run.jl calls this at the end).
#
#   julia 3_xx_xxx_grelu_minimal/plots.jl

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

termkind(n) = n == "I" ? "I" : count(isletter, n) == 1 ? "Z" : string(n[1], n[1])   # "X3X4" -> "XX"
const KINDS = ["I", "Z", "ZZ", "XX", "YY"]
const KCOLOR = Dict("I" => :black, "Z" => :seagreen, "ZZ" => :crimson, "XX" => :royalblue, "YY" => :darkorange)
default(framestyle=:box, grid=false, dpi=200)

h = readcsv("history.csv")
p1 = plot(h.step, h.train_loss, lw=2, c=:darkorange, yscale=:log10, xlabel="Adam step",
          ylabel="training loss", title="squared loss", legend=false)
p2 = plot(h.step, h.test_accuracy, lw=2, c=:darkorange, ylim=(0.4, 1.03), xlabel="Adam step",
          ylabel="test accuracy", title="test accuracy (exact outputs)", legend=false)
fig = plot(p1, p2, layout=(1, 2), size=(900, 350), margin=5Plots.mm)
savefig(fig, joinpath(FIG, "1_training.png"))
fig

tr = readcsv("theta_trajectory.csv")
steps = sort(unique(tr.step))
traj(n) = tr.theta[tr.term .== n]                 # rows are written in step order
termnames = tr.term[tr.step .== 0]
ps = map(KINDS) do k
  ns = filter(n -> termkind(n) == k, termnames)
  q = plot(xlabel="Adam step", ylabel="θ", title=k == "I" ? "bias (identity)" : "$k terms", legend=false)
  hline!(q, [0], c=:gray70, lw=0.5)
  for (i, n) in enumerate(ns)
    plot!(q, steps, traj(n), lw=1.5, c=k == "I" ? :black : get(cgrad(:viridis), (i - 1) / max(length(ns) - 1, 1) * 0.85))
  end
  q
end
l1 = [sum(abs, tr.theta[tr.step .== s]) for s in steps]
push!(ps, plot(steps, l1, lw=2, c=:black, xlabel="Adam step", ylabel="‖θ‖₁", title="‖θ‖₁", legend=false))
fig = plot(ps..., layout=(2, 3), size=(1050, 600), margin=4Plots.mm)
savefig(fig, joinpath(FIG, "2_theta_trajectories.png"))
fig

Th = reduce(hcat, [tr.theta[tr.step .== s] for s in steps])
G = reduce(hcat, [tr.grad[tr.step .== s] for s in steps])
slog(x) = sign(x) * log10(1 + abs(x) / 1e-4)       # symmetric log, linear below 1e-4
yt = (1:length(termnames), termnames)
c1 = maximum(abs, Th)
p1 = heatmap(steps, 1:length(termnames), Th, c=:RdBu, clims=(-c1, c1), yflip=true, yticks=yt,
             ytickfontsize=5, xlabel="Adam step", title="θⱼ")
c2 = maximum(abs, slog.(G))
p2 = heatmap(steps, 1:length(termnames), slog.(G), c=:PuOr, clims=(-c2, c2), yflip=true, yticks=yt,
             ytickfontsize=5, xlabel="Adam step", title="∂L/∂θⱼ  (sign · log₁₀(1 + |g|/10⁻⁴))")
fig = plot(p1, p2, layout=(1, 2), size=(1050, 520), margin=4Plots.mm)
savefig(fig, joinpath(FIG, "3_theta_heatmap.png"))
fig

w = readcsv("weights.csv")
lim = 1.1 * maximum(abs, w.theta)
p = plot(ylabel="θ", title="trained weights", size=(900, 400), legend=:bottomright, ylim=(-lim, lim),
         left_margin=5Plots.mm,
         xticks=(1:length(w.term), w.term), xrotation=90, xtickfontsize=6, bottom_margin=6Plots.mm)
for k in KINDS
  sel = termkind.(w.term) .== k
  bar!(p, findall(sel), w.theta[sel], bar_width=0.8, c=KCOLOR[k], lc=KCOLOR[k], label=k)
end
hline!(p, [0], c=:black, lw=0.5, label="")
savefig(p, joinpath(FIG, "4_weights.png"))
p

cv, cg, cc = readcsv("circuit_value.csv"), readcsv("circuit_gradient.csv"), readcsv("circuit_cost.csv")
cps = sort(unique(cv.step))
ccol(step) = get(cgrad(:viridis), 0.85 * (findfirst(==(step), cps) - 1) / max(length(cps) - 1, 1))
lab(step) = "step $(Int(step))"
p1 = plot(xscale=:log10, yscale=:log10, xlabel="value-block runs N", ylabel="|estimate − exact| / exact",
          title="value block, one state", legend=:bottomleft, legendfontsize=7)
p2 = plot(xscale=:log10, yscale=:log10, xlabel="Algorithm 8 runs N", ylabel="‖ĝ − g‖ / ‖g‖",
          title="full gradient", legend=false)
for step in cps
  rv, rg = cv.step .== step, cg.step .== step
  ev = abs.(cv.estimate[rv] .- cv.exact[rv]) ./ abs.(cv.exact[rv])
  plot!(p1, cv.runs[rv], ev, lw=2, marker=:circle, ms=3, c=ccol(step), label=lab(step))
  plot!(p1, cv.runs[rv], ev[1] .* sqrt.(cv.runs[rv][1] ./ cv.runs[rv]), ls=:dot, c=ccol(step), label="")
  plot!(p2, cg.runs[rg], cg.rel_error[rg], lw=2, marker=:circle, ms=3, c=ccol(step))
  plot!(p2, cg.runs[rg], cg.rel_error[rg][1] .* sqrt.(cg.runs[rg][1] ./ cg.runs[rg]), ls=:dot, c=ccol(step))
end
hline!(p2, [1], c=:gray, ls=:dash)
p3 = plot(cc.theta_l1, cc.runs_for_10pct, yscale=:log10, lw=1, c=:gray, marker=:circle, ms=5,
          mc=[ccol(st) for st in cc.step], msw=0, xlabel="‖θ‖₁", ylabel="runs for 10% gradient error",
          title="cost of the circuit gradient", legend=false,
          ylim=(minimum(cc.runs_for_10pct) / 10, maximum(cc.runs_for_10pct) * 100))
annotate!(p3, [(x, y * 2.5, text(lab(st), 7)) for (x, y, st) in zip(cc.theta_l1, cc.runs_for_10pct, cc.step)])
fig = plot(p1, p2, p3, layout=(1, 3), size=(1250, 410), margin=5Plots.mm, bottom_margin=8Plots.mm)
savefig(fig, joinpath(FIG, "5_circuit_check.png"))
fig

t = readcsv("test_scores.csv")
p = plot(xscale=:log10, xlabel="kT", ylabel="Tr[GReLU(H) ρ] − ½", size=(600, 380),
         title="test states (red XXX, grey XX)", legend=:right)
for (y, c, lab) in ((-1, :gray40, "XX"), (1, :crimson, "XXX"))
  r = t.y .== y
  scatter!(p, t.kT[r] .* exp.(0.04 .* randn(count(r))), t.score[r], ms=3, msw=0, alpha=0.7, c=c, label=lab)
end
hline!(p, [0], ls=:dash, c=:black, label="")
savefig(p, joinpath(FIG, "6_test_scores.png"))
p

f = readcsv("firing.csv")
p = plot(f.copies, f.accuracy, lw=2, marker=:circle, c=:darkorange, xscale=:log2, label="fired neuron",
         xlabel="copies of each test state", ylabel="test accuracy", ylim=(0.45, 1.03),
         title="classifying by firing the neuron (Algorithm 5)", size=(600, 380), legend=:bottomright)
hline!(p, [0.5], ls=:dash, c=:gray, label="chance")
savefig(p, joinpath(FIG, "7_firing.png"))
p
