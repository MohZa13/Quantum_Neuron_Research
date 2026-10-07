# plots.jl — the figures, drawn from results/*.csv (run.jl calls this at the end).
#
#   julia 3_xx_xxx_grelu_minimal/plots.jl

using Plots, Statistics

const RES = joinpath(@__DIR__, "results")
const FIG = mkpath(joinpath(@__DIR__, "figures"))

"Read a CSV written by run.jl into named columns (numbers parsed where possible)."
function readcsv(name)
  lines = readlines(joinpath(RES, name))
  head = Symbol.(split(lines[1], ","))
  rows = [split(l, ",") for l in lines[2:end]]
  col(i) = (v = getindex.(rows, i); all(x -> tryparse(Float64, x) !== nothing, v) ? parse.(Float64, v) : String.(v))
  return NamedTuple{Tuple(head)}(Tuple(col(i) for i in eachindex(head)))
end

const LABEL = Dict("margin" => "Alg 9 (margin loss)", "square" => "Alg 8 (squared loss)")
const COLOR = Dict("margin" => :royalblue, "square" => :darkorange)
default(framestyle=:box, grid=false, dpi=200)

# Figure 1: training loss and test accuracy
h = readcsv("history.csv")
p1 = plot(xlabel="Adam step", ylabel="training loss", yscale=:log10, title="loss")
p2 = plot(xlabel="Adam step", ylabel="test accuracy", ylim=(0.4, 1.03), title="test accuracy", legend=:bottomright)
for k in ("margin", "square")
  s = h.model .== k
  plot!(p1, h.step[s], h.train_loss[s], lw=2, c=COLOR[k], label=LABEL[k])
  plot!(p2, h.step[s], h.test_accuracy[s], lw=2, c=COLOR[k], label=LABEL[k])
end
savefig(plot(p1, p2, layout=(1, 2), size=(900, 350), margin=5Plots.mm), joinpath(FIG, "1_training.png"))

# Figure 2: learned weights, averaged over each kind of term and scaled so the largest is 1
w = readcsv("weights.csv")
termkind(n) = n == "I" ? "I" : count(isletter, n) == 1 ? "Z" : string(n[1], n[1])   # "X3X4" -> "XX"
kinds = ["Z", "ZZ", "XX", "YY"]
p = plot(ylabel="mean weight (largest |.| = 1)", title="what the neuron learned", size=(600, 380),
         xticks=(1:4, kinds), ylim=(-1.12, 1.12), legend=:topleft)
for (off, k) in ((-0.18, "margin"), (0.18, "square"))
  s = w.model .== k
  m = [mean(w.theta[s][termkind.(w.term[s]) .== tk]) for tk in kinds]
  bar!(p, (1:4) .+ off, m ./ maximum(abs, m), bar_width=0.34, c=COLOR[k], label=LABEL[k])
end
hline!(p, [0], c=:black, lw=0.5, label="")
savefig(p, joinpath(FIG, "2_weights.png"))

# Figure 3: test scores by temperature (sign of the score = predicted label)
t = readcsv("test_scores.csv")
ps = map(("margin", "square")) do k
  s = t.model .== k
  q = plot(xscale=:log10, xlabel="kT", ylabel=k == "margin" ? "Tr[H ρ]" : "Tr[GReLU(H) ρ] − ½",
           title=LABEL[k] * "  (red XXX, grey XX)", titlefontsize=11, legend=false)
  for (y, c, lab) in ((-1, :gray40, "XX"), (1, :crimson, "XXX"))
    r = s .& (t.y .== y)
    scatter!(q, t.kT[r] .* exp.(0.04 .* randn(count(r))), t.score[r], ms=3, msw=0, alpha=0.7, c=c, label=lab)
  end
  hline!(q, [0], ls=:dash, c=:black, label="")
end
savefig(plot(ps..., layout=(1, 2), size=(900, 350), margin=5Plots.mm), joinpath(FIG, "3_test_scores.png"))

# Figure 4: classifying by firing (Algorithm 5)
f = readcsv("firing.csv")
p = plot(xscale=:log2, xlabel="copies of each test state", ylabel="test accuracy", ylim=(0.45, 1.03),
         title="classifying by firing the neuron (Algorithm 5)", size=(600, 380), legend=:bottomright)
plot!(p, f.copies, f.accuracy_alg9_model, lw=2, marker=:circle, c=COLOR["margin"], label=LABEL["margin"] * " model")
plot!(p, f.copies, f.accuracy_alg8_model, lw=2, marker=:circle, c=COLOR["square"], label=LABEL["square"] * " model")
hline!(p, [0.5], ls=:dash, c=:gray, label="chance")
savefig(p, joinpath(FIG, "4_firing.png"))

println("wrote figures/ in ", @__DIR__)
