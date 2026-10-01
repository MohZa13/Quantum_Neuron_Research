# plot_xx_xxx_grelu.jl
#
# The six group-meeting figures from the CSVs written by train_xx_xxx_grelu.jl
# (see ../REPORT.md and ../plot_menu.md).  Re-renders without re-running training;
# each figure is skipped if its CSVs are not there yet.
#
#   julia experiments/xx_xxx_grelu/src/plot_xx_xxx_grelu.jl
#
#   A2  why the classes separate   (correlators.csv)
#   B1  training curves            (final_history.csv)
#   C   test scores by class / kT  (final_predictions.csv)
#   D1  learned H(theta)           (final_history.csv, last step)
#   E   gradient cost + shots      (gradvar.csv [+ gradvar_gnorm.csv], shots_*_history.csv, final_history.csv)
#   F1  temperature transfer       (kT_transfer_predictions.csv, final_predictions.csv)
#   G   classical vs quantum ansatz (classical_predictions.csv, final_predictions.csv)
#   H   readout cost at inference  (readout.csv)
#   G2  classical-ansatz errors by state: one row per test chain, kT across (classical_predictions.csv)
#   I   inference by firing the neuron (Gaussian Alg 5) vs readout circuits (fire.csv, readout.csv)
#   J   test-state scatter at fixed firing budgets + the exact limit (fire.csv, final_predictions.csv)
#   I2  Algorithm 5 with an explicit ITensor qumode vs the exact sampler, n = 4 fixture
#       (qumode_itensor_validation.csv, qumode_itensor_example.csv from qumode_itensor.jl)
#   K   quantum neuron vs feed-forward network: temperature transfer and accuracy vs copies of rho
#   L   correct / incorrect counts per class for the practical classifiers at 3 budgets
#       (fire.csv, ffnn_measure.csv)
#       (kT_transfer_predictions.csv, ffnn_kT_transfer_predictions.csv, fire.csv, ffnn_measure.csv)
#
# Output: experiments/xx_xxx_grelu/figures/<name>.png and .pdf

using Plots, Printf, Statistics

const RES = joinpath(@__DIR__, "..", "results")
const FIG = joinpath(@__DIR__, "..", "figures")

# ------------------------------------------------------------------ palette ---
# Validated (dataviz validate_palette.js, light surface): classes blue/orange and
# algorithms violet/aqua pass all-pairs; term types pass adjacent.  Low-contrast
# slots are always paired with a legend and direct labels.
const INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
const CLASS = Dict("XX" => "#2a78d6", "XXX" => "#eb6834")
const ALG = Dict("margin" => "#4a3aa7", "square" => "#1baf7a")
const ALGNAME = Dict("margin" => "Alg 9 · margin loss", "square" => "Alg 8 · squared loss")
const TERM = Dict("ZZ" => "#1baf7a", "XX" => "#4a3aa7", "YY" => "#e87ba4", "Z" => "#eda100")
const RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # ordinal blue: shot budgets
const FFNN = "#e87ba4"          # feed-forward network; CVD 6.1 vs aqua -> always square markers + dash-dot
const CLASSICAL = "#eda100"     # classical (Z/ZZ-only) ansatz; validated against both ALG hues

default(; fontfamily="Helvetica", titlefontsize=11, guidefontsize=10, tickfontsize=8,
        legendfontsize=8, foreground_color_axis=MUTED, foreground_color_border=MUTED,
        foreground_color_text=INK, gridcolor=GRID, gridalpha=1, gridlinewidth=0.5,
        framestyle=:axes, background_color="#fcfcfb", legend_foreground_color=nothing,
        linewidth=2, dpi=200)

# ---------------------------------------------------------------------- csv ---

function readcsv(name)
  path = joinpath(RES, name)
  isfile(path) || return nothing
  lines = readlines(path)
  hdr = split(lines[1], ',')
  rows = [split(l, ',') for l in lines[2:end]]
  parse_col(v) = (x = tryparse(Float64, v); x === nothing ? (v == "" ? NaN : String(v)) : x)
  return Dict(String(h) => [parse_col(r[i]) for r in rows] for (i, h) in enumerate(hdr))
end

rowsel(d, mask) = Dict(k => v[mask] for (k, v) in d)
sel(d; kw...) = rowsel(d, reduce(.&, (d[String(k)] .== v for (k, v) in kw); init=trues(length(first(values(d))))))

function save(p, name)
  mkpath(FIG)
  savefig(p, joinpath(FIG, name * ".png"))
  savefig(p, joinpath(FIG, name * ".pdf"))
  println("wrote experiments/xx_xxx_grelu/figures/$name.{png,pdf}")
end

# ------------------------------------------------------ A2: correlators -----

function fig_a2()
  d = readcsv("correlators.csv")
  d === nothing && return
  p = plot(; xlabel="⟨(XX + YY)/2⟩  (bond average)", ylabel="⟨ZZ⟩  (bond average)",
           title="XXX sits on ⟨ZZ⟩ = ⟨XX⟩ (SU(2)); XX has weaker ⟨ZZ⟩ at every kT",
           legend=:topleft, size=(720, 560), aspect_ratio=:equal)
  lo = minimum(vcat(d["zz"], (d["xx"] .+ d["yy"]) ./ 2)) - 0.03
  plot!(p, [lo, 0], [lo, 0]; color=MUTED, ls=:dash, lw=1, label="⟨ZZ⟩ = ⟨XX⟩")
  for model in ("XX", "XXX")
    first_ = true
    for g in unique(d["group"][d["model"] .== model])
      c = sel(d; group=g)
      o = sortperm(c["kT"])
      x, y = ((c["xx"] .+ c["yy"]) ./ 2)[o], c["zz"][o]
      plot!(p, x, y; color=CLASS[model], lw=0.8, alpha=0.45, label=false)
      scatter!(p, x, y; color=CLASS[model], ms=3, msw=0, alpha=0.8,
               label=first_ ? "$model  ($(length(unique(d["group"][d["model"] .== model]))) chains × 10 kT)" : false)
      first_ = false
    end
  end
  # mark the temperature direction on one XX chain
  c = sel(d; group=first(unique(d["group"][d["model"] .== "XX"])))
  for (kT, txt, al) in ((2.0, "kT = 2 (hot)", :left), (0.1, "kT = 0.1 (cold)", :right))
    i = findfirst(≈(kT), c["kT"])
    annotate!(p, (c["xx"][i] + c["yy"][i]) / 2 + (al === :left ? 0.012 : -0.012), c["zz"][i],
              text(txt, 8, MUTED, al))
  end
  save(p, "A2_correlators")
end

# ------------------------------------------------- B1: training curves ------

function fig_b1()
  h = readcsv("final_history.csv")
  h === nothing && return
  panels = []
  for kind in ("margin", "square")
    c = sel(h; loss_type=kind)
    pl = plot(; title=ALGNAME[kind], ylabel=kind == "margin" ? "loss" : "", legend=:topright,
              xlabel="")
    plot!(pl, c["step"], c["train_loss"]; color=ALG[kind], label="train")
    plot!(pl, c["step"], c["test_loss"]; color=ALG[kind], ls=:dash, label="test")
    pa = plot(; ylabel=kind == "margin" ? "accuracy" : "", xlabel="optimiser step (Adam)",
              ylim=(0.4, 1.02), legend=:right)
    hline!(pa, [0.5]; color=MUTED, ls=:dot, lw=1, label="chance")
    plot!(pa, c["step"], c["train_acc"]; color=ALG[kind], label="train")
    plot!(pa, c["step"], c["test_acc"]; color=ALG[kind], ls=:dash, label="test")
    fin = @sprintf("test acc %.3f · AUC %.3f", c["test_acc"][end], c["test_auc"][end])
    annotate!(pa, maximum(c["step"]) * 0.97, 0.47, text(fin, 8, INK, :right))
    push!(panels, (pl, pa))
  end
  p = plot(panels[1][1], panels[2][1], panels[1][2], panels[2][2]; layout=(2, 2),
           size=(1000, 620), left_margin=5Plots.mm, bottom_margin=4Plots.mm,
           plot_title="GReLU neuron, exact gradients (infinite-shot limit) — train 68 chains, test 16 held-out chains",
           plot_titlefontsize=11)
  save(p, "B1_training")
end

# ------------------------------------------------------ C: test scores ------

const SCORENAME = Dict("margin" => "decision score  Tr[H(θ) ρ]",
                       "square" => "decision score  Tr[GReLU(H) ρ] − ½")

function fig_c()
  d = readcsv("final_predictions.csv")
  d === nothing && return
  t = sel(d; split="test")
  panels = []
  for kind in ("margin", "square")
    c = sel(t; loss_type=kind)
    # C1: distribution by class
    ph = plot(; title=ALGNAME[kind], xlabel=SCORENAME[kind], ylabel="test states", legend=:top)
    edges = range(minimum(c["score"]), maximum(c["score"]); length=31)
    for model in ("XX", "XXX")
      histogram!(ph, c["score"][c["model"] .== model]; bins=edges, color=CLASS[model],
                 linecolor="#fcfcfb", lw=1, alpha=0.85, label=model)
    end
    vline!(ph, [0.0]; color=INK, lw=1, ls=:dash, label="threshold")
    # C2: score vs kT per chain
    pk = plot(; xlabel="kT (log scale)", ylabel=SCORENAME[kind], xscale=:log10,
              xticks=([0.1, 0.25, 0.5, 1.0, 2.0], ["0.1", "0.25", "0.5", "1", "2"]),
              legend=:right)
    hline!(pk, [0.0]; color=INK, lw=1, ls=:dash, label=false)
    for model in ("XX", "XXX")
      first_ = true
      for g in unique(c["group"][c["model"] .== model])
        cc = sel(c; group=g)
        o = sortperm(cc["kT"])
        plot!(pk, cc["kT"][o], cc["score"][o]; color=CLASS[model], lw=1.2, alpha=0.8,
              marker=:circle, ms=3, msw=0, label=first_ ? model : false)
        first_ = false
      end
    end
    nerr = count(==(0.0), c["correct"])
    annotate!(pk, 0.105, minimum(c["score"][c["model"] .== "XXX"]) / 2,
              text("$(nerr) / $(length(c["correct"])) test states misclassified", 8, INK, :left))
    push!(panels, (ph, pk))
  end
  p = plot(panels[1][1], panels[2][1], panels[1][2], panels[2][2]; layout=(2, 2),
           size=(1000, 680), left_margin=5Plots.mm, bottom_margin=4Plots.mm,
           plot_title="Held-out test chains: score distribution (top) and score across temperature (bottom)",
           plot_titlefontsize=11)
  save(p, "C_test_scores")
end

# --------------------------------------------------- D1: learned H(theta) ---

function fig_d1()
  h = readcsv("final_history.csv")
  h === nothing && return
  panels = []
  for kind in ("margin", "square")
    c = sel(h; loss_type=kind)
    last = findmax(c["step"])[2]
    th(name) = haskey(c, "theta_" * name) ? c["theta_" * name][last] : NaN
    n = count(k -> startswith(k, "theta_Z") && !occursin('Z', k[8:end]), keys(c))
    bonds = 1:(n - 1)
    kinds = [("ZZ", i -> "Z$(i)Z$(i+1)"), ("XX", i -> "X$(i)X$(i+1)"), ("YY", i -> "Y$(i)Y$(i+1)")]
    bias = haskey(c, "theta_I") && th("I") isa Real && !isnan(th("I")) ?
           @sprintf("   (identity / bias coefficient %.2f)", th("I")) : ""
    pb = plot(; title=ALGNAME[kind] * bias, xlabel="bond i (sites i, i+1)", ylabel="learned θ",
              xticks=bonds, legend=:outertopright)
    hline!(pb, [0.0]; color=MUTED, lw=1, label=false)
    w = 0.26
    for (k, (lab, nm)) in enumerate(kinds)
      bar!(pb, bonds .+ (k - 2) * w, [th(nm(i)) for i in bonds]; bar_width=w * 0.92,
           color=TERM[lab], linecolor="#fcfcfb", lw=0.5, label=lab * " bond terms")
    end
    scatter!(pb, (1:n) .- 0.5, [th("Z$i") for i in 1:n]; color=TERM["Z"], ms=5, msw=0,
             marker=:diamond, label="Z site terms (at i−½)")
    push!(panels, pb)
  end
  p = plot(panels...; layout=(2, 1), size=(900, 700), left_margin=5Plots.mm,
           plot_title="Learned Hamiltonian H(θ) = Σ θ · (Pauli term) along the 10-site chain",
           plot_titlefontsize=11)
  save(p, "D1_learned_theta")
end

# ------------------------------------------- E: gradient cost and shots -----

function fig_e()
  g, gn = readcsv("gradvar.csv"), readcsv("gradvar_gnorm.csv")
  f = readcsv("final_history.csv")
  g === nothing && return
  if haskey(g, "gnorm")          # exact ||g|| per row (runs after 2026-09-23)
    gn = Dict("loss_type" => g["loss_type"], "theta_at" => g["theta_at"], "gnorm" => g["gnorm"])
  end
  gn === nothing && return
  label(kind) = kind == "margin" ? "Alg 9" : "Alg 8"
  top = map(("init", "trained")) do where
    # absolute error: at the trained theta the exact gradient is ~0, so a relative
    # error is meaningless there.  The dotted line per algorithm is ||g|| itself:
    # below it the estimate carries signal, above it it is mostly noise.
    pe = plot(; xscale=:log10, yscale=:log10, xlabel="circuit runs N per gradient",
              ylabel=where == "init" ? "‖ĝ − g‖  (median, IQR band)" : "",
              title=where == "init" ? "gradient error at initial θ" : "gradient error at trained θ (exact g ≈ 0)",
              legend=false)
    Ns = sort(unique(g["nsamples"]))
    for kind in ("margin", "square")
      gnorm = sel(gn; loss_type=kind, theta_at=where)["gnorm"][1]
      for sampled in (1.0, 0.0)
        c = sel(g; loss_type=kind, theta_at=where, sampled=sampled)
        err(N) = c["relerr"][c["nsamples"] .== N] .* gnorm
        med = [median(err(N)) for N in Ns]
        q1, q3 = [quantile(err(N), 0.25) for N in Ns], [quantile(err(N), 0.75) for N in Ns]
        plot!(pe, Ns, med; ribbon=(med .- q1, q3 .- med), fillalpha=0.15, color=ALG[kind],
              ls=sampled == 1 ? :solid : :dash, marker=:circle, ms=4, msw=0)
      end
      hline!(pe, [gnorm]; color=ALG[kind], ls=:dot, lw=1.5)
    end
    c = sel(g; loss_type="margin", theta_at=where, sampled=1.0)
    m1 = median(c["relerr"][c["nsamples"] .== Ns[1]]) * sel(gn; loss_type="margin", theta_at=where)["gnorm"][1]
    plot!(pe, Ns, m1 .* sqrt.(Ns[1] ./ Ns); color=MUTED, lw=1, ls=:dashdot)
    pe
  end
  bottom = map(("margin", "square")) do kind
    ps = plot(; xlabel="optimiser step", ylabel=kind == "margin" ? "test accuracy" : "",
              title=ALGNAME[kind] * " — trained on single-shot gradients", ylim=(0.0, 1.02),
              legend=false)
    hline!(ps, [0.5]; color=MUTED, ls=:dot, lw=1)
    files = filter(x -> startswith(x, "shots_$(kind)_N") && endswith(x, "_history.csv"), readdir(RES))
    Ns = sort([parse(Int, match(r"_N(\d+)_", x)[1]) for x in files])
    for N in Ns
      c = readcsv("shots_$(kind)_N$(N)_history.csv")
      plot!(ps, c["step"], c["test_acc"]; color=RAMP[findfirst(==(N), (16, 64, 256))], lw=1.5)
    end
    f === nothing || (c = sel(f; loss_type=kind); plot!(ps, c["step"], c["test_acc"]; color=INK, ls=:dash, lw=1.5))
    ps
  end
  # legend-only strip: dummy series drawn outside the (hidden) axes
  lg = plot(; framestyle=:none, legend=:inside, legend_columns=4, xlim=(0, 1), ylim=(0, 1),
            legendfontsize=8, background_color_legend=nothing)
  nan = [NaN]
  for kind in ("margin", "square")
    plot!(lg, nan, nan; color=ALG[kind], marker=:circle, ms=4, msw=0, label=label(kind) * ", single shots")
    plot!(lg, nan, nan; color=ALG[kind], ls=:dash, marker=:circle, ms=4, msw=0, label=label(kind) * ", exact circuit expectations")
    plot!(lg, nan, nan; color=ALG[kind], ls=:dot, lw=1.5, label=label(kind) * " ‖g‖ (signal level)")
  end
  plot!(lg, nan, nan; color=MUTED, ls=:dashdot, lw=1, label="∝ 1/√N")
  for (i, N) in enumerate((16, 64, 256))
    plot!(lg, nan, nan; color=RAMP[i], lw=1.5, label="N = $N runs/step")
  end
  plot!(lg, nan, nan; color=INK, ls=:dash, lw=1.5, label="exact gradient")
  l = @layout [a b; c{0.13h}; d e]
  p = plot(top[1], top[2], lg, bottom[1], bottom[2]; layout=l, size=(1000, 820),
           left_margin=5Plots.mm, bottom_margin=3Plots.mm,
           plot_title="Quantum cost: gradient-estimator error (top) and training on single-shot gradients (bottom)",
           plot_titlefontsize=11)
  save(p, "E_gradient_cost_and_shots")
end

# ------------------------------------------------ F1: temperature transfer ---

function fig_f1()
  d = readcsv("kT_transfer_predictions.csv")
  d === nothing && return
  f = readcsv("final_predictions.csv")
  panels = []
  kTs = sort(unique(d["kT"]))
  for kind in ("margin", "square")
    pf = plot(; title=ALGNAME[kind], xscale=:log10, xlabel="test kT (log scale)",
              ylabel=kind == "margin" ? "test accuracy (16 held-out chains)" : "",
              xticks=([0.1, 0.25, 0.5, 1.0, 2.0], ["0.1", "0.25", "0.5", "1", "2"]),
              ylim=(0.0, 1.05), legend=:bottomleft)
    vspan!(pf, [0.09, 0.42]; color=GRID, alpha=0.5, label=false)
    annotate!(pf, 0.2, 0.62, text("cold (kT < 0.5)", 8, MUTED))
    annotate!(pf, 1.0, 0.62, text("hot (kT ≥ 0.5)", 8, MUTED))
    hline!(pf, [0.5]; color=MUTED, ls=:dot, lw=1, label="chance")
    if f !== nothing
      c = sel(f; loss_type=kind, split="test")
      ks = sort(unique(c["kT"]))
      plot!(pf, ks, [mean(c["correct"][c["kT"] .== k]) for k in ks]; color=INK, ls=:dash,
            lw=1.5, label="trained on all kT")
    end
    for (run, col, lab) in (("hot_to_cold", CLASS["XXX"], "trained hot → tested cold"),
                            ("cold_to_hot", CLASS["XX"], "trained cold → tested hot"))
      c = sel(d; loss_type=kind, run=run, split="test")
      ks = sort(unique(c["kT"]))
      plot!(pf, ks, [mean(c["correct"][c["kT"] .== k]) for k in ks]; color=col,
            marker=:circle, ms=5, msw=0, label=lab)
    end
    push!(panels, pf)
  end
  p = plot(panels...; layout=(1, 2), size=(1000, 420), left_margin=5Plots.mm,
           bottom_margin=5Plots.mm, top_margin=3Plots.mm,
           plot_title="Temperature transfer: train on one half of the kT ladder, test on the other",
           plot_titlefontsize=11)
  save(p, "F1_kT_transfer")
end

# ------------------------------------------- G: classical vs quantum ansatz ---

const KTICKS = ([0.1, 0.25, 0.5, 1.0, 2.0], ["0.1", "0.25", "0.5", "1", "2"])

function fig_g()
  d, f = readcsv("classical_predictions.csv"), readcsv("final_predictions.csv")
  (d === nothing || f === nothing) && return
  top, bottom = [], []
  for kind in ("margin", "square")
    pa = plot(; title=ALGNAME[kind], xscale=:log10, xticks=KTICKS, xlabel="test kT (log scale)",
              ylabel=kind == "margin" ? "test accuracy (16 held-out chains)" : "",
              ylim=(0.0, 1.05), legend=:bottomleft)
    hline!(pa, [0.5]; color=MUTED, ls=:dot, lw=1, label="chance")
    curve(c) = (ks = sort(unique(c["kT"])); (ks, [mean(c["correct"][c["kT"] .== k]) for k in ks]))
    q = sel(f; loss_type=kind, split="test")
    plot!(pa, curve(q)...; color=ALG[kind], marker=:circle, ms=5, msw=0,
          label=@sprintf("quantum ansatz (Z, ZZ, XX, YY)  — %.3f", mean(q["correct"])))
    runs = kind == "margin" ? (("classical_bias", "classical ansatz + bias (Z, ZZ, I)", :solid, :circle),
                               ("classical", "classical ansatz, no bias (Z, ZZ)", :dash, :utriangle)) :
                              (("classical", "classical ansatz (Z, ZZ, I)", :solid, :circle),)
    for (run, lab, ls, mk) in runs
      c = sel(d; loss_type=kind, run=run, split="test")
      isempty(c["kT"]) && continue
      plot!(pa, curve(c)...; color=CLASSICAL, ls=ls, marker=mk, ms=5, msw=0,
            label=@sprintf("%s  — %.3f", lab, mean(c["correct"])))
    end
    push!(top, pa)
    # why: the (fair, biased) classical neuron's score per test chain across temperature
    c = sel(d; loss_type=kind, split="test", run=kind == "margin" ? "classical_bias" : "classical")
    pk = plot(; xscale=:log10, xticks=KTICKS, xlabel="kT (log scale)",
              ylabel=kind == "margin" ? "classical-ansatz score" : "",
              title=kind == "margin" ? "classical ansatz + bias: score per test chain" :
                                       "classical ansatz: score per test chain", legend=:bottomleft)
    hline!(pk, [0.0]; color=INK, lw=1, ls=:dash, label=false)
    for model in ("XX", "XXX")
      first_ = true
      for g in unique(c["group"][c["model"] .== model])
        cc = sel(c; group=g)
        o = sortperm(cc["kT"])
        plot!(pk, cc["kT"][o], cc["score"][o]; color=CLASS[model], lw=1.2, alpha=0.8,
              marker=:circle, ms=3, msw=0, label=first_ ? model : false)
        first_ = false
      end
    end
    push!(bottom, pk)
  end
  p = plot(top[1], top[2], bottom[1], bottom[2]; layout=(2, 2), size=(1000, 720),
           left_margin=5Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm,
           plot_title="Does the quantum part matter?  Same neuron with only Z / ZZ terms (a classical Ising neuron)",
           plot_titlefontsize=11)
  save(p, "G_classical_ansatz")
end

# --------------------------------------------------- H: readout at inference ---

function fig_h()
  d, f = readcsv("readout.csv"), readcsv("final_predictions.csv")
  d === nothing && return
  panels = []
  for kind in ("margin", "square")
    c = sel(d; loss_type=kind)
    Ss = sort(unique(c["shots"]))
    reps = sort(unique(c["rep"]))
    ph = plot(; title=ALGNAME[kind], xscale=:log10, xlabel="measurements per test state",
              ylabel=kind == "margin" ? "test accuracy (mean over 10 repeats, min–max band)" : "",
              ylim=(0.4, 1.02), legend=kind == "margin" ? :bottomright : :right,
              xticks=(Ss, string.(Int.(Ss))))
    hline!(ph, [0.5]; color=MUTED, ls=:dot, lw=1, label="chance")
    for (lab, mask, ls, lw) in (("all states", trues(length(c["kT"])), :solid, 2.5),
                                ("cold (kT < 0.5)", c["kT"] .< 0.5, :dash, 1.8),
                                ("hot (kT ≥ 0.5)", c["kT"] .>= 0.5, :dot, 1.8))
      acc = [[mean(c["correct"][mask .& (c["shots"] .== S) .& (c["rep"] .== r)]) for r in reps] for S in Ss]
      m = mean.(acc)
      band = lab == "all states" ? (m .- minimum.(acc), maximum.(acc) .- m) : nothing
      plot!(ph, Ss, m; ribbon=band, fillalpha=0.15, color=ALG[kind], ls=ls, lw=lw,
            marker=:circle, ms=3.5, msw=0, label=lab)
    end
    i99 = findfirst(S -> mean(c["correct"][c["shots"] .== S]) >= 0.99, Ss)
    if i99 !== nothing
      annotate!(ph, Ss[1], 0.95, text("≥ 99% from $(Int(Ss[i99])) measurements/state", 8, INK, :left))
    elseif f !== nothing
      # per-measurement noise from the spread of repeated estimates at the largest
      # budget; margin = exact |score| of the median test state; 3-sigma budget
      S = maximum(Ss)
      cs = sel(c; shots=S)
      sig = median(std(cs["score"][cs["sample"] .== x]) * sqrt(S) for x in unique(cs["sample"]))
      fe = sel(f; loss_type=kind, split="test")
      need = (3 * sig / median(abs.(fe["score"])))^2
      annotate!(ph, Ss[1], 0.95, text(@sprintf("still at chance at %d: noise %.0f per measurement →", Int(S), sig), 8, INK, :left))
      annotate!(ph, Ss[1], 0.92, text(@sprintf("≈ %.1f million measurements/state for a typical state (3σ)", need / 1e6), 8, INK, :left))
    end
    push!(panels, ph)
  end
  p = plot(panels...; layout=(1, 2), size=(1000, 420), left_margin=5Plots.mm,
           bottom_margin=5Plots.mm, top_margin=3Plots.mm,
           plot_title="Inference cost: accuracy when each test state's score comes from a finite number of measurements",
           plot_titlefontsize=11)
  save(p, "H_readout_cost")
end

# ------------------------------------- per-state error map: chains x kT (G2, J) ---

"""
One marker per test state: x = kT, y = the test chain (8 XX rows, then 8 XXX
rows).  Colour and shape = true Hamiltonian; HOLLOW = correctly classified,
FILLED = misclassified.
"""
function scatter_panel(pred, title; legend=:topright, ylabel=true, ms=7, notesize=9, headroom=2.2)
  groups = unique(pred["group"])
  key(g) = (startswith(g, "XXX") ? 1 : 0, parse(Int, split(g, ':')[2]))
  order = sort(groups; by=key)
  row = Dict(g => i for (i, g) in enumerate(order))
  y = [row[g] for g in pred["group"]]
  nerr = count(==(0.0), pred["correct"])
  p = plot(; xscale=:log10, xticks=KTICKS, xlabel="kT (log scale)", xlims=(0.085, 2.6),
           ylabel=ylabel ? "test chain" : "", yticks=(1:length(order), order),
           ylims=(0.3, length(order) + headroom), title=title, legend=legend, yflip=false)
  hline!(p, [count(g -> !startswith(g, "XXX"), order) + 0.5]; color=GRID, lw=1.5, label=false)
  shape = Dict("XX" => :circle, "XXX" => :rect)
  for model in ("XX", "XXX"), ok in (1.0, 0.0)
    m = findall((pred["model"] .== model) .& (pred["correct"] .== ok))
    lab = "$model — " * (ok == 1 ? "correct (hollow)" : "misclassified (filled)") * "  n = $(length(m))"
    if isempty(m)                    # keep the legend entry so the encoding is explained
      scatter!(p, [NaN], [NaN]; marker=shape[model], ms=ms, color=CLASS[model],
               msc=CLASS[model], label=lab)
      continue
    end
    scatter!(p, pred["kT"][m], y[m]; marker=shape[model], ms=ms, msw=1.6,
             msc=CLASS[model], color=ok == 1 ? "#fcfcfb" : CLASS[model], label=lab)
  end
  annotate!(p, 0.09, length(order) + 1.2, text("$nerr / $(length(pred["correct"])) test states misclassified",
                                               notesize, INK, :left))
  return p
end

"G2: the classical-ansatz neuron's errors, state by state (exact outputs)."
function fig_g2()
  c = readcsv("classical_predictions.csv")
  c === nothing && return
  p1 = scatter_panel(sel(c; loss_type="margin", run="classical_bias", split="test"),
                     "Alg 9 · classical ansatz + bias"; ms=5.5, notesize=8)
  p2 = scatter_panel(sel(c; loss_type="square", run="classical", split="test"),
                     "Alg 8 · classical ansatz"; legend=false, ylabel=false, ms=5.5, notesize=8)
  p = plot(p1, p2; layout=(1, 2), size=(1300, 560), left_margin=6Plots.mm, bottom_margin=5Plots.mm,
           top_margin=3Plots.mm, legendfontsize=7,
           plot_title="Classical-ansatz neuron (Z, ZZ only), exact outputs: hollow = correct, filled = misclassified",
           plot_titlefontsize=11)
  save(p, "G2_scatter_classical")
end

# ------------------------------ I / J: inference by firing the neuron (Alg 5) ---

"Per-measurement noise and the 3σ budget for a typical state, from repeat spread."
function noise_budget(c, budgetcol, exact)
  B = maximum(c[budgetcol])
  cb = rowsel(c, c[budgetcol] .== B)
  tot = cb["total"][1]
  sig = median(std(cb["score"][cb["sample"] .== x]) * sqrt(tot) for x in unique(cb["sample"]))
  return sig, (3 * sig / median(abs.(exact)))^2
end

function fig_i()
  d, r, f = readcsv("fire.csv"), readcsv("readout.csv"), readcsv("final_predictions.csv")
  (d === nothing || f === nothing) && return
  panels = []
  for kind in ("margin", "square")
    c = sel(d; loss_type=kind)
    c["total"] = c["total_firings"]
    exact = sel(f; loss_type=kind, split="test")["score"]
    pm = plot(; title=ALGNAME[kind] * " model", xscale=:log10,
              xlabel="copies of ρ measured per test state",
              ylabel=kind == "margin" ? "test accuracy (mean of 10 repeats, min–max band)" : "",
              ylim=(0.4, 1.02), legend=:bottomright)
    hline!(pm, [0.5]; color=MUTED, ls=:dot, lw=1, label="chance")
    Ts = sort(unique(c["total"]))
    reps = sort(unique(c["rep"]))
    acc = [[mean(c["correct"][(c["total"] .== t) .& (c["rep"] .== q)]) for q in reps] for t in Ts]
    m = mean.(acc)
    plot!(pm, Ts, m; ribbon=(m .- minimum.(acc), maximum.(acc) .- m), fillalpha=0.15,
          color=ALG[kind], lw=2.5, marker=:circle, ms=4, msw=0,
          label=kind == "margin" ? "firing: Gaussian Alg 5, neurons on H and −H" :
                                   "firing: Gaussian Alg 5")
    if r !== nothing
      c2 = sel(r; loss_type=kind)
      Ss = sort(unique(c2["shots"]))
      plot!(pm, Ss, [mean(c2["correct"][c2["shots"] .== S]) for S in Ss]; color=ALG[kind],
            ls=:dash, lw=1.8, marker=:utriangle, ms=4, msw=0,
            label=kind == "margin" ? "readout: Pauli measurements of Tr[Hρ] (fig. H)" :
                                     "readout: value circuit (fig. H)")
    end
    sig, need = noise_budget(c, "total", exact)
    nx, ny = kind == "margin" ? (1.0, 0.975) : (90.0, 0.80)     # clear of each panel's curves
    annotate!(pm, nx, ny, text(@sprintf("firing: noise %.2f per copy of ρ", sig), 8, INK, :left))
    annotate!(pm, nx, ny - 0.025, text(@sprintf("≈ %.0f copies for a typical state (3σ)", need), 8, INK, :left))
    push!(panels, pm)
  end
  p = plot(panels...; layout=(1, 2), size=(1000, 430), left_margin=5Plots.mm,
           bottom_margin=5Plots.mm, top_margin=3Plots.mm,
           plot_title="Using the trained neurons: firing the neuron itself vs estimating its output with qubit circuits",
           plot_titlefontsize=11)
  save(p, "I_firing_accuracy")
end

"""
J: the quantum neurons classifying by firing (Gaussian Alg 5) at 4, 64 and 1024
firings per neuron (repeat 1 of 10), and the exact limit (infinitely many
firings = the exact output), which firing converges to.
"""
function fig_j()
  d, f = readcsv("fire.csv"), readcsv("final_predictions.csv")
  (d === nothing || f === nothing) && return
  budgets = (4, 64, 1024)
  panels = []
  for kind in ("margin", "square")
    name = kind == "margin" ? "Alg 9 model" : "Alg 8 model"
    for (j, N) in enumerate(budgets)
      c = sel(d; loss_type=kind, firings_per_neuron=Float64(N), rep=1.0)
      tot = kind == "margin" ? 2N : N
      push!(panels, scatter_panel(c, @sprintf("%s · %d firings/neuron (%d copies)", name, N, tot);
                                  legend=(kind == "margin" && j == 1) ? :topright : false,
                                  ylabel=j == 1, ms=4.5, notesize=8, headroom=6.0))
    end
    push!(panels, scatter_panel(sel(f; loss_type=kind, split="test"), "$name · exact limit (∞ firings)";
                                legend=false, ylabel=false, ms=4.5, notesize=8, headroom=6.0))
  end
  p = plot(panels...; layout=(2, 4), size=(1900, 1020), left_margin=11Plots.mm,
           bottom_margin=9Plots.mm, top_margin=2Plots.mm, titlefontsize=10, legendfontsize=7,
           guidefontsize=9,
           plot_title="Classifying by firing the trained quantum neurons (Gaussian Alg 5): test states, hollow = correct, filled = misclassified (repeat 1 of 10)",
           plot_titlefontsize=12)
  save(p, "J_scatter_firing")
end

# --------------------------------- K: quantum neuron vs feed-forward network ---

function fig_k()
  q, n = readcsv("kT_transfer_predictions.csv"), readcsv("ffnn_kT_transfer_predictions.csv")
  fq, fn = readcsv("fire.csv"), readcsv("ffnn_measure.csv")
  any(x -> x === nothing, (q, n, fq, fn)) && return
  bykT(c) = (ks = sort(unique(c["kT"])); (ks, [mean(c["correct"][c["kT"] .== k]) for k in ks]))
  panels = []
  for (run, title) in (("hot_to_cold", "trained hot → tested cold"), ("cold_to_hot", "trained cold → tested hot"))
    ks = run == "hot_to_cold" ? [0.1, 0.13, 0.18, 0.25, 0.35] : [0.5, 0.71, 1.0, 1.41, 2.0]
    pt = plot(; title=title, xscale=:log10, xticks=(ks, string.(ks)), xlims=(ks[1] / 1.12, ks[end] * 1.12),
              xlabel="test kT (log scale)",
              ylabel=run == "hot_to_cold" ? "test accuracy (16 held-out chains)" : "",
              ylim=(0.85, 1.01), legend=:bottomright)
    for kind in ("margin", "square")
      c = sel(q; loss_type=kind, run=run, split="test")
      plot!(pt, bykT(c)...; color=ALG[kind], marker=:circle, ms=5, msw=0,
            label=(kind == "margin" ? "Alg 9 neuron" : "Alg 8 neuron") * @sprintf(" — %.3f", mean(c["correct"])))
    end
    c = sel(n; run=run)
    plot!(pt, bykT(c)...; color=FFNN, ls=:dashdot, marker=:rect, ms=5, msw=0,
          label=@sprintf("feed-forward network (5 seeds) — %.3f", mean(c["correct"])))
    push!(panels, pt)
  end
  pc = plot(; title="accuracy vs copies of ρ per test state", xscale=:log10,
            xlabel="copies of ρ measured per test state", ylim=(0.5, 1.01), legend=:bottomright)
  for kind in ("margin", "square")
    c = sel(fq; loss_type=kind)
    Ts = sort(unique(c["total_firings"]))
    plot!(pc, Ts, [mean(c["correct"][c["total_firings"] .== t]) for t in Ts]; color=ALG[kind],
          marker=:circle, ms=4, msw=0,
          label=kind == "margin" ? "Alg 9 neuron, fired (H and −H)" : "Alg 8 neuron, fired")
  end
  Ss = sort(unique(fn["copies"]))
  plot!(pc, Ss, [mean(fn["correct"][fn["copies"] .== S]) for S in Ss]; color=FFNN, ls=:dashdot,
        marker=:rect, ms=4, msw=0, label="feed-forward network, inputs measured\n(S/3 copies each in Z, X, Y bases)")
  push!(panels, pc)
  p = plot(panels...; layout=(1, 3), size=(1500, 500), left_margin=7Plots.mm, bottom_margin=9Plots.mm,
           top_margin=7Plots.mm, legendfontsize=7,
           plot_title="Quantum GReLU neuron vs a classical feed-forward network on the same split",
           plot_titlefontsize=11)
  save(p, "K_quantum_vs_ffnn")
end

# ------------------------------------- L: correct / incorrect counts per class ---

function fig_l()
  fq, fn = readcsv("fire.csv"), readcsv("ffnn_measure.csv")
  (fq === nothing || fn === nothing) && return
  # roughly matched copies of rho per state: (Alg 9 total, Alg 8, network)
  levels = (("≈ 32 copies", 32, 16, 48), ("≈ 128 copies", 128, 64, 192), ("≈ 512 copies", 512, 256, 768))
  panels = []
  for (r, (lvl, n9, n8, nf)) in enumerate(levels)
    for (c, (name, d)) in enumerate((("Alg 9 neuron, fired", sel(fq; loss_type="margin", total_firings=Float64(n9))),
                                     ("Alg 8 neuron, fired", sel(fq; loss_type="square", total_firings=Float64(n8))),
                                     ("network, measured inputs", sel(fn; copies=Float64(nf)))))
      copies = c == 1 ? n9 : c == 2 ? n8 : nf
      runs = c == 3 ? length(unique(d["rep"])) * length(unique(d["seed"])) : length(unique(d["rep"]))
      p = plot(; title="$name · $copies copies · mean of $runs runs", ylim=(0, 92),
               legend=(r == 1 && c == 1) ? :topright : false,
               ylabel=c == 1 ? "test states ($lvl)" : "",
               xticks=([1, 2], ["XX", "XXX"]), xlims=(0.4, 2.6), titlefontsize=9)
      for (k, model) in enumerate(("XX", "XXX"))
        m = d["model"] .== model
        ok = sum(d["correct"][m]) / runs
        bad = count(m) / runs - ok
        bar!(p, [k - 0.18], [ok]; bar_width=0.34, color="#fcfcfb", linecolor=CLASS[model], lw=2,
             label=k == 1 ? "correct (hollow)" : false)
        bar!(p, [k + 0.18], [bad]; bar_width=0.34, color=CLASS[model], linecolor=CLASS[model], lw=2,
             label=k == 1 ? "misclassified (filled)" : false)
        annotate!(p, k - 0.18, ok + 4, text(@sprintf("%.1f", ok), 8, INK))
        annotate!(p, k + 0.18, bad + 4, text(@sprintf("%.1f", bad), 8, INK))
      end
      push!(panels, p)
    end
  end
  p = plot(panels...; layout=(3, 3), size=(1200, 1050), left_margin=6Plots.mm, bottom_margin=3Plots.mm,
           top_margin=2Plots.mm, legendfontsize=8,
           plot_title="Test states correctly and incorrectly classified, by true Hamiltonian (80 XX + 80 XXX)",
           plot_titlefontsize=11)
  save(p, "L_classified_counts")
end

# -------------------------- I2: the qumode simulated in ITensor vs exact sampler ---

function fig_i2()
  v, e = readcsv("qumode_itensor_validation.csv"), readcsv("qumode_itensor_example.csv")
  (v === nothing || e === nothing) && return
  # left: output CDF of one example state, Alg 8 model (the hardest case), vacuum qumode
  pa = plot(; title="Alg 8 neuron, one fixture state: output distribution",
            xlabel="neuron output  T2·ReLU(p)", ylabel="P(output ≤ y)", legend=:bottomright)
  c = sel(e; loss_type="square")
  vac = abs.(c["T1"] .- 1 / sqrt(2)) .< 1e-6
  ds_ = sort(unique(c["d"][vac]))
  for (i, d) in enumerate(ds_)
    m = vac .& (c["d"] .== d)
    plot!(pa, c["y_out"][m], c["cdf_itensor"][m]; color=RAMP[i + 1], lw=2,
          label="ITensor qumode, Fock cutoff d = $(Int(d))")
  end
  m = vac .& (c["d"] .== ds_[end])
  plot!(pa, c["y_out"][m], c["cdf_exact"][m]; color=INK, ls=:dash, lw=1.5, label="exact sampler (analytic)")
  # right: worst-case error vs Fock cutoff, both models, two squeezings
  pb = plot(; title="worst case over 24 fixture states", xlabel="Fock cutoff d",
            ylabel="max |mean output − exact|", yscale=:log10, legend=:topright, xticks=[20, 40, 60, 80])
  for kind in ("margin", "square"), (T1, ls, mk, lab) in ((1 / sqrt(2), :solid, :circle, "vacuum, T1 = 1/√2"),
                                                         (0.45, :dash, :rect, "squeezed, T1 = 0.45"))
    r = sel(v; loss_type=kind)
    r = rowsel(r, abs.(r["T1"] .- T1) .< 1e-6)
    ds2 = sort(unique(r["d"]))
    err = [maximum(abs.(r["mean_itensor"][r["d"] .== d] .- r["mean_exact"][r["d"] .== d])) for d in ds2]
    plot!(pb, ds2, max.(err, 1e-8); color=ALG[kind], ls=ls, marker=mk, ms=5, msw=0,
          label=(kind == "margin" ? "Alg 9 neuron" : "Alg 8 neuron") * ", " * lab)
  end
  p = plot(pa, pb; layout=(1, 2), size=(1150, 450), left_margin=6Plots.mm, bottom_margin=6Plots.mm,
           top_margin=3Plots.mm, legendfontsize=8,
           plot_title="Algorithm 5 with an explicit qumode (ITensor \"Boson\" site) reproduces the exact firing sampler",
           plot_titlefontsize=11)
  save(p, "I2_qumode_itensor")
end

gr()
for f in (fig_a2, fig_b1, fig_c, fig_d1, fig_e, fig_f1, fig_g, fig_g2, fig_h, fig_i, fig_i2, fig_j, fig_k, fig_l)
  f()
end
