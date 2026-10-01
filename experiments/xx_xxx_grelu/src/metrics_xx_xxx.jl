# metrics_xx_xxx.jl
#
# Classification metrics on the 160 held-out test states for every trained
# model.
#
#   julia experiments/xx_xxx_grelu/src/metrics_xx_xxx.jl
#
# Positive class = XXX.  For a binary task:
#   accuracy            (TP + TN) / N
#   precision, recall   for XXX; F1 their harmonic mean
#   IoU (XXX)           TP / (TP + FP + FN)  -- Jaccard index of the predicted
#                       and true XXX sets
#   mIoU                mean of the XXX and XX IoUs
#   AP                  average precision from the ranked scores (area under the
#                       precision-recall curve, step-wise, no interpolation)
#   mAP                 mean of the XXX AP (score) and XX AP (-score)
#   AUC                 ROC area (Mann-Whitney)
# Accuracy, precision, recall, F1 and IoU use the model's own decision rule;
# AP, mAP and AUC use only the ranking of its scores.
#
# Output: ../results/classification_metrics.csv

using Printf, Statistics

const EXP = normpath(joinpath(@__DIR__, ".."))
const RES = joinpath(EXP, "results")

function readcsv(path)
  lines = readlines(path)
  hdr = split(lines[1], ',')
  rows = [split(l, ',') for l in lines[2:end]]
  pc(v) = (x = tryparse(Float64, v); x === nothing ? String(v) : x)
  return Dict(String(h) => [pc(r[i]) for r in rows] for (i, h) in enumerate(hdr))
end
rowsel(d, mask) = Dict(k => v[mask] for (k, v) in d)

"Average precision of ranking `score` for positives `pos` (Bool vector)."
function average_precision(score, pos)
  o = sortperm(score; rev=true)
  tp, ap, P = 0, 0.0, count(pos)
  for (i, j) in enumerate(o)
    pos[j] || continue
    tp += 1
    ap += tp / i                  # precision at this recall step
  end
  return ap / P
end

auc(score, pos) = mean((p > q) + 0.5 * (p == q) for p in score[pos], q in score[.!pos])

function metrics(score, pred_pos, pos)
  TP, FP = count(pred_pos .& pos), count(pred_pos .& .!pos)
  FN, TN = count(.!pred_pos .& pos), count(.!pred_pos .& .!pos)
  prec = TP + FP == 0 ? NaN : TP / (TP + FP)
  rec = TP / (TP + FN)
  f1 = 2prec * rec / (prec + rec)
  iou_pos = TP / (TP + FP + FN)
  iou_neg = TN / (TN + FN + FP)
  ap_pos, ap_neg = average_precision(score, pos), average_precision(-score, .!pos)
  return (; n=length(pos), TP=Float64(TP), FP=Float64(FP), FN=Float64(FN), TN=Float64(TN), accuracy=(TP + TN) / length(pos), precision=prec,
          recall=rec, f1, iou=iou_pos, miou=(iou_pos + iou_neg) / 2, ap=ap_pos,
          map=(ap_pos + ap_neg) / 2, auc=auc(score, pos))
end

function main()
  rows = NamedTuple[]
  add(name, params, d, scorecol) = begin
    pos = d["model"] .== "XXX"
    score = Float64.(d[scorecol])
    pred = score .>= 0                         # every model's rule: XXX iff score >= 0
    push!(rows, (; model=name, parameters=params, metrics(score, pred, pos)...))
  end

  f = readcsv(joinpath(RES, "final_predictions.csv"))
  t = rowsel(f, f["split"] .== "test")
  add("GReLU neuron, Alg 9 (quantum ansatz)", 37, rowsel(t, t["loss_type"] .== "margin"), "score")
  add("GReLU neuron, Alg 8 (quantum ansatz)", 38, rowsel(t, t["loss_type"] .== "square"), "score")

  c = readcsv(joinpath(RES, "classical_predictions.csv"))
  t = rowsel(c, c["split"] .== "test")
  add("GReLU neuron, Alg 9 (classical ansatz + bias)", 20,
      rowsel(t, (t["loss_type"] .== "margin") .& (t["run"] .== "classical_bias")), "score")
  add("GReLU neuron, Alg 8 (classical ansatz)", 20,
      rowsel(t, (t["loss_type"] .== "square") .& (t["run"] .== "classical")), "score")

  n = readcsv(joinpath(RES, "ffnn_predictions.csv"))
  add("Feed-forward ReLU network (seed 1)", 391, n, "logit")

  # the trained quantum neurons used by FIRING them (Gaussian Algorithm 5) at
  # finite budgets: metrics per repeat, averaged over the 10 repeats
  fp = joinpath(RES, "fire.csv")
  if isfile(fp)
    fr = readcsv(fp)
    for (kind, label, J) in (("margin", "Alg 9", 37), ("square", "Alg 8", 38)), N in (16, 64, 256, 1024)
      ms = map(sort(unique(fr["rep"]))) do r
        d = rowsel(fr, (fr["loss_type"] .== kind) .& (fr["firings_per_neuron"] .== N) .& (fr["rep"] .== r))
        pos = d["model"] .== "XXX"
        metrics(Float64.(d["score"]), d["score"] .>= 0, pos)
      end
      tot = kind == "margin" ? 2N : N
      avg = NamedTuple{keys(ms[1])}(Tuple(mean(getfield(m, k) for m in ms) for k in keys(ms[1])))
      push!(rows, (; model="GReLU neuron, $label, fired $N×/neuron ($tot copies of ρ; mean of $(length(ms)) repeats)",
                   parameters=J, avg...))
    end
  end

  open(joinpath(RES, "classification_metrics.csv"), "w") do io
    ks = collect(keys(first(rows)))
    println(io, join(ks, ","))
    for r in rows
      # text fields are quoted: model names contain commas
      println(io, join((v isa AbstractString ? "\"" * replace(v, "\"" => "\"\"") * "\"" :
                        v isa Integer ? string(v) : @sprintf("%.4f", v) for v in values(r)), ","))
    end
  end
  @printf("%-48s %5s %5s %5s %5s %5s %5s %5s %5s\n", "model", "acc", "prec", "rec", "F1", "IoU", "mIoU", "mAP", "AUC")
  for r in rows
    @printf("%-48s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", r.model, r.accuracy, r.precision,
            r.recall, r.f1, r.iou, r.miou, r.map, r.auc)
  end

  println("wrote results/classification_metrics.csv")
end

main()
