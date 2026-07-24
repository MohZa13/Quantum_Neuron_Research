# train_alg9.jl
#
# Minimise the LOGISTIC loss using Algorithm 9 as the gradient oracle and
# Optimisers.jl for the update.  Unlike train_alg8.jl, the gradient here is the
# gradient of the objective being minimised, so this is honest gradient descent
# on L^log rather than descent on a different loss.
#
#   L^log(theta) = (1/M) sum_m T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m],  y_m in {-1,+1}
#
# Optimisers.jl needs no glue: theta is a plain Vector{Float64}, a valid
# Optimisers parameter tree, and Algorithm 9 returns a gradient of that shape.
#
# Backends (same interface, pick with `backend=`):
#   Alg9Yao      -- statevector, fast, the default for training
#   Alg9ITensor  -- MPS/DMRG, for validation and larger n; too slow to loop
#
#   julia> import Pkg; Pkg.add(["Yao", "ITensors", "ITensorMPS", "Optimisers"])
#   julia> include("train_alg9.jl")

ENV["ALG9_DEMO"] = "0"            # skip the modules' self-test blocks
include("algorithm9_yao.jl")
# include("algorithm9.jl")

using Optimisers
using LinearAlgebra, Printf, Logging

# --------------------------------------------------------------- training ---

"""
    train(; backend, n, T, steps, rule, nsamples, sampled, seed, compare)

Gradient descent on L^log driven by Algorithm 9.  `sampled=false` (Alg9Yao only)
uses exact expectation values inside the Hadamard test while keeping the Monte
Carlo over s, t, k -- that is what makes a few hundred samples per step enough.
`sampled=true` is the paper's single-shot estimator and needs far more samples.

Each step also prints cos(g_alg9, g_exact) against a central-difference
reference.  If that is not comfortably positive the estimator is too noisy for
the loss curve to mean anything.
"""
function train(; backend=Alg9Yao, n::Int=4, T::Float64=2.0, steps::Int=40,
               rule=Optimisers.Adam(0.05), nsamples::Int=400,
               sampled::Bool=false, seed::Int=1234, compare::Bool=true)

  st = backend.setup(; n, T, seed)
  theta0 = copy(st.theta)
  theta = copy(st.theta)

  # Optimisers.jl on a plain Vector: no model wrapper needed.
  opt = Optimisers.setup(rule, theta)

  # Alg9ITensor takes expectation values only -- it has no `sampled` option.
  gradkw = backend === Alg9Yao ? (; nsamples, sampled) : (; nsamples)

  hist = NamedTuple[]
  println("step     L^log     |g_alg9|   cos(g_alg9, g_exact)")
  for step in 1:steps
    st = backend.with_theta(st, theta)

    g = with_logger(NullLogger()) do
      backend.logloss_gradient(st; seed=step, gradkw...)
    end
    L = backend.exact_logloss(st, theta)
    c = compare ? (ge = backend.exact_logloss_gradient(st, theta);
                   dot(g, ge) / (norm(g) * norm(ge))) : NaN

    push!(hist, (; step, logloss=L, gnorm=norm(g), cos=c))
    @printf("%4d  %9.5f  %9.4f  %9.3f\n", step, L, norm(g), c)

    opt, theta = Optimisers.update!(opt, theta, g)
  end

  st = backend.with_theta(st, theta)
  return (; theta, theta0, st, hist, backend)
end

# ------------------------------------------------------- parameter report ---

"""
Print every H_j and its optimised weight theta_j, so the optimal
H(theta) = sum_j theta_j H_j can be read off directly.
"""
function report_parameters(res; bylabel=true)
  A8 = res.backend
  terms, th, th0 = res.st.terms, res.theta, res.theta0
  ord = bylabel ? eachindex(th) : sortperm(abs.(th); rev=true)

  println("\nOptimal H(theta) = sum_j theta_j H_j     (J = $(length(th)) terms)")
  println("   j  H_j            theta_j(init)   theta_j(opt)     change")
  for j in ord
    @printf("%4d  %-13s  %13.5f  %13.5f  %10.5f\n",
            j, A8.term_label(terms[j]), th0[j], th[j], th[j] - th0[j])
  end
  @printf("\n||theta||_1 = %.5f    ||theta||_2 = %.5f\n", norm(th, 1), norm(th, 2))

  println("\nLargest-magnitude terms:")
  for j in sortperm(abs.(th); rev=true)[1:min(5, length(th))]
    @printf("   %-13s  %+.5f\n", A8.term_label(terms[j]), th[j])
  end

  # Per-example logistic loss and the sign the model predicts for each y_m.
  println("\nPer-example loss and predicted label:")
  for m in eachindex(res.st.vecs)
    stm = merge(res.st, (; vecs=[res.st.vecs[m]], y=[res.st.y[m]]))
    Lm = A8.exact_logloss(stm, th)
    stp = merge(res.st, (; vecs=[res.st.vecs[m]], y=[+1.0]))
    stn = merge(res.st, (; vecs=[res.st.vecs[m]], y=[-1.0]))
    pred = A8.exact_logloss(stp, th) < A8.exact_logloss(stn, th) ? +1.0 : -1.0
    @printf("   m=%d   loss %8.5f   predicted %+.0f   true %+.0f   %s\n",
            m, Lm, pred, res.st.y[m], pred == res.st.y[m] ? "ok" : "MISCLASSIFIED")
  end
  return nothing
end

# ----------------------------------------------------------------- run -----

res = train(; backend=Alg9Yao, n=4, steps=40, rule=Optimisers.Adam(0.05),
            nsamples=400, sampled=false)

@printf("\nL^log:  %.5f -> %.5f\n", res.hist[1].logloss, res.hist[end].logloss)
@printf("logloss at final theta:  Eq.(C44) estimate %.4f   exact %.4f\n",
        Alg9Yao.logloss(res.st; nsamples=4000),
        Alg9Yao.exact_logloss(res.st, res.theta))

report_parameters(res)

# Cross-check the MPS backend's gradient against the statevector one at the
# optimised parameters.  Both implement Eq. (C27); they should agree.
#
#   sti = Alg9ITensor.setup(; n=4)
#   sti = Alg9ITensor.with_theta(sti, res.theta)
#   gi  = Alg9ITensor.logloss_gradient(sti; nsamples=64)
#   gy  = Alg9Yao.logloss_gradient(res.st; nsamples=400, sampled=false)
#   @show dot(gi, gy) / (norm(gi) * norm(gy))
