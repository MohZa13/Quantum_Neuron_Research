# convergence_checks.jl

# Checks 0-3 are correctness (compare against something known).  Checks 4-6 are
# convergence (vary a knob until the answer stops moving).  Check 7 is the cost
# curve that tells you where MPS stops being cheap.
#
#   0  training states     DMRG rho_m vs exact-diagonalisation rho_m
#   1  ancilla-free path   :overlap == :hadamard (identity, so ~machine eps)
#   2  vs exact gradient   Algorithm 9 vs central differences on L^log
#   3  backend agreement   Alg9ITensor vs Alg9Yao on identical sample draws
#   4  Trotter step        sweep dtmax    (expect 2nd order: error ~ dtmax^2)
#   5  bond dimension      sweep maxdim   (expect a plateau)
#   6  truncation cutoff   sweep cutoff   (expect a plateau)
#   7  Monte Carlo         sweep nsamples (expect error ~ 1/sqrt(nsamples))
#   8  cost vs n           bond dimension and wall time as n grows
#
# Checks 4-7 reuse one fixed seed, so the (s, t, k) draws are identical across
# a sweep -- that isolates the discretisation error from Monte Carlo noise.
#
#   julia> include("convergence_checks.jl")
#   julia> run_all(; n=4)                 # correctness + convergence
#   julia> cost_vs_n(; ns=4:2:12)         # check 8 on its own, slower

ENV["ALG9_DEMO"] = "0"
include("algorithm9_yao.jl")
include("algorithm9.jl")

using ITensorMPS: maxlinkdim
using LinearAlgebra, Printf, Statistics, Random

const SEED = 20260724

relerr(a, b) = norm(a - b) / max(norm(b), eps())
cosim(a, b) = dot(a, b) / (norm(a) * norm(b))

header(s) = (println("\n", "="^72); println(s); println("="^72))

# --- 0: are the two backends even starting from the same training states? ---

"""
Alg9Yao builds rho_m by exact diagonalisation, Alg9ITensor by DMRG.  If those
disagree nothing downstream can agree.  The TFIM is near-degenerate at small
h and finite n, so DMRG can land on a different combination -- that shows up
here as an overlap below 1 rather than as a mysterious gradient mismatch later.
"""
function check_training_states(; n=4, T=2.0, seed=1234)
  header("0. training states: DMRG (ITensor) vs exact diagonalisation (Yao)")
  sty = Alg9Yao.setup(; n, T, seed)
  sti = Alg9ITensor.setup(; n, T, seed)
  ok = true
  for m in eachindex(sty.vecs)
    ov = abs(dot(sty.vecs[m], sti.vecs[m]))       # global phase is irrelevant
    ok &= ov > 1 - 1e-6
    @printf("   m=%d   |<psi_ED | psi_DMRG>| = %.10f   %s\n",
            m, ov, ov > 1 - 1e-6 ? "ok" : "MISMATCH")
  end
  ok || println("   -> raise DMRG nsweeps, or expect checks 3 and 8 to disagree.")
  return (; sty, sti, ok)
end

# --- 1: the ancilla-free path is an identity, not an approximation ----------

"""
:overlap and :hadamard(sampled=false) compute the same number by different
routes.  Any disagreement beyond roundoff is a bug in the controlled-gate
construction, not a convergence issue -- so this is the check that has to pass
exactly before the cheap path can be used at large n.
"""
function check_overlap_identity(; n=4, T=2.0, seed=1234, nsamples=40)
  header("1. ancilla-free identity: :overlap vs :hadamard")

  sty = Alg9Yao.setup(; n, T, seed)
  gh = Alg9Yao.logloss_gradient(sty; nsamples, seed=SEED, sampled=false, method=:hadamard)
  go = Alg9Yao.logloss_gradient(sty; nsamples, seed=SEED, sampled=false, method=:overlap)
  @printf("   Yao        rel.err %.3e   cos %.12f\n", relerr(go, gh), cosim(go, gh))

  sti = Alg9ITensor.setup(; n, T, seed)
  ih = Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED, method=:hadamard)
  io = Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED, method=:overlap)
  @printf("   ITensor    rel.err %.3e   cos %.12f\n", relerr(io, ih), cosim(io, ih))
  println("   (Yao should be ~1e-14; ITensor is limited by maxdim/cutoff, not by method.)")
  return nothing
end

# --- 2: does Algorithm 9 reproduce the true gradient? -----------------------

"""
Algorithm 9 is a Monte Carlo estimator, so it converges to -- not equals -- the
central-difference gradient.  Watch the trend across nsamples, not any single
number.  This is the only check with access to ground truth, so it only works
at n small enough to diagonalise.
"""
function check_vs_exact(; n=4, T=2.0, seed=1234, sweep=(50, 100, 200, 400, 800, 1600))
  header("2. Algorithm 9 vs exact gradient (central differences on L^log)")
  st = Alg9Yao.setup(; n, T, seed)
  ge = Alg9Yao.exact_logloss_gradient(st, st.theta)
  @printf("   |g_exact| = %.6f\n\n", norm(ge))
  println("   nsamples   rel.err      cos       rel.err*sqrt(nsamples)")
  for ns in sweep
    g = Alg9Yao.logloss_gradient(st; nsamples=ns, seed=SEED, sampled=false, method=:overlap)
    e = relerr(g, ge)
    @printf("   %8d   %.5f   %.5f   %8.3f\n", ns, e, cosim(g, ge), e * sqrt(ns))
  end
  println("   (last column should flatten -- that is the 1/sqrt(nsamples) law)")
  return nothing
end

# --- 3: do the two backends agree on identical draws? ----------------------

"""
Both modules draw (s, t, k) from the same MersenneTwister seed in the same
order, off identical gamma grids, so they see the SAME samples.  With that
noise removed the only remaining difference is MPS truncation, which makes this
a direct measurement of the tensor-network error.
"""
function check_backends(; n=4, T=2.0, seed=1234, nsamples=100, maxdim=200, cutoff=1e-12)
  header("3. backend agreement on identical sample draws")
  sty = Alg9Yao.setup(; n, T, seed)
  sti = Alg9ITensor.setup(; n, T, seed)
  gy = Alg9Yao.logloss_gradient(sty; nsamples, seed=SEED, sampled=false, method=:overlap)
  gi = Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED, method=:overlap,
                                    maxdim, cutoff, dtmax=0.05)
  @printf("   rel.err %.3e   cos %.10f\n", relerr(gi, gy), cosim(gi, gy))
  println("   (residual here is MPS truncation + Trotter error, not Monte Carlo)")
  return nothing
end

# --- 4-6: MPS discretisation sweeps ----------------------------------------

"""
Sweep one knob, hold the sample draws fixed, and measure against the most
accurate setting in the sweep.  A converged setting is one where tightening
further stops changing the answer.
"""
function sweep_knob(name, values, gradfn; ref=nothing)
  header("$name sweep")
  res = map(values) do v
    local g
    t = @elapsed (g = gradfn(v))
    (; v, g, t)
  end
  gref = ref === nothing ? res[end].g : ref
  @printf("   reference: %s\n\n", ref === nothing ? "tightest in sweep" : "supplied")
  println("   value          rel.err vs ref    cos        seconds")
  for r in res
    @printf("   %-12s   %.3e        %.8f   %7.2f\n",
            string(r.v), relerr(r.g, gref), cosim(r.g, gref), r.t)
  end
  return res
end

function check_trotter(sti; nsamples=40, maxdim=200, cutoff=1e-12,
                       sweep=(0.4, 0.2, 0.1, 0.05, 0.025))
  sweep_knob("4. Trotter step dtmax (2nd order: error should fall ~4x per halving)",
             sweep,
             dt -> Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED,
                                                method=:overlap, maxdim, cutoff, dtmax=dt))
end

function check_maxdim(sti; nsamples=40, cutoff=1e-14, dtmax=0.05,
                      sweep=(8, 16, 32, 64, 128, 256))
  sweep_knob("5. bond dimension maxdim (expect a plateau)", sweep,
             md -> Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED,
                                                method=:overlap, maxdim=md, cutoff, dtmax))
end

function check_cutoff(sti; nsamples=40, maxdim=256, dtmax=0.05,
                      sweep=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14))
  sweep_knob("6. truncation cutoff (expect a plateau)", sweep,
             c -> Alg9ITensor.logloss_gradient(sti; nsamples, seed=SEED,
                                               method=:overlap, maxdim, cutoff=c, dtmax))
end

# --- 8: where does MPS stop being cheap? -----------------------------------

"""
Bond dimension and wall time as n grows, for both the ancilla-free path and the
paper's controlled-gate circuit.  The gap between the two columns is the cost
of the ancilla: it sits at site 1 while the Trotter gates act on (i, i+1), so
every controlled gate is non-local and `apply` pays O(N) swaps for it.

Uses a single worst-case-ish evolution time rather than sampling, so the
numbers are an upper bound on what a typical (s, t) draw costs.
"""
function cost_vs_n(; ns=4:2:12, T=2.0, seed=1234, t=1.0, s=0.5,
                   maxdim=256, cutoff=1e-12, dtmax=0.05)
  header("8. cost vs n   (t=$t, s=$s, maxdim=$maxdim, cutoff=$cutoff, dtmax=$dtmax)")
  println("      n     J    chi(overlap)   sec(overlap)   chi(hadamard)   sec(hadamard)")
  for n in ns
    st = Alg9ITensor.setup(; n, T, seed)
    hb = Alg9ITensor.bond_hamiltonians(st.sys, st.y[1] .* st.theta, st.terms)
    ev = (; maxdim, cutoff, dtmax)

    # ancilla-free: two ordinary local evolutions
    to = @elapsed begin
      phi = Alg9ITensor.evolve(st.rho[1], hb, s * t / T; ev...)
      chi = Alg9ITensor.evolve(phi, hb, -t / T; ev...)
    end
    chio = max(maxlinkdim(phi), maxlinkdim(chi))

    # paper's circuit: ancilla at site 1, every controlled gate non-local
    th = @elapsed begin
      full = Alg9ITensor.with_ancilla(st.anc, st.rho[1])
      mid = Alg9ITensor.hadamard_prefix(st.anc, full, hb, t, s, T; ev...)
    end
    @printf("   %4d  %4d   %10d   %12.2f   %11d   %13.2f\n",
            n, length(st.terms), chio, to, maxlinkdim(mid), th)
  end
  println("   chi hitting maxdim means the sweep in check 5 was not converged at this n.")
  return nothing
end

# --------------------------------------------------------------------- run --

"""
    run_all(; n, nsamples)

Correctness and convergence at a size small enough to have an exact reference.
`cost_vs_n` is separate because it is much slower.
"""
function run_all(; n=4, T=2.0, seed=1234, nsamples=40)
  r = check_training_states(; n, T, seed)
  check_overlap_identity(; n, T, seed, nsamples)
  check_vs_exact(; n, T, seed)
  check_backends(; n, T, seed, nsamples=100)
  check_trotter(r.sti; nsamples)
  check_maxdim(r.sti; nsamples)
  check_cutoff(r.sti; nsamples)
  header("done -- checks 0-3 are correctness, 4-6 convergence")
  println("Run cost_vs_n(; ns=4:2:12) separately for the scaling curve.")
  return nothing
end

run_all(; n=4)
