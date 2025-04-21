# 1‑D gh scan at (x_shift, Ca_shift) = (‑1, ‑36)
using Pkg
Pkg.activate("./lyapunov_old")

using StaticArrays, OrdinaryDiffEq, DynamicalSystems, ProgressMeter, GLMakie
import LinearAlgebra, Base.Threads

# --- model definition & fixed data ------------------------------------------------
include("../model/Plant.jl")       # assumes Plant.default_params is available

function melibeNew(u::AbstractArray{T}, p, t) where T
    @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

u0 = @SVector Float64[0.2, 0.1, 0.137, 0.389, 1.0, -62.0]   # initial state
x_shift_fixed  = -1.2
Ca_shift_fixed = -37.0

# --- parameter vector builder -----------------------------------------------------
function build_params(gh)
    # default_params[4] is gh; last two entries are (x_shift, Ca_shift)
    [Plant.default_params[1:3];
     gh;
     Plant.default_params[5:15];
     x_shift_fixed;
     Ca_shift_fixed]
end

# --- scan specification -----------------------------------------------------------
n_gh      = 100
gh_range  = LinRange(0.0, 0.02, n_gh)      # same numeric range as previous work
n_steps   = 500_000                        # integration length for lyapunovspectrum

λ1 = zeros(n_gh)
λ2 = zeros(n_gh)
λ3 = zeros(n_gh)

progress = Progress(n_gh, desc = "1‑D gh scan", dt = 1)

Threads.@threads for k in 1:n_gh
    p   = build_params(gh_range[k])
    sys = ContinuousDynamicalSystem(melibeNew, u0, p)
    ly  = lyapunovspectrum(sys, n_steps)
    λ1[k] = ly[1]
    λ2[k] = length(ly) ≥ 2 ? ly[2] : 0.0
    λ3[k] = length(ly) ≥ 3 ? ly[3] : 0.0
    next!(progress)
end

# --- plotting ---------------------------------------------------------------------
GLMakie.activate!()                             # use OpenGL backend
fig = Figure(resolution = (800, 500))
ax  = Axis(fig[1, 1],
           xlabel = "gh",
           ylabel = "Lyapunov exponent",
           title  = "Lyapunov spectrum at (x_shift, Ca_shift) = (‑1, ‑36)")

lines!(ax, gh_range, λ1, label = "λ₁")
lines!(ax, gh_range, λ2, label = "λ₂")
lines!(ax, gh_range, λ3, label = "λ₃")
axislegend(ax, position = :rb)

save("lyapunov_1d_gh_scan.png", fig)   # optional
display(fig)
