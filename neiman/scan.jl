# ========================= scan.jl =========================
# Sweep all gh values, compute full Lyapunov maps, and save
# each result to disk. A single progress bar tracks the entire
# parameter sweep across every (gh, x_shift, Ca_shift) point.
# -----------------------------------------------------------

using Pkg
Pkg.activate(".")

using StaticArrays, OrdinaryDiffEq, LinearAlgebra, ForwardDiff
using DynamicalSystems, ProgressMeter, JLD2
import LinearAlgebra.BLAS

# Prevent oversubscription when Julia already uses threads
BLAS.set_num_threads(1)

# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
include("../model/Plant.jl")

function melibeNew(u::AbstractVector{T}, p, t) where T
    @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

# ------------------------------------------------------------------
# Constants and parameter grid
# ------------------------------------------------------------------
const u0          = @SVector Float64[0.2, 0.1, 0.137, 0.389, 1.0, -62.0]
const resolution  = 1000
const Ca_shifts   = LinRange(-50.0, -20.0, resolution)
const x_shifts    = LinRange(-1.6, 0.4,   resolution)
const gh_values   = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.02]

# Total number of lattice points over all gh
const TOTAL_POINTS = length(gh_values) * resolution^2

# Global progress bar for the *entire* scan
const global_progress = Progress(TOTAL_POINTS; desc="Full parameter sweep", dt=1)

# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------
function compute_lyapunov_for_gh(gh::Float64, progress::Progress)
    # Build parameter array (resolution × resolution)
    ps = [[Plant.default_params[1:3]; [gh]; Plant.default_params[5:15]; [x_shifts[i], Ca_shifts[j]]]
          for i in 1:resolution, j in 1:resolution]

    lyap_vals = zeros(4, resolution, resolution)

    Threads.@threads for i in 1:resolution
        local sys, lyaps
        for j in 1:resolution
            sys   = ContinuousDynamicalSystem(melibeNew, u0, ps[i,j])
            lyaps = lyapunovspectrum(sys, 500_000)
            @inbounds for k = 1:min(4, length(lyaps))
                lyap_vals[k,i,j] = lyaps[k]
            end
            next!(progress)      # advance *global* progress bar
        end
    end

    return lyap_vals
end

# ------------------------------------------------------------------
# Main sweep: compute → save, with unified progress bar
# ------------------------------------------------------------------
for gh in gh_values
    @info "Scanning gh = $gh"
    lyap_vals = compute_lyapunov_for_gh(gh, global_progress)
    fname = "lyapunov_scan_$(replace(string(gh), '.' => '_')).jld2"
    @info "Writing $fname"
    @save fname lyap_vals gh Ca_shifts x_shifts
end

# Ensure the final state of the progress bar is printed
finish!(global_progress)
