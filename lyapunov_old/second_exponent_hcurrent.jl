using Pkg
Pkg.activate("../lyapunov_old/")
Pkg.instantiate()
using GLMakie, StaticArrays, OrdinaryDiffEq, LinearAlgebra, ImageShow
import LinearAlgebra

gh = .0005

include("../model/Plant.jl")
function melibeNew(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

u0 = @SVector Float64[
    .2;     # x
    .1;     #y
    0.137e0;   # n
    0.389e0;   # h
    1.0e0;     # Cadisp
    -62.0e0;   # V
]

# set up parameter space
start_p = Float64[Plant.default_params...]
start_p[4] = gh
start_p[17] = -50.0 # Cashift
start_p[16] = -1.6 # xshift
end_p = Float64[Plant.default_params...]
end_p[4] = gh
end_p[17] = -20.0 # Cashift
end_p[16] = .4 # xshift

resolution = 1000 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:3];[gh];Plant.default_params[5:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

# test plot
begin
    fig = Figure()
    ax = Axis(fig[1,1])
    tspan = (0.0, 1000000.0)
    p = [ps[1,1][1:3]; gh; ps[1,1][5:15]; [-1.0, -38.0]]
    prob = ODEProblem(melibeNew, u0, tspan, p)
    @time sol = solve(prob, Rosenbrock23(), saveat = 1.0)
    @time begin
        integ = init(prob, 
        Rosenbrock23(), 
        save_everystep = false, 
        save_start = false, 
        save_end = false)
        while integ.t < 1000000.0
            step!(integ)
        end
    end
    lines!(ax, sol.t, sol[6, :])
    ax2 = Axis3(fig[1,2])
    lines!(ax2, sol[5, :], sol[1, :], sol[2, :])
    fig
end

using DynamicalSystems
p = [ps[1,1][1:3]; gh; ps[1,1][5:15]; [21.0, -30.0]]
sys = ContinuousDynamicalSystem(melibeNew, u0, p)

# test lyapunov
lyapunov = lyapunovspectrum(sys, 1000000)

# Allocate space for first three Lyapunov exponents
lyap_vals = zeros(4, resolution, resolution)

using ProgressMeter
total_points = resolution * resolution
progress = Progress(total_points, desc="Parameter Scan", dt=1)

Threads.@threads for i in 1:resolution
    for j in 1:resolution
        p_ij = ps[i, j]
        sys = ContinuousDynamicalSystem(melibeNew, u0, p_ij)
        # For demonstration, use fewer steps if runtime is excessive
        lyaps = lyapunovspectrum(sys, 500000)
        # Store the first three exponents if available, else pad with zeros
        lyap_vals[1, i, j] = length(lyaps) >= 1 ? lyaps[1] : 0
        lyap_vals[2, i, j] = length(lyaps) >= 2 ? lyaps[2] : 0
        lyap_vals[3, i, j] = length(lyaps) >= 3 ? lyaps[3] : 0
        lyap_vals[4, i, j] = length(lyaps) >= 4 ? lyaps[4] : 0
        next!(progress)
    end
end

# Normalize each exponent channel to [0,1] for plotting
function normv(v, mn, mx)
    mx == mn && return 0.0
    return (v - mn) / (mx - mn)
end

function shape(v, mn, mx, bot, top)
    v = v > top ? top : v
    v = v < bot ? bot : v
    mx == mn && return 0.0
    return (v - bot) / (top - bot) 
end

min1, max1 = extrema(lyap_vals[1, :, :])
min2, max2 = extrema(lyap_vals[2, :, :])
min3, max3 = extrema(lyap_vals[3, :, :])
top3 = lyap_vals[1, :, :] .+ lyap_vals[2, :, :] .+ lyap_vals[3, :, :]
tp = [top > 0 ? top : 0.0 for top in top3]
min4, max4 = extrema(tp)

img = rotl90([RGBf(shape(lyap_vals[1, i, j], min1, max1, .0, 3e-5),
            shape(lyap_vals[2, i, j], min2, max2, .000000, 0.0000025),
            shape(tp[i,j], min4, max4, 0, 0.00001 ))
       for j in 1:resolution, i in 1:resolution])

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
image!(ax, Ca_shifts, x_shifts, rotr90(img))
fig

maximum(lyap_vals[1,:,:])

# save data
using JLD2
@save "lyapunov.jld2" lyap_vals

# loop for next three
