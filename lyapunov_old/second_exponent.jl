using Pkg
Pkg.activate("../lyapunov_old/")
using GLMakie
using StaticArrays, ProgressMeter

include("../model/Plant.jl")

function melibe5(u::AbstractArray{T}, p, t) where T
    # TODO: REVERT THIS! u[1], u[2], u[3], u[4], u[5], u[6], u[7] = u

    # du1 = dx(p, u[1] V)
    # du2 = dy(y, V)
    # du3 = dn(n, V)
    # du4 = dh(h, V)
    # du5 = dCa(p, Ca, u[1] V)
    # du6 = dV(p, u[1] y, n, h, Ca, V, Isyn)
    # du7 = 0.0e0
    # return @SVector T[du1, du2, du3, du4, du5, du6, du7]
    return @SVector T[
        Plant.dx(p, u[1], u[5]),
        Plant.dn(u[2], u[5]),
        Plant.dh(u[3], u[5]),
        Plant.dCa(p, u[4], u[1], u[5]),
        Plant.dV(p, u[1], 0.0, u[2], u[3], u[4], u[5], 0.0),
    ]
end

u0 = @SVector Float32[
    .2;     # x
    0.137e0;   # n
    0.389e0;   # h
    1.0e0;     # Ca
    -62.0e0;   # V
]

include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -60.0 # Cashift
start_p[16] = -2 # xshift
end_p = [Plant.default_params...]
end_p[17] = 0.0 # Cashift
end_p[16] = .5 # xshift

start_p[17] = 15 # Cashift
start_p[16] = -2.8 # xshift
end_p = [Plant.default_params...]
end_p[17] = 35 # Cashift
end_p[16] = -2.4 # xshift

resolution = 15 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]


using DynamicalSystems, ChaosTools
import LinearAlgebra
sys = CoupledODEs(melibe5, u0, ps[1])

N = Threads.nthreads()

begin
    lyaparray = Array{Float64}(undef, resolution, resolution)
    lyap2array = Array{Float64}(undef, resolution, resolution)
    lyap3array = Array{Float64}(undef, resolution, resolution)
    lyap4array = Array{Float64}(undef, resolution, resolution)
    lyap5array = Array{Float64}(undef, resolution, resolution)
    
    # Initialize a progress bar
    p = Progress(resolution^2, 1, "Computing Lyapunov Exponents: ", 50)

    Threads.@threads for i in 1:resolution
        for j in 1:resolution
            sys = CoupledODEs(melibe5, u0, ps[i,j])
            ls = lyapunovspectrum(sys, 1000000, 5; Ttr = 500000)
            lyaparray[i,j] = ls[1]
            lyap2array[i,j] = ls[2]
            lyap3array[i,j] = ls[3]
            lyap4array[i,j] = ls[4]
            lyap5array[i,j] = ls[5]
            next!(p)
        end
    end
end

scale(x, min, max) = x < min ? 0.0 : x > max ? 1.0 : (x - min) / (max - min) 
using ImageShow
lcarr = rotl90([Makie.RGB(
    let a = lyaparray'[i,j]
        abs2(scale(a, 0, 0.00021))
    end, let a = lyap2array'[i,j]
        sqrt(scale(-a, 0.0000, 0.0015))
    end, let a = lyap2array'[i,j]
        sqrt(sqrt(scale(-a, .00, 0.00019)))
    end
) for i in 1:resolution, j in 1:resolution])

using JLD2
save("lyapunov_3color.jld2", "lcarr", lcarr)