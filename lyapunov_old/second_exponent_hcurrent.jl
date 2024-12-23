using Pkg
Pkg.activate("../lyapunov_old/")
Pkg.instantiate()
using GLMakie
using StaticArrays, ProgressMeter

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
    1.0e0;     # Ca
    -62.0e0;   # V
]

#include("../tools/equilibria.jl")
gh = .0015
start_p = Float64[Plant.default_params...]
start_p[4] = gh
start_p[17] = -60.0 # Cashift
start_p[16] = -2 # xshift
end_p = Float64[Plant.default_params...]
end_p[4] = gh
end_p[17] = 0.0 # Cashift
end_p[16] = .5 # xshift

resolution = 2 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:3];[gh];Plant.default_params[5:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

using DynamicalSystems, ChaosTools
import LinearAlgebra

using OrdinaryDiffEq
p = vcat(Plant.default_params[1:3], gh, Plant.default_params[5:15], [-1.0, -38.0])
diffeq = (alg = RK4(), abstol = 1e-8, reltol = 1e-8)
sys = CoupledODEs(melibeNew, u0, p; diffeq)
tands = TangentDynamicalSystem(sys)

#test trajectory
begin
    prob = ODEProblem(melibeNew, u0, (0.0, 1000000.0), p)
    sol = solve(prob, RK4(), abstol = 1e-8, reltol = 1e-8)

    #plot test trajectory
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, sol.t, sol[2, :], color = :blue)
    ax2 = Axis3(fig[1, 2])
    lines!(ax2, sol[1, :], sol[5, :], sol[6, :], color = :blue)
    fig
end

lyapunovspectrum(sys, 1000, 5, Ttr = 0)

# test lyapunov spectrum
Δt = 1.0
Ttr = 449
u0 = current_state(tands)
reinit!(tands, u0)

function _buffered_qr(B::SMatrix, Y) # Y are the deviations
    Q, R = LinearAlgebra.qr(Y)
    return Q, R
end
function _buffered_qr(B::Matrix, Y) # Y are the deviations
    B .= Y
    Q, R = LinearAlgebra.qr!(B)
    return Q, R
end
function set_Q_as_deviations!(tands::TangentDynamicalSystem{true}, Q)
    devs = current_deviations(tands) # it is a view
    if size(Q) ≠ size(devs)
        copyto!(devs, LinearAlgebra.I)
        LinearAlgebra.lmul!(Q, devs)
        set_deviations!(tands, devs)
    else
        set_deviations!(tands, Q)
    end
end

function set_Q_as_deviations!(tands::TangentDynamicalSystem{false}, Q)
    # here `devs` is a static vector
    devs = current_deviations(tands)
    ks = axes(devs, 2) # it is a `StaticArrays.SOneTo(k)`
    set_deviations!(tands, Q[:, ks])
end

B = copy(current_deviations(tands))
if Ttr > 0 # This is useful to start orienting the deviation vectors
    t0 = current_time(tands)
    while (current_time(tands) < t0 + Ttr)
        if any(x -> abs(x) > 1000, current_deviations(tands))
            g = false
        end
        step!(tands, Δt)
        Q, R = _buffered_qr(B, current_deviations(tands))
        set_Q_as_deviations!(tands, Q)
    end
end

k = size(current_deviations(tands))[2]
λ = zeros(eltype(current_deviations(tands)), k)
t0 = current_time(tands)

for i in 1:N
    step!(tands, Δt)
    Q, R = _buffered_qr(B, current_deviations(tands))
    for j in 1:k
        @inbounds λ[j] += log(abs(R[j,j]))
    end
    set_Q_as_deviations!(tands, Q)
    ProgressMeter.update!(progress, i)
end





import LinearAlgebra

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
            sys = CoupledODEs(melibeNew, u0, ps[i,j])
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
        sqrt(scale(a, 0.0000, 0.0015))
    end, let a = lyap2array'[i,j]
        sqrt(sqrt(scale(-a, .00, 0.00019)))
    end
) for i in 1:resolution, j in 1:resolution])

using JLD2
save("lyapunov_3color.jld2", "lcarr", lcarr)