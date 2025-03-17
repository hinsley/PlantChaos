using Pkg
Pkg.activate("../lyapunov_old/")
Pkg.instantiate()
using GLMakie, StaticArrays, OrdinaryDiffEq, LinearAlgebra, ForwardDiff
import LinearAlgebra

gh = .002

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

# set up parameter space
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

mutable struct Pars
    p::Vector{Float64}
    temp::Vector{Float64}
    J::Matrix{Float64}
end

mutable struct State
    u::Vector{Float64}
    Q::Matrix{Float64}
end

# Extended system for Lyapunov exponent calculation
function extended_system!(du::State, u::State, p::Pars, t)
    du.u .= melibeNew(u.u, p.p, t)
    ForwardDiff.jacobian!(p.J, x -> melibeNew(x, p.p, t), u.u)
    mul!(du.Q, p.J, u.Q)
end

# Reorthonormalize columns of matrix Q

function condition(u, t, integrator)
    true  # trigger after every successful step
end

function affect!(integrator)
    Q = integrator.u.Q
    temp = integrator.p.temp
    d = size(Q, 1)

    @inbounds begin
        for i in 1:d
            temp .= 0.0  # Reset temp
            for j in 1:i-1
                @. temp += dot(Q[:, i], Q[:, j]) * Q[:, j]
            end
            Q[:, i] .= Q[:, i] - temp
            Q[:, i] ./= norm(Q[:, i])
        end
    end
    reorthonormalize!(integrator.u.Q)
end

cb =  DiscreteCallback(condition, affect!)

# Calculate Lyapunov exponents for a single parameter set
function lyap_exponents(pars::Pars; Ttr = 2000.0, Tdata = 4000.0)
    u0 = State(
        Float64[
            .2;     # x
            .1;     #y
            0.1;   # n
            0.3;   # h
            1.0;     # Ca
            -62.0;   # V
        ], 
        Matrix{Float64}(I, 6, 6)
    )
    prob = ODEProblem(
        extended_system!,
        u0, (0.0, Ttr + Tdata), pars.p
    )
    integ = init(prob,
        RK4(),
        save_everystep = false,
        save_start = false,
        save_end = false,
        abstol = 1e-9, reltol = 1e-9, callback = cb
    )

    # 1) Transient integration until t >= Ttr
    while integ.t < Ttr
        step!(integ)
        if any(x -> abs(x) > 10000, integ.u.Q)
            return integ.u
        end
    end
    t0 = integ.t
    # 2) Measure growth of perturbations from t = Ttr to t = Ttr + Tdata
    λ = zeros(d)
    while integ.t < Ttr + Tdata
        step!(integ)  # advance by Δt
        if any(x -> abs(x) > 10000, integ.u.Q)
            return integ.u
        end
        for i in 1:d
            λ[i] += log(norm(integ.u.Q[:, i]))
        end
    end
    return λ ./ (integ.t - t0)
end

# test the function
p = vcat(Plant.default_params[1:3], gh, Plant.default_params[5:15], [-1.0, -38.0])
ps = Pars(p, zeros(6), zeros(6, 6))
@time lyap_exponents(ps; Ttr = 10000, Tdata = 10000)

# Allocate storage for spectra over the 2D grid
spectra = Array{Float64}(undef, 6, resolution, resolution)

# Loop through parameter sets, compute, and store results
for i in 1:resolution
    for j in 1:resolution
        spectra[:, i, j] = lyap_exponents(ps[i, j])
    end
end

# Now, for instance, you could visualize an exponent of interest:
fig = Figure()
ax = Axis(fig[1,1], title="First Lyapunov Exponent")
heatmap!(ax, x_shifts, Ca_shifts, spectra[1, :, :]')
fig