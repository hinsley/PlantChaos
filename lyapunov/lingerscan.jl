# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.
using Pkg
Pkg.activate("./lyapunov")
using GLMakie, StaticArrays
using LinearAlgebra: norm
using OrdinaryDiffEq
include("../model/Plant.jl")

# set up parameter space for scan
start_p = Float64[Plant.default_params...]
start_p[17] = -41
#start_p[17] = -70.0 # Cashift
start_p[16] = -1.4# # xshift
#start_p[16] = -3. # xshift
end_p = Float64[Plant.default_params...]
end_p[17] = -30.#150.0 # Cashift
#end_p[17] = 50.#10.0 # Cashift
end_p[16] = -1.15#1. # xshift
#end_p[16] = 1.#1. # xshift

resolution = 200 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)

_ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

mutable struct PDat
    const p::Vector{Float64}
    const eq::Vector{Float64}
    count::Int
    linger::Vector{Float64}
end

# find first equilibria
initial_eq_guess = include("./eq_guess.jl")
function melibeNew(u::AbstractArray{T}, p) where T
    dx = Plant.dx(p, u[1], u[5])
    dn = Plant.dn(u[2], u[5])
    dh = Plant.dh(u[3], u[5])
    dCa = Plant.dCa(p, u[4], u[1], u[5])
    dV = Plant.dV(p, u[1], 0, u[2], u[3], u[4], u[5])
    return @SVector T[dx, dn, dh, dCa, dV]
end
using NonlinearSolve
eqprob = NonlinearProblem(melibeNew, initial_eq_guess, _ps[1])
eqsol = solve(eqprob, NewtonRaphson())
# solve equilibria for all parameters
eqs = Array{Vector{Float64}}(undef, resolution, resolution)
eqs[1,1] = eqsol.u
for i in 1:resolution, j in 1:resolution
    i*j==1 && continue
    closest_eq = i == 1 ? eqs[i,j-1] : eqs[i-1,j]
    eqprob = NonlinearProblem(melibeNew, closest_eq, _ps[i,j])
    eqsol = solve(eqprob, NewtonRaphson())
    eqs[i,j] = eqsol.u
end
eigs = Array{Vector{ComplexF64}}(undef, resolution, resolution)

using ForwardDiff
function jacobian(f, x, p)
    _f = x -> f(x, p)
    ForwardDiff.jacobian(_f, x)
end
J =jacobian(melibeNew, eqs[1,1], _ps[1,1])
using LinearAlgebra: eigen

eig = eigen(J)
imixs = findall(x -> imag(x) != 0, eig.values)
e1 = sum(eig.vectors[:,imixs], dims = 2)
e2 = sum(im.*eig.vectors[:,imixs], dims = 2)

rank([e1 e2])

for i in 1:resolution, j in 1:resolution
    eigs[i,j] = eigvals(jacobian(melibeNew, eqs[i,j], _ps[i,j]))
end


# set up systems
u0 = @SVector [
    0.8e0;     # x
    0.137e0;   # n
    0.389e0;   # h
    0.8e0;     # Ca
    -62.0e0;   # V
    0e0;       #linger
]

using LinearAlgebra: norm
function melibeNew(u::AbstractArray{T}, ps, t) where T
    p = ps[1]
    eq = 
    return @SVector T[
        Plant.dx(p, u[1], u[5]),
        Plant.dn(u[2], u[5]),
        Plant.dh(u[3], u[5]),
        Plant.dCa(p, u[4], u[1], u[5]),
        Plant.dV(p, u[1], 0, u[2], u[3], u[4], u[5]),
    ]
end

prob = ODEProblem{false}(melibeNew, u0, (0e0, 1e6), ps[1], saveat = 10.)
sol = solve(prob, BS3(), dtmax = 6.0, abstol = 1e-7, reltol = 1e-7)
prob_func(prob, i, repeat) = remake(prob, p = ps[i])
output_func(sol,i) = (sol.prob.p[2], false)

function condition(u, t, integrator)
    p = integrator.p
    # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
    if u[6] > -20 return -1f0 end
    (t < 50) ? 1f0 : -(p[15] * (p[13] * u[1] * (p[12] - u[6] + p[17]) - u[5]))
end
funtion affect!(integrator)
    terminate!(integrator) # Stop the solver
cb = ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions = (false,false))


monteprob = EnsembleProblem(prob, prob_func=prob_func,output_func= output_func, safetycopy=false)

#fig

#save(homedir() * "/Dropbox/lyapunov_pd.png",fig)

using CSV
CSV.write(homedir() * "/Dropbox/lyapunov_pd.csv", CSV.Tables.table(lyaparray2))
