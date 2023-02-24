using DifferentialEquations
using DiffEqGPU
using LinearAlgebra
using Roots
using StaticArrays

include("../model/Plant.jl")

state = Plant.default_state
u0 = @SVector Float32[0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0]

IKCa(p, V) = p[2]*Plant.hinf(V)*Plant.minf(V)^3.0f0*(p[8]-V) + p[3]*Plant.ninf(V)^4.0f0*(p[9]-V) + p[6]*Plant.xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)
xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end

function Ca_null_Ca(p, v)
    return p[13]*Plant.xinf(p, v)*(p[12]-v+p[17])
end

# The function which must be minimized to find the equilibrium voltage.
function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

# Finds the equilibrium in the slow subsystem.
function Ca_x_eq(p)
    v_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
    Ca_eq = Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

Δx = -2.2f0
ΔCa = -38.0f0
print("Δx: ")
Δx = parse(Float32, readline())
print("ΔCa: ")
ΔCa = parse(Float32, readline())
tspan = (0.0f0, 1.0f6)

p = @SVector Float32[
    Plant.default_params[1],  # Cₘ
    Plant.default_params[2],  # gI
    Plant.default_params[3],  # gK
    Plant.default_params[4],  # gₕ
    Plant.default_params[5],  # gL
    Plant.default_params[6],  # gT
    Plant.default_params[7],  # gKCa
    Plant.default_params[8],  # EI
    Plant.default_params[9],  # EK
    Plant.default_params[10], # Eₕ
    Plant.default_params[11], # EL
    Plant.default_params[12], # ECa
    Plant.default_params[13], # Kc
    Plant.default_params[14], # τₓ
    Plant.default_params[15], # ρ
    Δx,                          # Δx
    ΔCa                          # ΔCa
]

function initial_conditions(p)
    try
        v_eq, Ca_eq, x_eq = Ca_x_eq(p)
        u0 = @SVector Float32[x_eq, state[2], state[3], state[4], Ca_eq-0.2, v_eq, state[7]]
    catch e
        # This should trigger when the Ca_x_eq function fails to converge.
        u0 = state
    end
    return u0
end

prob = ODEProblem{false}(Plant.melibeNew, u0, tspan, p)
prob_func(prob, i, repeat) = remake(prob, u0=initial_conditions(prob.p))

monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
#@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), adaptive=true, trajectories=1, dt=1.0f0, abstol=1e-6, reltol=1e-6)
#@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), adaptive=false, trajectories=1, dt=3.0f0, abstol=1f-6, reltol=1f-6)
@time sol = solve(monteprob, Tsit5(), EnsembleThreads(), adaptive=false, trajectories=1, dt=1.0f0, abstol=1f-6, reltol=1f-6, verbose=false)

using Plots

min_Ca = min(sol[1](sol[1].t, idxs=(5))...)
max_Ca = max(sol[1](sol[1].t, idxs=(5))...)
min_x = min(sol[1](sol[1].t, idxs=(1))...)
max_x = max(sol[1](sol[1].t, idxs=(1))...)
min_Ca = 0.0f0
max_Ca = 1.0f0
min_x = 0.01f0
max_x = 0.99f0
if isnan(min_Ca) || isnan(max_Ca)
    min_Ca = 0.0f0
    max_Ca = -1.0f0
end
if isnan(min_x) || isnan(max_x)
    min_x = 0.01f0
    max_x = 0.99f0
end
plt = plot(sol, idxs=(5, 1), lw=0.2, legend=false, xlims=(min_Ca, max_Ca), ylims=(min_x, max_x), dpi=500, size=(1280, 720), xlabel="Ca", ylabel="x", title="\$\\Delta_x = $(Δx), \\Delta_{Ca} = $(ΔCa)\$")
v_eq, Ca_eq, x_eq = Ca_x_eq(p)
V_range = nothing
try
    V_range = range(xinfinv(p, min_x), xinfinv(p, max_x), length=1000)
catch e
    V_range = range(-70, 20, length=1000)
end
plot!(plt, [Ca_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range])
plot!(plt, [x_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range])
scatter!(plt, [Ca_eq], [x_eq])

display(plt)