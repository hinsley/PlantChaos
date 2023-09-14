include("./Plant.jl")
using OrdinaryDiffEq
using BenchmarkTools

u0 = Plant.default_state
p = Plant.default_params
tspan = (0f0, 1f6)

prob = ODEProblem(Plant.melibeNew,u0,tspan,p)
@btime solve(prob, RK4());

using DiffEqGPU, CUDA, StaticArrays
prob_func = (prob, i, repeat) -> remake(prob, u0 = u0)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 1000, adaptive = false, dt = 1f0)
CUDA.reclaim()