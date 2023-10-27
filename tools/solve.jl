module Solve export solve

using OrdinaryDiffEq

include("../model/Plant.jl")

function solve(u0s=[Plant.default_state], ps=[Plant.default_params], tspans=[(0.0f0, 1.0f4)])
    # Solve the system for the given initial conditions and parameters
    # u0s: initial conditions
    # ps: parameters
    # tspans: time spans
    prob = ODEProblem(Plant.melibeNew, u0s[1], tspans[1], ps[1])
    prob_func(prob, i, repeat) = remake(prob, u0=u0s[i], p=ps[i])
    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    
    @time sol = DifferentialEquations.solve(monteprob, RK4(), EnsembleThreads(), trajectories=length(u0s), abstol=1e-6, reltol=1e-6, verbose=false)
end

end