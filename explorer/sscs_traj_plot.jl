include("../model/Plant.jl")
p = vcat(Plant.default_params, [-.35, -47.2])
u0 = Plant.default_state
tspan = (0.0, 100000.0)
using OrdinaryDiffEq, GLMakie




prob = ODEProblem(Plant.melibeNew, u0, tspan, p)
sol = solve(prob, RK4())


fig = Figure(resolution = (1500, 100))
ax = Axis(fig[1,1], xlabel = "t", ylabel = "V")

