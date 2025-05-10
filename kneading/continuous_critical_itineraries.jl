# Compute the critical itineraries for the SiN model assuming Γ_SD^- and Γ_SD^+
# have the same image under the return map to the reinsertion loop and the
# return map is continuous (we are to the right of the homSF curve).

using Pkg
Pkg.activate("./kneading")
Pkg.instantiate()

using GLMakie, OrdinaryDiffEq, StaticArrays, Roots, NonlinearSolve
using Interpolations

include("../model/Plant.jl")
using .Plant # Make functions from Plant module available
include("../tools/equilibria.jl") # Included Equilibria

# How many iterates of the return map to compute for the critical itineraries.
RETURN_ITERATES = 10

# Define the parameter values to sweep over.
# Δx = 0.01
# ΔCas = range(0.0, 40.0, step=0.01)
# Import the model file to get default parameters.

# Parameters from map_vis/burst_stabilization.jl.
# The 16th element corresponds to Δx, and the 17th to ΔCa_shift.
Δx = -0.81
ΔCa = -41.0

base_params = Plant.default_params[1:15]
p_svector = SVector{17, Float64}([base_params..., Δx, ΔCa])

p = Observable(p_svector)

# Compute the equilibria of the slow subsystem.
eqs = find_zeros(v -> Equilibria.Ca_difference(p[], v), Plant.xinfinv(p[], 0.99e0), Plant.xinfinv(p[], 0.01e0))

# Compute the location of the saddle-focus equilibrium (SF).
V_eq = eqs[2]
Ca_eq = Equilibria.Ca_null_Ca(p[], V_eq)
x_eq = Plant.xinf(p[], V_eq)
n_eq = Plant.ninf(V_eq)
h_eq = Plant.hinf(V_eq)
SF_eq = @SVector [x_eq, 0.0, n_eq, h_eq, V_eq, Ca_eq]

# Compute the location of the upper saddle equilibrium SD.
V_eq = eqs[1]
Ca_eq = Equilibria.Ca_null_Ca(p[], V_eq)
x_eq = Plant.xinf(p[], V_eq)
n_eq = Plant.ninf(V_eq)
h_eq = Plant.hinf(V_eq)
SD_eq = @SVector [x_eq, 0.0, n_eq, h_eq, V_eq, Ca_eq]

include("../map_vis/return_map_utils.jl")

# Obtain an initial condition for Γ_SD^-.
V_eqs = find_zeros(v -> Equilibria.Ca_difference(p[], v), Plant.xinfinv(p[], 0.99e0), Plant.xinfinv(p[], 0.01e0))
if length(V_eqs) < 3
    return fill(NaN, 6)
end
v_eq = V_eqs[3]
Ca_eq = Equilibria.Ca_null_Ca(p[], v_eq)
x_eq = Plant.xinf(p[], v_eq)
saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
jac = ForwardDiff.jacobian(u -> melibeNew(u,p[],0), saddle)
vals,vecs = eigen(jac)
_,i = findmax(real.(vals))
eps = .001
Γ_SD_minus0 = SVector{6}(saddle .- eps .* real.(vecs)[:,i])

# Condition for the callback: du[5] (Ca derivative) crossing zero from negative to positive.
dCas = Float64[]
function condition(u, t, integrator)
    du = get_du(integrator)
    push!(dCas, du[5])
    return du[5] # Trigger when this is zero.
end

function affect!(integrator)
    terminate!(integrator)
end

# affect_pos! (for positive-to-negative crossings of condition) should do nothing.
# affect_neg! (for negative-to-positive crossings of condition) should call affect! (terminate).
# Since condition returns du[5], this will trigger at a Ca minimum.
cb = ContinuousCallback(condition, affect_pos! = nothing, affect_neg! = affect!)

# Set up and solve the ODE problem
tspan = (0.0, 1e5)  # Set a sufficiently long time span
prob = ODEProblem(melibeNew, Γ_SD_minus0, tspan, p[])
sol = solve(prob, Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8)#, save_everystep=false)

# Store only the endpoint at the calcium minimum
Γ_SD_minus_Ca_min = sol.u[end]

# Plot the solution in the Ca-x phase plane.
fig = Figure(size=(1000, 1000))
ax = Axis(fig[1, 1], xlabel="Calcium (Ca)", ylabel="x", title="Ca-x Phase Plane")
lines!(ax, [u[5] for u in sol.u], [u[1] for u in sol.u])
display(fig)
