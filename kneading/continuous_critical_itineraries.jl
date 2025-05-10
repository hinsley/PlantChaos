# Compute the critical itineraries for the SiN model assuming Γ_SD^- and Γ_SD^+
# have the same image under the return map to the reinsertion loop and the
# return map is continuous (we are to the right of the homSF curve).

using Pkg
Pkg.activate("./kneading")
Pkg.instantiate()

using GLMakie, OrdinaryDiffEq, StaticArrays, Roots, NonlinearSolve
using Interpolations, SciMLSensitivity, ForwardDiff

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
Δx = -1.0
ΔCa = -32.0

base_params = Plant.default_params[1:15]
p_svector = SVector{17, Float64}([base_params..., Δx, ΔCa])

p = Observable(p_svector)

# Compute the equilibria of the slow subsystem.
V_eqs = find_zeros(v -> Equilibria.Ca_difference(p[], v), Plant.xinfinv(p[], 0.99e0), Plant.xinfinv(p[], 0.01e0))

# Compute the location of the saddle-focus equilibrium (SF).
V_eq_SF = V_eqs[2]
Ca_eq_SF = Equilibria.Ca_null_Ca(p[], V_eq_SF)
x_eq_SF = Plant.xinf(p[], V_eq_SF)
n_eq_SF = Plant.ninf(V_eq_SF)
h_eq_SF = Plant.hinf(V_eq_SF)
SF_eq = @SVector [x_eq_SF, 0.0, n_eq_SF, h_eq_SF, Ca_eq_SF, V_eq_SF]

# Compute the location of the upper saddle equilibrium SD.
V_eq_SD = V_eqs[3]
Ca_eq_SD = Equilibria.Ca_null_Ca(p[], V_eq_SD)
x_eq_SD = Plant.xinf(p[], V_eq_SD)
n_eq_SD = Plant.ninf(V_eq_SD)
h_eq_SD = Plant.hinf(V_eq_SD)
SD_eq = @SVector [x_eq_SD, 0.0, n_eq_SD, h_eq_SD, Ca_eq_SD, V_eq_SD]

include("../map_vis/return_map_utils.jl")

# Obtain an initial condition for Γ_SD^-.
jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p[],0), SD_eq)
vals,vecs = eigen(jac)
_,i = findmax(real.(vals))
eps = .001
Γ_SD_minus0 = SVector{6}(SD_eq .- eps .* real.(vecs)[:,i])

# Condition for the callback: du[5] (Ca derivative) crossing zero from negative to positive.
function condition(u, t, integrator)
  if t < 1e3 || u[1] > x_eq_SF
    return 1.0
  end
  dCa = Plant.melibeNew(u, integrator.p, integrator.t)[5]
  return dCa
end

function affect!(integrator)
  terminate!(integrator)
end

cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)

# Set up and solve the ODE problem.
tspan = (0.0, 1e5) # Set a sufficiently long time span.
prob = ODEProblem(melibeNew, Γ_SD_minus0, tspan, p[])
sol = solve(prob, Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

# Store only the endpoint at the calcium minimum.
Γ_SD_minus_Ca_min_V = sol.u[end][6]
Γ_SD_minus_Ca_min = sol.u[end][5]

# Generate a range of initial conditions along the Ca nullcline between
# SF and Γ_SD_minus_Ca_min.
map_resolution = 500
function Ca_x_eq(p)
    V_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
    Ca_eq = Ca_null_Ca(p, V_eq)
    x_eq = Plant.xinf(p, V_eq)
    return V_eq, Ca_eq, x_eq
end
function Ca_null_Ca(p, V)
    return p[13]*Plant.xinf(p, V)*(p[12]-V+p[17])
end
Vs = range(V_eq_SF, Γ_SD_minus_Ca_min_V, length=map_resolution)
x_offset = 1f-4 # Offset from xinf to avoid numerical issues.
u0s = [
  SVector{6, Float64}([
    Plant.xinf(p[], V)-x_offset,
    SF_eq[2:4]...,
    Ca_null_Ca(p[], V),
    V])
  for V in Vs
]
Ca0s = [u0[5] for u0 in u0s]

# Calculate the return map.
template_prob = ODEProblem(Plant.melibeNew, u0s[1], tspan, p[])
function prob_func(prob, i, repeat)
    remake(prob, u0 = u0s[i])
end
ensemble_prob = EnsembleProblem(template_prob, prob_func = prob_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = length(u0s), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

return_Ca_mins = [s.u[end][5] for s in ensemble_sol]

# Plot the return map as a scatter plot of Ca0 vs return Ca min.
fig = Figure(size=(800, 1200)) # Adjusted figure size for two plots.
ax_return_map = Axis(fig[1, 1], title="Return Map", xlabel="Initial Ca (Ca₀)", ylabel="Return Ca at Minimum", aspect=DataAspect())
scatter!(ax_return_map, Ca0s, return_Ca_mins, markersize = 4)

# Add identity line
all_ca_values = vcat(Ca0s, return_Ca_mins)
min_ca, max_ca = extrema(all_ca_values)
lines!(ax_return_map, [min_ca, max_ca], [min_ca, max_ca], color=:gray, linestyle=:dash)

# Plot all trajectories from the ensemble solve.
ax_trajectories = Axis(fig[2, 1], title="Trajectories (Ca vs x)", xlabel="Ca", ylabel="x")
for s in ensemble_sol
    # Extract Ca (index 5) and x (index 1) for each point in the trajectory.
    x_vals = [pt[1] for pt in s.u]
    ca_vals = [pt[5] for pt in s.u]
    lines!(ax_trajectories, ca_vals, x_vals) # Makie will cycle colors.
end

# Add the trajectory for Γ_SD_minus0 (solution in sol) in red.
if @isdefined(sol) && !isempty(sol.u)
    x_vals_gamma_sd_minus = [pt[1] for pt in sol.u]
    ca_vals_gamma_sd_minus = [pt[5] for pt in sol.u]
    lines!(ax_trajectories, ca_vals_gamma_sd_minus, x_vals_gamma_sd_minus, color = :red, linewidth = 2, linestyle = :dot, label = "Γ_SD⁻ trajectory")
else
    println("Warning: `sol` for Γ_SD_minus0 is not defined or empty, cannot plot its trajectory.")
end

# Obtain the first guess at the calcium value for the critical point associated
# with the 1-spike preimage of T.
# Find the first local maximum in the return map.
function find_first_local_maximum(x, y)
    for i in 2:(length(y)-1)
        if y[i] > y[i-1] && y[i] > y[i+1]
            return i
        end
    end
    return nothing  # Return nothing if no local maximum is found.
end

# Get the index of the first local maximum in return_Ca_mins.
first_max_index = find_first_local_maximum(Ca0s, return_Ca_mins)

if first_max_index !== nothing
    println("T found at index: ", first_max_index)
    println("Corresponding Ca₀ value: ", Ca0s[first_max_index])
    println("Return Ca value: ", return_Ca_mins[first_max_index])
    
    # Mark the first local maximum on the return map plot.
    scatter!(ax_return_map, [Ca0s[first_max_index]], [return_Ca_mins[first_max_index]], 
             color = :red, markersize = 8, marker = :star5)
    
    # Mark the initial condition in the trajectories plot.
    scatter!(ax_trajectories, [Ca0s[first_max_index]], [u0s[first_max_index][1]], 
             color = :red, markersize = 8, marker = :star5)
    
    # Plot the trajectory associated with the maximum in extra large sized lines.
    max_trajectory = ensemble_sol[first_max_index]
    x_vals_max = [pt[1] for pt in max_trajectory.u]
    ca_vals_max = [pt[5] for pt in max_trajectory.u]
    lines!(ax_trajectories, ca_vals_max, x_vals_max, 
           color = :red, linewidth = 4, linestyle = :solid, 
           label = "1-spike preimage of T trajectory")
    axislegend(ax_trajectories, position = (:right, :bottom))
else
    println("No local maximum found in the return map.")
end

display(fig)

# Compute the itinerary of the critical point associated with the 1-spike
# preimage of T.

function f(V0, p)
  # Compute the return map at an initial voltage value V0.
  function _condition(u, t, integrator)
    if t < 1e3 || u[1] > x_eq_SF
      return 1.0
    end
    dCa = Plant.melibeNew(u, p, t)[5]
    return dCa
  end
  function _affect!(integrator)
    terminate!(integrator)
  end
  _cb = ContinuousCallback(_condition, _affect!)
  _prob = remake(prob, u0 = SVector{6}([
    Plant.xinf(p, V0)-x_offset,
    SF_eq[2:4]...,
    Ca_null_Ca(p, V0),
    V0
  ]))
  _sol = solve(
    _prob,
    Tsit5(),
    callback=_cb,
    abstol=1e-8,
    reltol=1e-8,
    save_everystep=false
  )
  return _sol.u[end][6]
end

# Compute the first and second derivatives of f with respect to V0.
function differentiate_f(p, V0)
  # Define a function of V0 only, with p fixed from the outer scope.
  # This represents f(V_val; p) as a function of V_val.
  f_with_fixed_p = V_val -> f(V_val, p)

  # Define a function that computes the first derivative of f_with_fixed_p at any given point.
  # This function, let's call it g(x), returns f_with_fixed_p'(x).
  first_derivative_func = x -> ForwardDiff.derivative(f_with_fixed_p, x)

  # Compute the value of the first derivative at the specific point V0.
  dV = first_derivative_func(V0)
  
  # Compute the value of the second derivative at the specific point V0.
  d2V = ForwardDiff.derivative(first_derivative_func, V0)
  
  return dV, d2V
end

V0 = Vs[first_max_index]
newton_iters = 20
for i in 1:newton_iters
  T_dV, T_d2V = differentiate_f(p[], V0)
  V0 = V0 - T_dV / T_d2V
end
T_Ca0 = Ca_null_Ca(p[], V0)

# Run the trajectory associated with the refined critical point.
__prob = remake(prob, u0 = SVector{6}([
  Plant.xinf(p[], V0)-x_offset,
  SF_eq[2:4]...,
  T_Ca0,
  V0
]))
__sol = solve(__prob, Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

# Plot the critical point on the return map.
# scatter!(ax_return_map, [T_Ca0], [return_Ca_mins[first_max_index]], 
#          color = :blue, markersize = 8, marker = :star5)
scatter!(ax_return_map, [__sol.u[1][5]], [__sol.u[end][5]], 
         color = :blue, markersize = 8, marker = :star5)

# Plot the trajectory in the trajectories plot.
x_vals_critical = [pt[1] for pt in __sol.u]
ca_vals_critical = [pt[5] for pt in __sol.u]
lines!(ax_trajectories, ca_vals_critical, x_vals_critical, 
        color = :blue, linewidth = 4, linestyle = :solid, 
        label = "Critical trajectory")

display(fig)