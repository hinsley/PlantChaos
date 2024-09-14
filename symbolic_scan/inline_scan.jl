using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, IterTools, OrdinaryDiffEq, Roots, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/symbolics.jl")
include("../tools/equilibria.jl")

p_base = SVector{length(Plant.default_params)}(Plant.default_params)
u0_base = SVector{6}(Plant.default_state[1], 0., Plant.default_state[2:end]...)
prob_base = ODEProblem(Plant.melibeNew, u0_base, (0.0, 3e5), p_base) # Adjust the time span here.

@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

function condition(out, u, t, integrator)
  Vdot = Plant.dV(integrator.p, u...)
  out[1] = -Vdot

  # out[2] is -Vddot, but we must use finite differencing to
  # calculate it accurately, as the analytical derivative of
  # minf is numerically unstable.
  out[2] = -Plant.numerical_derivative(
    (p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot) -> Vdot,
    u,
    integrator.p,
    1e-4
  )
end

# Define the scan parameters.
ΔCa_values = collect(range(-45.0, 20.0, length=100)) # Adjust this.
ΔVx_values = collect(range(-1.5, -0.5, length=100)) # Adjust this.
param_list = collect(Iterators.product(ΔCa_values, ΔVx_values))
state_list = [Dict( # State machines for symbolic encoder.
  :scs => [], # Signed spike count sequence.
  :count => 0, # Accumulator for spike count within a single burst.
  :last_symbol => Void, # Most recently detected symbol.
  :last2_symbol => Void, # Second-most recently detected symbol.
  :V_sd => 0.0 # The voltage value of the upper saddle equilibrium.
) for i in 1:length(param_list)]

# Adjust the problem for each parameter vector in the scan.
function prob_func(prob, i, repeat)
  # Load the parameters from the pre-defined array.
  ΔCa, ΔVx = param_list[i]

  # Construct new parameter SVector.
  p_new = SVector{length(prob.p)}(prob.p[1:15]..., ΔCa, ΔVx)

  # Recalculate the initial conditions at the upper saddle equilibrium.
  v_eqs = find_zeros(v -> Equilibria.Ca_difference(p_new, v), Plant.xinfinv(p_new, 0.99e0), Plant.xinfinv(p_new, 0.01e0))
  if length(v_eqs) < 3
    v_eq = 0.0
    Ca_eq = 0.0
    x_eq = 0.0
    # error("Unable to find upper saddle equilibrium for parameter vector $i: $p_new. Found only v_eqs $v_eqs.")
  else
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p_new, v_eq)
    x_eq = Plant.xinf(p_new, v_eq)
  end
  u0 = SVector{6}(x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq)

  # Update the state machine with the appropriate V_sd value.
  state_list[i][:V_sd] = v_eq

  # Define the affect! function inside of prob_func as a closure so it can access the problem index i.
  function affect!(integrator, idx)
    if idx == 1
      if integrator.u[6] > state_list[i][:V_sd]
        state_list[i][:count] += 1
        state_list[i][:last2_symbol] = state_list[i][:last_symbol]
        state_list[i][:last_symbol] = Vplus
      else
        push!(
          state_list[i][:scs], # Spike count sequence.
          (state_list[i][:last2_symbol] == Vplus ? -1 : 1) * state_list[i][:count]
        )
        state_list[i][:count] = 0
        state_list[i][:last2_symbol] = state_list[i][:last_symbol]
        state_list[i][:last_symbol] = Vminus
      end
    elseif idx == 2
      state_list[i][:last2_symbol] = state_list[i][:last_symbol]
      state_list[i][:last_symbol] = I
    end
  end

  # Define callback to symbolically encode trajectories.
  cb = VectorContinuousCallback(condition, affect!, nothing, 2)

  # Return the modified problem.
  return remake(
    prob,
    p=p_new,
    u0=u0,
    callback=cb
  )
end

# Define an output function to discard the solution trajectory and only
# return the signed spike count symbolic sequence.
function output_func(sol, i)
  return (state_list[i][:scs], false) # (output, rerun).
end

# Define the EnsembleProblem using prob_func to customize each trajectory.
ensemble_prob = EnsembleProblem(
  prob_base,
  prob_func=prob_func,
  output_func=output_func
)

# Solve the ensemble problem, saving only the signed spike count sequence.
sol = solve(
  ensemble_prob,
  Tsit5(),
  EnsembleThreads(),
  trajectories=length(param_list),
  save_on=false,
  progress=true
)

# Calculate normalized Lempel-Ziv complexity for each solution
lz_complexities = [length(sequence) > 0 ? normalized_LZ_complexity(Vector{Int}(sequence)) : 0.0 for sequence in sol]

# Reshape the LZ complexities into a 2D array
lz_complexity_matrix = reshape(lz_complexities, (length(ΔCa_values), length(ΔVx_values)))

# Create a heatmap of the complexities
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], 
    xlabel="ΔCa", 
    ylabel="ΔVx",
    title="Normalized Lempel-Ziv Complexity of Signed Spike-Count Sequences")

hm = heatmap!(ax, ΔCa_values, ΔVx_values, lz_complexity_matrix, colormap=:viridis)
Colorbar(fig[1, 2], hm, label="LZ Complexity")

# Adjust the layout
colsize!(fig.layout, 1, Aspect(1, 1.0))
colgap!(fig.layout, 20)

# Display the figure
display(fig)

# Optionally, save the figure
# save("lz_complexity_scatter.png", fig)

