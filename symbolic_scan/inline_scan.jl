using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, Colors, IterTools, OrdinaryDiffEq, Roots, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/equilibria.jl")
include("../tools/symbolics.jl")

p_base = SVector{length(Plant.default_params)}(Plant.default_params)
u0_base = SVector{6}(Plant.default_state[1], 0., Plant.default_state[2:end]...)
prob_base = ODEProblem(Plant.melibeNew, u0_base, (0.0, 3e5), p_base) # Adjust the time span here.

@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

transient_time = 1e2 # Time to wait before beginning to detect events.
function condition(out, u, t, integrator)
  if t < transient_time
    out[1] = 0.0
    out[2] = 0.0
  else
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
end

# Define the scan parameters.
ΔCa_values = collect(range(-60.0, 100.0, length=500)) # Adjust this.
ΔVx_values = collect(range(-4.0, 1.0, length=500)) # Adjust this.
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
  p_new = SVector{length(prob.p)}(prob.p[1:15]..., ΔVx, ΔCa)

  # Recalculate the initial conditions at the upper saddle equilibrium.
  v_eqs = find_zeros(v -> Equilibria.Ca_difference(p_new, v), Plant.xinfinv(p_new, 0.99), Plant.xinfinv(p_new, 0.01))
  if length(v_eqs) < 3
    v_eqs = [0.0, 0.0, 0.0] # TODO: Codesmell. This is a bad approximation and may produce artifacts.
    # error("Unable to find upper saddle equilibrium for parameter vector $i: $p_new. Found only v_eqs $v_eqs.")
  end
  v_eq = v_eqs[3]
  Ca_eq = Equilibria.Ca_null_Ca(p_new, v_eq)
  x_eq = Plant.xinf(p_new, v_eq)
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
@time sol = solve(
  ensemble_prob,
  Tsit5(),
  abstol=3e-6,
  reltol=3e-6,
  EnsembleThreads(),
  trajectories=length(param_list),
  save_on=false,
  progress=true
)

# Calculate & render normalized Lempel-Ziv complexity heatmap.
begin
  # Calculate normalized Lempel-Ziv complexity for each solution.
  last_n = 5000 # Length of tail of signed spike-count sequences to use for LZ complexity calculation.
  # Before calculating normalized LZ complexities, I discard any sequence comprising more than 99.3% subthreshold oscillations.
  # @time lz_complexities = [
  #     (count(x -> x == 0, sequence) / length(sequence) > 0.993) ? 0.0 : 
  #     (length(sequence) >= last_n ? normalized_LZ_complexity(Vector{Int}(sequence[end-last_n+1:end])) : 
  #     (length(sequence) > 0 ? normalized_LZ_complexity(Vector{Int}(sequence)) : 0.0)) 
  #     for sequence in sol
  # ]

  # Reshape the LZ complexities into a 2D array.
  lz_complexity_matrix = reshape(lz_complexities, (length(ΔCa_values), length(ΔVx_values)))

  # Create a heatmap of the complexities.
  fig_lz = Figure(size=(800, 700))
  ax = Axis(fig_lz[1, 1], 
      xlabel="ΔCa", 
      ylabel="ΔVx",
      title="Normalized Lempel-Ziv Complexity of Signed Spike-Count Sequences")

  color_gradient = cgrad([
    Colors.RGB(0.0, 0.0, 0.0),
    Colors.RGB(0.0, 0.15, 0.6),
    Colors.RGB(1.0, 0.0, 0.0)
  ], [0.17, 0.19, 0.2, 0.21, 0.22, 0.3, 0.31]) # for last_n = 5000 on 500x500 grid
  # ], [0.18, 0.21, 0.23]) # for last_n = 2000
  # ], [0.061, 0.065, 0.18, 0.39, 0.4]) # for last_n = 10000
  # ], [0.0, 0.29, 0.3, 0.5]) # for last_n = 500, I think? Maybe 1000.
  hm = heatmap!(ax, ΔCa_values, ΔVx_values, lz_complexity_matrix, colormap=color_gradient)
  Colorbar(fig_lz[1, 2], hm, label="LZ Complexity")

  # Adjust the layout.
  colsize!(fig_lz.layout, 1, Aspect(1, 1.0))
  colgap!(fig_lz.layout, 20)

  # Display the figure.
  display(fig_lz)

  # Optionally, save the figure.
  # save("lz_complexity_heatmap.png", fig)
end

# Calculate & render conditional block entropy heatmap.
begin
  # Calculate conditional block entropy for each solution.
  block_size = 3 # Block size for conditional block entropy calculation.
  @time CBEs = [length(sequence) > 0 ? conditional_block_entropy(Vector{Int}(sequence), block_size) : 0.0 for sequence in sol]

  # Reshape the CBEs into a 2D array.
  CBE_matrix = reshape(CBEs, (length(ΔCa_values), length(ΔVx_values)))

  # Create a heatmap of the conditional block entropies.
  fig_cbe = Figure(size=(800, 700))
  ax_cbe = Axis(fig_cbe[1, 1], 
      xlabel="ΔCa", 
      ylabel="ΔVx",
      title="Conditional $(block_size)-Block Entropy of Signed Spike-Count Sequences")

  hm_cbe = heatmap!(ax_cbe, ΔCa_values, ΔVx_values, CBE_matrix, colormap=:thermal)
  Colorbar(fig_cbe[1, 2], hm_cbe, label="Conditional Block Entropy")

  # Adjust the layout.
  colsize!(fig_cbe.layout, 1, Aspect(1, 1.0))
  colgap!(fig_cbe.layout, 20)

  # Display the figure.
  display(fig_cbe)

  # Optionally, save the figure.
  # save("conditional_block_entropy_heatmap.png", fig_cbe)
end