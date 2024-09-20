using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, Colors, IterTools, JLD2, OrdinaryDiffEq, Roots, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/equilibria.jl")
include("../tools/symbolics.jl")

p_base = SVector{length(Plant.default_params)}(Plant.default_params)
u0_base = SVector{6}(Plant.default_state[1], 0., Plant.default_state[2:end]...)
prob_base = ODEProblem(Plant.melibeNew, u0_base, (0.0, 1e6), p_base) # Adjust the time span here.

@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

# Define the scan parameters.
ΔCa_values = collect(range(-60.0, 100.0, length=2000)) # Adjust this.
ΔVx_values = collect(range(-4.0, 1.0, length=2000)) # Adjust this.
param_list = collect(Iterators.product(ΔCa_values, ΔVx_values))
state_list = [Dict( # State machines for symbolic encoder.
  :scs => [], # Signed spike count sequence.
  :count => 0, # Accumulator for spike count within a single burst.
  :last_symbol => Void, # Most recently detected symbol.
  :last2_symbol => Void, # Second-most recently detected symbol.
  :V_sd => 0.0 # The voltage value of the upper saddle equilibrium.
) for i in 1:length(param_list)]
max_seq_length = 80 # Maximum length of signed spike counts before terminating trajectory integration.
max_spike_count = 35 # Maximum number of spikes in a single burst before considering the trajectory tonic-spiking and terminating.
transient_time = 1e2 # Time to wait before beginning to detect events.

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

  # Define the condition function inside of prob_func as a closure so it can access the problem index i.
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

  # Define the affect! function inside of prob_func as a closure so it can access the problem index i.
  function affect!(integrator, idx)
    if idx == 1
      if integrator.u[6] > state_list[i][:V_sd]
        state_list[i][:count] += 1
        state_list[i][:last2_symbol] = state_list[i][:last_symbol]
        state_list[i][:last_symbol] = Vplus
        if state_list[i][:count] > max_spike_count
          terminate!(integrator) # Early termination upon tonic-spiking detected.
        end
      else
        push!(
          state_list[i][:scs], # Spike count sequence.
          (state_list[i][:last2_symbol] == Vplus ? -1 : 1) * state_list[i][:count]
        )
        state_list[i][:count] = 0
        state_list[i][:last2_symbol] = state_list[i][:last_symbol]
        state_list[i][:last_symbol] = Vminus
        if length(state_list[i][:scs]) > max_seq_length
          terminate!(integrator) # Early termination upon satisfactory sequence length.
        end
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

@save "inline_scan_SSC_sequences.jld2" sol

# Calculate & render normalized Lempel-Ziv complexity heatmap.
begin
  # Calculate normalized Lempel-Ziv complexity for each solution.
  last_n = 70 # Length of tail of signed spike-count sequences to use for LZ complexity calculation.
  prune_threshold = 1.0 # Discard any sequence comprising more than this fraction of subthreshold oscillations.
  # @time lz_complexities = [
  #     (count(x -> x == 0, sequence) / length(sequence) > prune_threshold) ? 0.0 : 
  #     (length(sequence) >= last_n ? normalized_LZ76_complexity(Vector{Int}(sequence[end-last_n+1:end])) : 
  #     (length(sequence) > 0 ? normalized_LZ76_complexity(Vector{Int}(sequence)) : 0.0)) 
  #     for sequence in sol
  # ]

  # Reshape the LZ complexities into a 2D array.
  lz_complexity_matrix = reshape(lz_complexities, (length(ΔCa_values), length(ΔVx_values)))

  # Create a heatmap of the complexities.
  fig_lz = Figure(size=(1100, 1000))
  ax = Axis(fig_lz[1, 1],
      xlabel="ΔCa",
      ylabel="ΔVx",
      title="Normalized Lempel-Ziv Complexity of\nSigned Spike-Count Sequences",
      xlabelsize=28,
      ylabelsize=28,
      titlesize=32,  # Increased title size
      xticksize=12,  # Increased x-axis tick size
      yticksize=12,  # Increased y-axis tick size
      xticklabelsize=24,  # Increased x-axis tick label size
      yticklabelsize=24)  # Increased y-axis tick label size

  # hm = heatmap!(ax, ΔCa_values, ΔVx_values, clipped_lz_complexity_matrix, colormap=:gist_heat, colorrange=(0.4, 1.0))
  hm = heatmap!(ax, ΔCa_values, ΔVx_values, clipped_lz_complexity_matrix, colormap=Reverse(:gist_heat), colorrange=(0.4, 1.0))
  Colorbar(fig_lz[1, 2], hm, label="LZ Complexity", labelsize=28, ticklabelsize=24)  # Increased colorbar tick label size

  # Adjust the layout.
  colsize!(fig_lz.layout, 1, Aspect(1, 1.0))
  colgap!(fig_lz.layout, 20)

  # Display the figure.
  display(fig_lz)

  # Optionally, save the figure.
  save("lz_complexity_heatmap.png", fig_lz)
end

# Calculate & render conditional block entropy heatmap.
begin
  # Calculate conditional block entropy for each solution.
  block_size = 5 # Block size for conditional block entropy calculation.
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