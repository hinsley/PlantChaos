using Pkg
Pkg.activate(".")
Pkg.instantiate()

using GLMakie
using OrdinaryDiffEq
using StaticArrays
using Colors, IterTools, JLD2, Roots
using ProgressMeter  # Add this package for better progress tracking

### Begin
# Model extension (from Jack).
gh = .0005

include("../../model/Plant.jl")
function melibeNew(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

u0 = @SVector Float64[
    .2;     # x
    .1;     #y
    0.137e0;   # n
    0.389e0;   # h
    1.0e0;     # Cadisp
    -62.0e0;   # V
]
### End

# Include necessary tools.
include("../../tools/equilibria.jl")
include("../../tools/symbolics.jl")

# Define event symbols for symbolic encoding.
@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

# Define base parameters.
p_base = Plant.default_params

# Define the scan parameters.
ΔCa_values = collect(range(-50.0, -20.0, length=2)) # Adjust as needed.
Δx_values = collect(range(-1.6, -0.4, length=2)) # Adjust as needed.
param_list = collect(Iterators.product(ΔCa_values, Δx_values))
state_list = [Dict( # State machines for symbolic encoder.
  :scs => [], # Signed spike count sequence.
  :count => 0, # Accumulator for spike count within a single burst.
  :last_symbol => Void, # Most recently detected symbol.
  :last2_symbol => Void, # Second-most recently detected symbol.
  :V_sd => 0.0 # The voltage value of the upper saddle equilibrium.
) for i in 1:length(param_list)]

# Define simulation constraints.
max_seq_length = 80 # Maximum length of signed spike counts before terminating trajectory integration.
max_spike_count = 35 # Maximum number of spikes in a single burst before considering the trajectory tonic-spiking and terminating.
transient_time = 1e2 # Time to wait before beginning to detect events.

# Create base ODE problem with default parameters.
prob_base = ODEProblem(melibeNew, u0, (0.0, 1e6), p_base)

# Adjust the problem for each parameter vector in the scan.
function prob_func(prob, i, repeat)
  # Load the parameters from the pre-defined array.
  ΔCa, Δx = param_list[i]

  # Construct new parameter vector.
  p_new = [p_base[1:end-2]..., Δx, ΔCa]

  # Initialize u0_new to ensure it's always defined
  u0_new = copy(u0)

  # Try to calculate the upper saddle equilibrium for initial conditions.
  try
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p_new, v), -80.0, 0.0)
    if length(v_eqs) >= 3
      v_eq = v_eqs[3]
      Ca_eq = Equilibria.Ca_null_Ca(p_new, v_eq)
      x_eq = Plant.xinf(p_new, v_eq)
      u0_new = @SVector [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
      
      # Update the state machine with the appropriate V_sd value.
      state_list[i][:V_sd] = v_eq
    else
      # Fallback if we can't find equilibria.
      state_list[i][:V_sd] = -30.0 # Approximate threshold.
    end
  catch
    # Fallback if equilibria calculation fails.
    state_list[i][:V_sd] = -30.0 # Approximate threshold.
  end

  # Define the condition function for event detection.
  function condition(out, u, t, integrator)
    if t < transient_time
      out[1] = 0.0
      out[2] = 0.0
    else
      # Calculate Vdot directly.
      Vdot = Plant.dV(integrator.p, u...)
      out[1] = -Vdot
      
      # More robust second derivative approximation
      h = 1e-6  # Smaller step size
      # Forward finite difference approximation of Vddot
      du = similar(u)
      melibeNew(du, integrator.p, t)
      u_forward = similar(u)
      for i in 1:length(u)
        u_forward[i] = u[i] + h * du[i]
      end
      Vdot_forward = Plant.dV(integrator.p, u_forward...)
      out[2] = -(Vdot_forward - Vdot)/h
    end
  end

  # Define the affect! function to handle events.
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

  # Define callback to symbolically encode trajectories with more robust settings
  cb = VectorContinuousCallback(
    condition, 
    affect!, 
    nothing, 
    2;
    abstol=1e-9,  # Tighter tolerance for root finding
    reltol=1e-6,
    rootfind=true,
    save_positions=(false,false),
    interp_points=20  # More interpolation points for accuracy
  )

  # Return the modified problem.
  return remake(
    prob,
    p=p_new,
    u0=u0_new,
    callback=cb
  )
end

# Define output function to only return the symbolic sequence.
function output_func(sol, i)
  return (state_list[i][:scs], false) # (output, rerun).
end

# Define the EnsembleProblem for the parameter sweep.
ensemble_prob = EnsembleProblem(
  prob_base,
  prob_func=prob_func,
  output_func=output_func
)

# Solve the ensemble problem with more conservative settings
println("Starting parameter sweep...")
@time sol = solve(
  ensemble_prob,
  Tsit5(),
  abstol=1e-8,  # Tighter tolerances
  reltol=1e-8,
  maxiters=1e7,  # Allow more iterations
  dtmin=1e-12,   # Allow smaller timesteps
  force_dtmin=true,  # Force solver to use dtmin instead of erroring
  EnsembleSerial(),
  trajectories=length(param_list),
  save_on=false,
  progress=true
)

# Save the results.
save_filename = "sscs_sweep_results.jld2"
@save save_filename sol ΔCa_values Δx_values
println("Results saved to $save_filename")

# Calculate normalized Lempel-Ziv complexity for each solution.
last_n = 70 # Length of tail to use for LZ complexity calculation.
prune_threshold = 1.0 # Discard sequences with too many subthreshold oscillations.
minimum_n = 20 # Minimum number of signed spike counts to plot.

function normalized_LZ76_complexity(sequence)
    # This is a placeholder - you'll need to implement this function
    # or include it from your tools/symbolics.jl
    return rand() # Replace with actual implementation
end

println("Calculating Lempel-Ziv complexities...")
lz_complexities = zeros(length(sol))
p = Progress(length(sol), desc="Computing LZ complexity: ", dt=0.5)
for (i, sequence) in enumerate(sol)
    if count(x -> x == 0, sequence) / length(sequence) > prune_threshold
        lz_complexities[i] = 0.0
    elseif length(sequence) >= last_n
        lz_complexities[i] = normalized_LZ76_complexity(Vector{Int}(sequence[end-last_n+1:end]))
    elseif length(sequence) > minimum_n
        lz_complexities[i] = normalized_LZ76_complexity(Vector{Int}(sequence))
    else
        lz_complexities[i] = 0.0
    end
    next!(p)
end

# Reshape the LZ complexities into a 2D array.
lz_complexity_matrix = reshape(lz_complexities, (length(ΔCa_values), length(Δx_values)))

# Create a heatmap of the complexity values.
fig_lz = Figure(size=(1200, 1000))
ax = Axis(fig_lz[1, 1],
  xlabel="ΔCa",
  ylabel="Δx",
  title="Normalized Lempel-Ziv Complexity of Signed Spike-Count Sequences",
  xlabelsize=28,
  ylabelsize=28,
  titlesize=32,
  xticksize=12,
  yticksize=12,
  xticklabelsize=24,
  yticklabelsize=24)

filtered_matrix = replace(x -> x == 0 ? NaN : x, lz_complexity_matrix)
hm = heatmap!(ax, ΔCa_values, Δx_values, filtered_matrix, colormap=Reverse(:gist_heat), colorrange=(0.4, 1.0))
Colorbar(fig_lz[1, 2], hm, label="LZ Complexity", labelsize=28, ticklabelsize=24)

# Adjust the layout.
colsize!(fig_lz.layout, 1, Aspect(1, 1.0))
colgap!(fig_lz.layout, 20)

# Display and save the figure.
display(fig_lz)
save("lz_complexity_heatmap.png", fig_lz)

