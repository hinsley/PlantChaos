using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, OrdinaryDiffEq, Roots, StaticArrays

include("../model/Plant.jl")
using .Plant

p = vcat(Plant.default_params[1:15], [-0.81, -41]) # Parameters: Delta V_x and Delta Ca
u0 = SVector{6}(Plant.default_state[1], 0., Plant.default_state[2:end]...)

include("../tools/equilibria.jl")
include("../tools/symbolics.jl")

function get_saddle(p)
  v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
  if length(v_eqs) < 3
      return fill(NaN, 6)
  end
  v_eq = v_eqs[3]
  Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
  x_eq = Plant.xinf(p, v_eq)
  saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]

  return saddle
end

# Define parameter ranges for delta x and delta Ca
delta_Ca_range = range(-45, -20, length=100)
delta_x_range = range(-1.5, -0.5, length=100)

# Create a grid of parameter combinations
param_grid = [(dCa, dx) for dx in delta_x_range for dCa in delta_Ca_range]

# Function to create ODE problem for each parameter set
function prob_func(prob, i, repeat)
    delta_Ca, delta_x = param_grid[i]
    p_local = vcat(Plant.default_params[1:15], [delta_x, delta_Ca])
    saddle = get_saddle(p_local)
    u0_local = SVector{6}(saddle[1], 0., saddle[3:end]...)
    
    remake(prob, u0=u0_local, p=p_local)
end

# Create the base problem
base_prob = ODEProblem(Plant.melibeNew, u0, (0., 3e5), p)

# Create and solve the EnsembleProblem
ensemble_prob = EnsembleProblem(base_prob, prob_func=prob_func)
ensemble_sol = solve(
  ensemble_prob,
  Tsit5(),
  EnsembleThreads(),
  trajectories=length(param_grid),
  saveat=1e1
)

# Now 'ensemble_sol' contains the ODE solutions for each parameter combination
# You can process these solutions further as needed

# Process each trajectory to get kneading sequences and calculate LZ complexity
kneading_sequences = Vector{Vector{Int}}(undef, length(ensemble_sol))
lz_complexities = Vector{Float64}(undef, length(ensemble_sol))
conditional_block_entropies = Vector{Float64}(undef, length(ensemble_sol))

transient_to_truncate = Int(1e3)
#Threads.@threads for i in 1:length(ensemble_sol) # Seems to crash sometimes on Carter's laptop.
for i in 1:length(ensemble_sol)
    sol = ensemble_sol[i][transient_to_truncate:end]
    voltage_trace = [u[6] for u in sol.u] # Extract voltage from each state vector
    times = sol.t

    # Convert voltage trace to itinerary
    itinerary = voltage_trace_to_itinerary(voltage_trace, times)

    # Convert itinerary to kneading sequence
    kneading_sequence = itinerary_to_kneading_sequence(itinerary)

    kneading_sequences[i] = kneading_sequence

    # Calculate normalized Lempel-Ziv complexity
    lz_complexities[i] = normalized_LZ_complexity(kneading_sequence)

    # Calculate conditional block entropy
    conditional_block_entropies[i] = conditional_block_entropy(kneading_sequence, 5)
end

# Now 'kneading_sequences' contains the kneading sequence for each parameter combination
# and 'lz_complexities' contains the normalized Lempel-Ziv complexity for each sequence

# Create a heatmap of LZ complexities

# Reshape the LZ complexities into a 2D array
lz_complexity_matrix = reshape(lz_complexities, (length(delta_Ca_range), length(delta_x_range)))

# Reshape the conditional block entropies into a 2D array
conditional_block_entropy_matrix = reshape(conditional_block_entropies, (length(delta_Ca_range), length(delta_x_range)))

# Create the heatmap
fig = Figure(size=(800, 600))
ax = Axis(
    fig[1, 1],
    xlabel="ΔCa", 
    ylabel="ΔV_x",
    title="Normalized Lempel-Ziv Complexity"
    #title="Conditional Block Entropy"
)

hm = heatmap!(ax, delta_Ca_range, delta_x_range, lz_complexity_matrix,
    colormap=:thermal)
#hm = heatmap!(ax, delta_Ca_range, delta_x_range, conditional_block_entropy_matrix,
#    colormap=:thermal)

Colorbar(fig[1, 2], hm, label="LZ Complexity")
#Colorbar(fig[1, 2], hm, label="Conditional Block Entropy")

# Adjust the layout
colsize!(fig.layout, 1, Aspect(1, 1.0))
colgap!(fig.layout, 20)

# Save the figure
#save("lz_complexity_heatmap.png", fig)

# Display the figure (if in an interactive environment)
display(fig)