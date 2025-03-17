# ===== Current Range Specification =====
# Default constant current for simulation.
I_default = 0.0

# To perform a sweep over a range of constant currents, manually set I_min, I_max, and N_points.
# For a single simulation with the default current, set I_min = I_max = I_default.
I_min = -8e-3
I_max = 5e-3
N_points = 100

using Pkg
Pkg.activate("./network_symbolics")
Pkg.instantiate()

using GLMakie, OrdinaryDiffEq, StaticArrays, Roots, Printf

include("../model/Plant.jl")
using .Plant

include("../tools/symbolics.jl")
include("../tools/equilibria.jl")

# Define parameters and initial conditions.
p = vcat(Plant.default_params[1:15], [-0.9, -38.285])
u0 = SVector{7}(Plant.default_state_Isyn[1], 0.0, Plant.default_state_Isyn[2:end]...)
tspan = (0.0, 1e5)

# Compute the upper saddle equilibrium and set V_sd.
v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
if length(v_eqs) < 3
    error("Unable to find upper saddle equilibrium")
end
v_eq = v_eqs[3]
Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
x_eq = Plant.xinf(p, v_eq)
saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq, 0.0]
V_sd = saddle[6]

# Define event symbols and encoder state.
@enum EventSymbol begin
    Void   # Nothing detected yet.
    I      # Vdot maximum.
    Vplus  # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

mutable struct EncoderState
    symbols::Vector{Int}
    count::Int
    last_symbol::EventSymbol
    last2_symbol::EventSymbol
end

const STATE = Ref(EncoderState([], 0, Void, Void))

# Global synaptic current value (this will be varied).
global I_syn_value = I_default
function synaptic_current(t)
    return I_syn_value
end

# Callback condition: uses the model's derivative and its numerical derivative.
function condition(out, u, t, integrator)
    u_Isyn = [u[1:end-1]..., synaptic_current(t)]
    Vdot = Plant.dV(integrator.p, u_Isyn...)
    out[1] = -Vdot
    out[2] = -Plant.numerical_derivative_Isyn(
        (p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot) -> Vdot,
        synaptic_current,
        u_Isyn,
        integrator.p,
        t,
        1e-4
    )
end

# Callback affect!: update the encoder state based on the event type.
function affect!(integrator, event_index)
    if event_index == 1
        if integrator.u[6] > V_sd
            STATE[].count += 1
            STATE[].last2_symbol = STATE[].last_symbol
            STATE[].last_symbol = Vplus
        else
            push!(STATE[].symbols, STATE[].last2_symbol == Vplus ? -STATE[].count : STATE[].count)
            STATE[].count = 0
            STATE[].last2_symbol = STATE[].last_symbol
            STATE[].last_symbol = Vminus
        end
    elseif event_index == 2
        STATE[].last2_symbol = STATE[].last_symbol
        STATE[].last_symbol = I
    end
end

# Update the synaptic current in the state vector at each timestep.
function update_isyn!(integrator)
    integrator.u = SVector{7}(integrator.u[1:6]..., synaptic_current(integrator.t))
    return nothing
end

# Set up callbacks.
cb = VectorContinuousCallback(condition, affect!, nothing, 2)
cb_isyn = DiscreteCallback((u, t, integrator) -> true, update_isyn!)
cb_combined = CallbackSet(cb, cb_isyn)

# Define the ODE problem (remains the same for each run).
prob = ODEProblem(Plant.melibeNewIsyn, u0, tspan, p)

# Function to compute the branch coordinate from the spike–count symbols.
function sscs_to_branch_coordinate(sscs::Vector{Int})
    coordinate_interval = (0.0, 1.0)
    orientation = 1
    new_orientation = orientation
    for i in 1:length(sscs)
        if sscs[i] <= 0
            individual_coordinate_interval = (1.0 - 2.0^sscs[i], 1.0 - 3.0*2.0^(sscs[i]-2))
            new_orientation *= -1
        else
            individual_coordinate_interval = (1.0 - 3.0*2.0^(-sscs[i]-2), 1.0 - 2.0^(-sscs[i]-1))
        end
        a, b = individual_coordinate_interval
        if orientation == 1
            coordinate_interval = ((1-a)*coordinate_interval[1] + a*coordinate_interval[2],
                                   (1-b)*coordinate_interval[1] + b*coordinate_interval[2])
        else
            coordinate_interval = (a*coordinate_interval[1] + (1-a)*coordinate_interval[2],
                                   b*coordinate_interval[1] + (1-b)*coordinate_interval[2])
        end
        orientation = new_orientation
    end
    return coordinate_interval
end

# Simulate the model for a given constant current and return the center branch coordinate.
function simulate_current(I_const)
    global I_syn_value = I_const
    # Reset encoder state.
    STATE[] = EncoderState([], 0, Void, Void)
    sol = solve(prob, Tsit5(), abstol=3e-6, reltol=3e-6, callback=cb_combined)
    # Compute branch coordinate from the collected symbols (skipping the first entry).
    if length(STATE[].symbols) < 2
        return NaN
    end
    branch_interval = sscs_to_branch_coordinate(STATE[].symbols)
    center = sum(branch_interval) / 2
    return center
end

# Determine the range of constant currents to simulate.
# If a sweep is desired, N_points should be > 1. Otherwise, a single simulation is run.
I_values = N_points > 1 ? range(I_min, I_max, length=N_points) : [I_default]

# Run simulations for each current value and store the center branch coordinates.
center_coords = Float64[]
println("Running simulations for constant currents...")
for I in I_values
    println("Simulating for I = ", I)
    center = simulate_current(I)
    push!(center_coords, center)
    if isnan(center)
        println("  I = $(Printf.@sprintf("%.3e", I)): No valid symbols detected")
    else
        println("  I = $(Printf.@sprintf("%.3e", I)) → Center coordinate = $(Printf.@sprintf("%.3e", center))")
    end
end

# Plot the graph: constant current (horizontal) vs. center branch coordinate (vertical).
fig = Figure(resolution = (1200, 900))
ax = Axis(fig[1, 1], xlabel="Constant Synaptic Current (mA)", ylabel="Center Branch Coordinate")
scatterlines!(ax, I_values, center_coords, markersize=0, linewidth=2)
display(fig)