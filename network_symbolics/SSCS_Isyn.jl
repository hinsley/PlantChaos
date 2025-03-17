using Pkg
Pkg.activate("./network_symbolics")
Pkg.instantiate()
using GLMakie, OrdinaryDiffEq, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/symbolics.jl")

p = vcat(Plant.default_params[1:15], [-0.9, -38.285]) # Parameters: Delta V_x and Delta Ca
u0 = SVector{7}(Plant.default_state_Isyn[1], 0., Plant.default_state_Isyn[2:end]...)
tspan = (0.0, 1e5)

# Define a synaptic current function.
function synaptic_current(t)
    # Constant current.
    return -3e-3
    
    # Sinusoidal current.
    # amplitude = 0.05
    # frequency = 1e-2
    # offset = -amplitude
    # return amplitude * sin(frequency / 2pi * t) + offset
    
    # Pulse train.
    # return t % 200 < 50 ? 0.5 : 0.0
end

##########
## Symbolic (spike-count) encoding algorithm

# Solve for the upper saddle equilibrium.
include("../tools/equilibria.jl")
using Roots
v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
if length(v_eqs) < 3
  error("Unable to find upper saddle equilibrium")
end
v_eq = v_eqs[3]
Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
x_eq = Plant.xinf(p, v_eq)
saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq, 0.0]
V_sd = saddle[6]

@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

mutable struct EncoderState
  symbols::Vector{Int} # TODO: Rename this. It's not the same type as last_symbol or last2_symbol.
  count::Int
  last_symbol::EventSymbol # Most recently detected event.
  last2_symbol::EventSymbol # Second-most recently detected event.
end
const global STATE = Ref(EncoderState(
  [],
  0,
  Void,
  Void
))

function condition(out, u, t, integrator)
  u_Isyn = [u[1:end-1]..., synaptic_current(t)]
  Vdot = Plant.dV(integrator.p, u_Isyn...)
  out[1] = -Vdot

  # out[2] is -Vddot, but we must use finite differencing to calculate it accurately as the analytical derivative of minf is numerically unstable.
  out[2] = -Plant.numerical_derivative_Isyn(
    (p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot) -> Vdot,
    synaptic_current,
    u_Isyn,
    integrator.p,
    t,
    1e-4
  )
end

I_times::Vector{Float64} = []
Vplus_times::Vector{Float64} = []
Vminus_times::Vector{Float64} = []

function affect!(integrator, event_index)
  if event_index == 1
    if integrator.u[6] > V_sd
      STATE[].count += 1
      STATE[].last2_symbol = STATE[].last_symbol
      STATE[].last_symbol = Vplus
      push!(Vplus_times, integrator.t)
    else
      push!(
        STATE[].symbols,
        STATE[].last2_symbol == Vplus ? -STATE[].count : STATE[].count
      )
      STATE[].count = 0
      STATE[].last2_symbol = STATE[].last_symbol
      STATE[].last_symbol = Vminus
      push!(Vminus_times, integrator.t)
    end
  elseif event_index == 2
    STATE[].last2_symbol = STATE[].last_symbol
    STATE[].last_symbol = I
    push!(I_times, integrator.t)
  end
end

cb = VectorContinuousCallback(condition, affect!, nothing, 2)

# Add this function to update Isyn during integration
function update_isyn!(integrator)
    # Update the last component of the state vector (Isyn) at each timestep
    integrator.u = SVector{7}(integrator.u[1:6]..., synaptic_current(integrator.t))
    return nothing
end

cb_isyn = DiscreteCallback(
    (u,t,integrator) -> true, # Always triggered
    update_isyn!
)

cb_combined = CallbackSet(cb, cb_isyn)

prob = ODEProblem(Plant.melibeNewIsyn, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol=3e-6, reltol=3e-6, callback=cb_combined)#, save_everystep=false)

# Plot voltage trace and synaptic current in aligned subplots.
begin
  # Extract V(t) and time values from the solution
  ttr = 1
  u2 = sol.u[ttr:end]
  V_values = [u[6][] for u in u2]
  t_values = sol.t[ttr:end]

  # Create figure with two vertically aligned subplots
  fig = Figure(resolution=(3000, 400), fontsize=36)  # Increased height for two subplots

  # Voltage trace plot (top panel)
  ax1 = Axis(fig[1, 1], ylabel=L"V(t)")
  lines!(ax1, t_values, V_values, color=:black, linewidth=2)
  
  # Plot SSCS events and labels (existing functionality)
  Vminus_indices = [findfirst(>=(t), sol.t) for t in Vminus_times]
  scatter!(ax1, Vminus_times, [u[6] for u in sol.u[Vminus_indices]],
          color=:red, markersize=10)
  
  # Add SSCS labels
  vertical_offset = 12
  for (i, t) in enumerate(Vminus_times)
    text!(ax1, string(STATE[].symbols[i]),
          position=(t, sol.u[Vminus_indices[i]][6] + 
                   (STATE[].symbols[i] == 0 ? 2.4 * vertical_offset : -vertical_offset)),
          align=(:center, :top), color=:red, fontsize=23)
  end

  # Synaptic current plot (bottom panel)
  ax2 = Axis(fig[2, 1], xlabel=L"t", ylabel=L"I_{\mathrm{syn}}")
  
  # Get observed current values and extend range to include zero
  isyn_vals = synaptic_current.(t_values)
  min_unit = min(floor(Int, minimum(isyn_vals)), 0)
  max_unit = max(ceil(Int, maximum(isyn_vals)), 0)
  levels = min_unit:max_unit
  
  # Add horizontal rules at each integer unit (including zero)
  for level in levels
      lines!(ax2, [t_values[1], t_values[end]], [level, level], 
             color=:black, linestyle=:dot, linewidth=1)
  end
  
  # Plot synaptic current on top
  lines!(ax2, t_values, synaptic_current.(t_values), 
        color=:blue, linewidth=2)

  # Shared formatting
  linkxaxes!(ax1, ax2)  # Ensure time alignment
  xlims!(ax1, sol.t[ttr], t_values[end]+1e3)
  ylims!(ax1, -80, 37)
  
  hidespines!(ax1)
  hidespines!(ax2)
  hidedecorations!(ax1, label=false)
  hidedecorations!(ax2, label=false)

  display(fig)
  # save("dual_trajectory_plot.png", fig, dpi=1200)
end

function sscs_to_branch_coordinate(sscs::Vector{Int})
    coordinate_interval = (0, 1)
    orientation = 1
    new_orientation = orientation
    for i in 1:length(sscs)
        if sscs[i] <= 0
            individual_coordinate_interval = (1.0-2.0^sscs[i], 1.0-3.0*2.0^(sscs[i]-2))
            new_orientation *= -1
        else
            individual_coordinate_interval = (1.0-3.0*2.0^(-sscs[i]-2), 1.0-2.0^(-sscs[i]-1))
        end
        a, b = individual_coordinate_interval
        if orientation == 1
            coordinate_interval = (
                (1-a)*coordinate_interval[1] + a*coordinate_interval[2],
                (1-b)*coordinate_interval[1] + b*coordinate_interval[2]
            )
        else
            coordinate_interval = (
                a*coordinate_interval[1] + (1-a)*coordinate_interval[2],
                b*coordinate_interval[1] + (1-b)*coordinate_interval[2]
            )
        end
        orientation = new_orientation
    end
    return coordinate_interval
end

# Print the range of possible branch coordinate values
branch_coordinate_range = sscs_to_branch_coordinate(STATE[].symbols)
println("Branch coordinate range:")
println(branch_coordinate_range)
println("Center branch coordinate:")
println(sum(branch_coordinate_range) / 2)