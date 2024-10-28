using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, OrdinaryDiffEq, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/symbolics.jl")

# p = vcat(Plant.default_params[1:15], [-.35, -47.2]) # Parameters: Delta V_x and Delta Ca
p = vcat(Plant.default_params[1:15], [-0.68008, -40.0]) # Parameters: Delta V_x and Delta Ca
u0 = SVector{6}(Plant.default_state[1], 0., Plant.default_state[2:end]...)

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
saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
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
  Vdot = Plant.dV(integrator.p, u...)
  out[1] = -Vdot

  # out[2] is -Vddot, but we must use finite differencing to calculate it accurately as the analytical derivative of minf is numerically unstable.
  out[2] = -Plant.numerical_derivative(
    (p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot) -> Vdot,
    u,
    integrator.p,
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
## End algorithm
##########

prob = ODEProblem(Plant.melibeNew, u0, (0., 4e5), p)
sol = solve(prob, Tsit5(), abstol=3e-6, reltol=3e-6, callback=cb)#, save_everystep=false)

# Plot voltage trace and Vdot over time.
begin
  # Extract V(t) and time values from the solution
  ttr = 20
  u2 = sol.u[ttr:end]
  V_values = [u[6][] for u in u2]
  t_values = sol.t[ttr:end]

  # Calculate Vdot values using the analytical formula
  Vdot_values = [Plant.dV(p, u...) for u in u2]

  # Create a figure with two panels: V(t) and Vdot(t)
  fig = Figure(resolution=(1500, 150))

  # Plot V(t) time trace in the first panel
  ax1 = Axis(fig[1, 1], xlabel="t", ylabel="V")

  lines!(ax1, t_values, V_values, label="V(t)", color=:black, linewidth=1)

  # Plot SSCS registration event times on V(t) plot
  Vminus_indices = [findfirst(>=(t), sol.t) for t in Vminus_times]
  scatter!(
    ax1,
    Vminus_times,
    [u[6] for u in sol.u[Vminus_indices]],
    label="Vminus times",
    color=:red,
    markersize=10
  )
  # Add SSCS symbol labels under scatter points
  vertical_offset = 12
  for (i, t) in enumerate(Vminus_times)
    text!(
      ax1,
      string(STATE[].symbols[i]),
      position=(t, sol.u[Vminus_indices[i]][6] + (STATE[].symbols[i] == 0 ? 2.2 * vertical_offset : -vertical_offset)),
      align=(:center, :top),
      color=:red
    )
  end

  xlims!(ax1, sol.t[ttr], t_values[end])
  ylims!(ax1, -75, 30)
  hidespines!(ax1)
  hidedecorations!(ax1, label=false)

  # Display the figure
  display(fig)

  # Save the figure
  save("single_trajectory_SSCS.png", fig, dpi=300)
end

