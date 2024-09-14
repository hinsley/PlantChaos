using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()
using CairoMakie, OrdinaryDiffEq, StaticArrays

include("../model/Plant.jl")
using .Plant

include("../tools/symbolics.jl")

p = vcat(Plant.default_params[1:15], [-0.81, -41]) # Parameters: Delta V_x and Delta Ca
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

prob = ODEProblem(Plant.melibeNew, u0, (0., 1e5), p)
sol = solve(prob, RK4(), abstol=1e-14, reltol=1e-14, callback=cb)#, save_everystep=false)

# Plot voltage trace and Vdot over time.
begin
  # Extract V(t) and time values from the solution
  V_values = [u[6] for u in sol.u]
  t_values = sol.t

  # Create a figure with two panels: V(t) and Vdot(t)
  fig = Figure(resolution=(800, 600))

  # Plot V(t) time trace in the first panel
  ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Voltage (mV)", title="V(t)")
  lines!(ax1, t_values, V_values, label="V(t)", color=:green)

  # Plot event times on V(t) plot
  scatter!(
    ax1,
    I_times,
    fill(-40, length(I_times)),
    label="I times",
    color=:red,
    markersize=10
  )
  scatter!(
    ax1,
    Vplus_times,
    fill(-36, length(Vplus_times)),
    label="Vplus times",
    color=:blue,
    markersize=10
  )
  scatter!(
    ax1,
    Vminus_times,
    fill(-38, length(Vminus_times)),
    label="Vminus times",
    color=:black,
    markersize=10
  )
  axislegend(ax1)

  # Display the figure
  display(fig)
end