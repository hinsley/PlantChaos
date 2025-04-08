using Pkg
Pkg.activate(".")
Pkg.instantiate()

using GLMakie
using OrdinaryDiffEq
using StaticArrays

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

# Define parameters.
ΔCa = -30.0
Δx = -0.8
p = [Plant.default_params[1:end-2]..., Δx, ΔCa]

# Define time span for integration.
tspan = (0.0, 1e8)

# Create ODE problem.
prob = ODEProblem(melibeNew, u0, tspan, p)

# Solve the ODE problem.
sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

begin
  # Create a figure for plotting the solution.
  fig = Figure(resolution = (1200, 800))

  # Create a 3D axis for the phase space plot.
  ax3d = Axis3(fig[1, 1], 
             title = "3D Phase Space (Ca, x, V)",
             xlabel = "Ca", 
             ylabel = "x", 
             zlabel = "V")

  # Create a 2D axis for the voltage trace.
  axv = Axis(fig[2, 1], 
             title = "Voltage Trace",
             xlabel = "Time", 
             ylabel = "Voltage (mV)")

  # Extract solution components.
  t = sol.t
  V = [u[6] for u in sol.u]
  Ca = [u[5] for u in sol.u]
  x = [u[1] for u in sol.u]

  # Plot 3D trajectory.
  lines!(ax3d, Ca, x, V, color = :blue, linewidth = 2)

  # Plot voltage trace.
  lines!(axv, t, V, color = :red, linewidth = 1.5)

  # Add some perspective to better visualize the 3D space.
  ax3d.azimuth = 0.7
  ax3d.elevation = 0.3

  # Display the figure.
  display(fig)
end
