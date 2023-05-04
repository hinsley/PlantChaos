using Pkg
Pkg.activate("./explorer")
Pkg.instantiate()
include("../model/Plant.jl")
using OrdinaryDiffEq, DynamicalSystems, GLMakie
using FileIO, LinearAlgebra, Roots
using DataStructures: CircularBuffer

p = Observable(Plant.default_params)
ΔCa = @lift($p[end])
Δx = @lift($p[end-1])

u0 = Observable(Plant.default_state)

#use DynamicalSystems interface
dynsys = @lift CoupledODEs(Plant.melibeNew, $u0, $p, diffeq = (
    alg = BS3(),
))

set_theme!(theme_black())
set_window_config!(framerate=60.0, focus_on_show=true, title = "Melibe Leonina Swim InterNeuron (SIN) Model")

fig = Figure(resolution = (1500, 1000).*1.3);

trajax = Axis3(fig[1:2,1], azimuth = 5pi/13, elevation = pi/25)
trajax.title = "Trajectory"
trajax.xlabel = "Ca"
trajax.ylabel = "x"
trajax.zlabel = "V"

bifax = Axis(fig[1:2,2])
bifax.title = "Bifurcation Diagram (ΔCa: $(ΔCa[]), Δx: $(Δx[]))"
bifax.xlabel = "ΔCa"
bifax.ylabel = "Δx"

onany(ΔCa, Δx) do delCa, delx
    bifax.title = "Bifurcation Diagram (ΔCa: $delCa, Δx: $delx)"
end

mapax = Axis(fig[3:4,2], aspect = DataAspect())
mapax.title = "1D Map"
mapax.xlabel = rich("Ca", subscript("n"))
mapax.ylabel = rich("Ca", subscript("n+1"))

traceax = Axis(fig[3,1])
traceax.title = "Voltage Trace"
traceax.ylabel = "V"
traceax.xlabel = "t"

widgetax = GridLayout(fig[4,1], tellwidth = false)
widgetax[1,1] = pausebutton = Button(fig, label = "pause", buttoncolor = RGBf(.2,.2,.2))
widgetax[1,2] = clearbutton = Button(fig, label = "clear", buttoncolor = RGBf(.2,.2,.2))
widgetax[1,3] = resetbutton = Button(fig, label = "reset", buttoncolor = RGBf(.2,.2,.2))
speedslider = SliderGrid(widgetax[2,:], (label = "speed", range=1:.1:5, format = "{:.1f}", startvalue = 2))

ctrlax = GridLayout(fig[5,1:2], tellwidth = false)

bifctrlax = GridLayout(ctrlax[1,2], tellwidth = false)

include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")

display(fig) # display
run_traj()