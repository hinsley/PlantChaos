using Pkg
Pkg.activate("./explorer")
include("../model/Plant.jl")
using .Plant
using OrdinaryDiffEq, DynamicalSystems, GLMakie
using FileIO, LinearAlgebra, Roots
using DataStructures: CircularBuffer

p = Observable(default_params)
u0 = default_state

#use DynamicalSystems interface
dynsys = @lift CoupledODEs(melibeNew, u0, $p, diffeq = (
    alg = BS3(), dt = 1f0
))

set_theme!(theme_black())
set_window_config!(framerate=60.0, focus_on_show=true, title = "Melibe Leonina Swim InterNeuron (SIN) Model")

fig = Figure(resolution = (1500, 1000).*1.3);

trajax = Axis3(fig[1,1], azimuth = 5pi/13, elevation = pi/25)
trajax.title = "Trajectory"

bifax = Axis(fig[1,2])
bifax.title = "Bifurcation Diagram"

mapax = Axis(fig[2,2], aspect = DataAspect())
mapax.title = "1D Map"
rowsize!(fig.layout,2,Relative(1/2))

widgetax = GridLayout(fig[2,1], tellwidth = false)
widgetax[1,1] = pausebutton = Button(fig, label = "pause", buttoncolor = RGBf(.2,.2,.2))
widgetax[1,2] = clearbutton = Button(fig, label = "clear", buttoncolor = RGBf(.2,.2,.2))
widgetax[1,3] = resetbutton = Button(fig, label = "reset", buttoncolor = RGBf(.2,.2,.2))
speedslider = labelslider!(fig, "speed", 1:.1:4; startvalue = 2)
widgetax[2,:] = speedslider.layout
widgetax[3,:] = scantype = Menu(fig,
    options = ["LZ Complexity", "Spike Count Variance", "Conditional Block Entropy"])

include("./trajectory.jl")
include("./bifurcation.jl")

display(fig) # display
run_traj()