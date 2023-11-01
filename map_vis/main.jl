using Pkg
Pkg.activate("./map_vis")
Pkg.instantiate()
using GLMakie, OrdinaryDiffEq, StaticArrays, FileIO, Roots, NonlinearSolve, DelimitedFiles
using DataStructures: CircularBuffer
include("../model/Plant.jl")
using .Plant

p = Observable(convert.(Float64,(Plant.default_params)));
ΔCa = @lift($p[end]);
Δx = @lift($p[end-1]);

u0 = Observable(convert.(Float64,Plant.default_state));

set_theme!(theme_black())
set_window_config!(framerate=60.0, focus_on_show=true, title = "Melibe Leonina Swim InterNeuron (SIN) Model")

begin
    fig = Figure(resolution = (1500, 1000).*1.3);
    lights = [
        AmbientLight(0.8),
    ]
  
    trajax = Axis3(fig[1:2,1], azimuth = 5pi/13, elevation = pi/25)
    trajax.scene.lights = lights
    trajax.title = "Trajectory"
    trajax.xlabel = "Ca"
    trajax.ylabel = "x"
    trajax.zlabel = "V"

    bifax = Axis(fig[1:2,2], xrectzoom=false, yrectzoom=false)
    bifax.title = "Bifurcation Diagram (ΔCa: $(ΔCa[]), Δx: $(Δx[]))"
    bifax.xlabel = "ΔCa"
    bifax.ylabel = "Δx"
    bifax.title = "bifurcation diagram"
    mapwidgetax = GridLayout(fig[3:4,2], tellwidth = false)
    mapax = Axis(mapwidgetax[1,1], aspect=1)
    mapax.title = "1D Map"
    mapax.xlabel = rich("x", subscript("n"))
    mapax.ylabel = rich("x", subscript("n+1"))

    widgetax = GridLayout(fig[4,1], tellwidth = false)
    mapslider = SliderGrid(widgetax[2,:], 
        (label = "map end", range=.01:.01:0.4, format = "{:.0}",
             startvalue = .2, snap = false),
        (label = "map begin", range=.01:.01:0.4, format = "{:.0}",
             startvalue = 0., snap = false),
        )
end

#include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")

display(fig)