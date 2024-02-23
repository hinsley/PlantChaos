using Pkg
Pkg.activate("./map_vis")
Pkg.instantiate()
using GLMakie, OrdinaryDiffEq, StaticArrays, FileIO,
    Roots, DelimitedFiles, NonlinearSolve,
    LinearAlgebra, ForwardDiff, Optim

using DataStructures: CircularBuffer
include("../model/Plant.jl")
using .Plant
using Peaks, Interpolations

p = Observable(convert.(Float64,(Plant.default_params)));
p[] = vcat(p[][1:15], [-1.09, -36.78]);
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
    bifax.title = "Bifurcation Diagram"
    bifax.xlabel = "ΔCa"
    bifax.ylabel = "Δx"
    bifax.title = "bifurcation diagram"
    mapwidgetax = GridLayout(fig[3:4,2], tellwidth = false)
    mapax = Axis(mapwidgetax[1,1], aspect=1)
    mapax.title = "1D Map"
    mapax.xlabel = rich("x", subscript("n"))
    mapax.ylabel = rich("x", subscript("n+1"))

    cmapax = Axis(mapwidgetax[1,2], aspect=1)
    cmapax.title = "1D circle Map"
    cmapax.xlabel = rich("Θ", subscript("n"))
    cmapax.ylabel = rich("Θ", subscript("n+1"))

    widgetax = GridLayout(fig[4,1], tellwidth = false)
    refinemin_button = Button(widgetax[1,1], label = "refine min", labelcolor = :black)
    refinemax_button = Button(widgetax[1,2], label = "refine max", labelcolor = :black)
    show_unstable_button = Button(widgetax[1,3], label = "show unstable", labelcolor = :black)
    switch_traj_button = Button(widgetax[1,4], label = "toggle trajectory sample", labelcolor = :black)
    print_button = Button(widgetax[1,5], label = "generate fig", labelcolor = :black)

    mapslider = SliderGrid(widgetax[4,:],
        (label = "map end", range=.001:.001:8, format = "{:.0}",
             startvalue = 20, snap = false),
        (label = "map begin", range=.001:.001:8, format = "{:.0}",
             startvalue = 0., snap = false),
        (label = "map iterates", range=1:1:500, format = "{:.0}",
             startvalue = 1, snap = false),
        (label = "circle map radius", range=.0001:.0001:.001, format = "{:.0}",
             startvalue = .001, snap = false),
        (label = "circle map begin", range=0.0:.001:2pi, format = "{:.0}",
             startvalue = 0, snap = false),
        (label = "circle map end", range=0.0:.001:2pi, format = "{:.0}",
             startvalue = 2pi, snap = false);
        width = 900,
        tellwidth = false
        )
end

#include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")
#include("./hom_map.jl")

display(fig)