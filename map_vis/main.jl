using Pkg
Pkg.activate("./map_vis")
Pkg.instantiate()
using GLMakie, OrdinaryDiffEq, StaticArrays, FileIO
using DataStructures: CircularBuffer
using Roots

include("../model/Plant.jl")
p = Observable(convert.(Float64,(Plant.default_params)));
ΔCa = @lift($p[end]);
Δx = @lift($p[end-1]);

u0 = Observable(convert.(Float64,Plant.default_state));
prob = @lift ODEProblem(
    Plant.melibeNew, $u0, (0.0, 100.0), $p);
integ = @lift init($prob, RK4(), abstol=1e-6, reltol=1e-6, dtmax = 6.0);

set_theme!(theme_black())
set_window_config!(framerate=60.0, focus_on_show=true, title = "Melibe Leonina Swim InterNeuron (SIN) Model")

begin
    fig = Figure(resolution = (1500, 1000).*1.3);

    trajax = Axis3(fig[1:2,1], azimuth = 5pi/13, elevation = pi/25)
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
    widgetax[1,1] = pausebutton = Button(fig, label = "pause", buttoncolor = RGBf(.2,.2,.2))
    widgetax[1,2] = clearbutton = Button(fig, label = "clear", buttoncolor = RGBf(.2,.2,.2))
    widgetax[1,3] = resetbutton = Button(fig, label = "reset", buttoncolor = RGBf(.2,.2,.2))
    speedslider = SliderGrid(widgetax[2,:], (label = "speed", range=1:.1:5, format = "{:.1f}", startvalue = 2))
end

#include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")

display(fig)
#run_traj()
