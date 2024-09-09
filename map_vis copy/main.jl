using Pkg
Pkg.activate("./map_vis_copy")
Pkg.instantiate()
using GLMakie, OrdinaryDiffEq, StaticArrays, FileIO,
    Roots, DelimitedFiles, NonlinearSolve,
    LinearAlgebra, ForwardDiff, Optim

using DataStructures: CircularBuffer
include("../model/Plant.jl")
using .Plant
using Peaks, Interpolations

p = Observable(convert.(Float64,(Plant.default_params)));
p[] = vcat(p[][1:15], [-1.078, -36.3154]);
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
    mapax.xlabel = rich("V", subscript("n"))
    mapax.ylabel = rich("V", subscript("n+1"))

    cmapax = Axis(mapwidgetax[1,2], aspect=1)
    cmapax.title = "eigenvalues"
    cmapax.xlabel = rich("real")
    cmapax.ylabel = rich("imag")

    widgetax = GridLayout(fig[4,1], tellwidth = false)
    refinemin_button = Button(widgetax[1,1], label = "refine min", labelcolor = :black)
    refinemax_button = Button(widgetax[1,2], label = "refine max", labelcolor = :black)
    show_unstable_button = Button(widgetax[1,3], label = "show unstable", labelcolor = :black)
    print_button = Button(widgetax[1,4], label = "generate fig", labelcolor = :black)
    run_map_button = Button(widgetax[1,5], label = "run map", labelcolor = :black)

    mapslider = SliderGrid(widgetax[4,:],
        (label = "map end", range=.0001:.0001:1, format = "{:.0}",
             startvalue = .5, snap = false),
        (label = "map begin", range=.0001:.0001:1, format = "{:.0}",
             startvalue = 1, snap = false),
        (label = "map iterates", range=1:1:500, format = "{:.0}",
             startvalue = 1, snap = false),;
        width = 900,
        tellwidth = false
        )
end

#include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")
#include("./hom_map.jl")

display(fig)

function get_eigs(p)
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v),
     Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[2]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> melibeNew(u,p,0), saddle)
    vals,vecs = eigen(jac)
    arr = [(real(e), imag(e)) for e in vals if real(e) != 0.0][1:5]
    return arr
end

eigs = @lift get_eigs($p)
lines
scatter!(cmapax, eigs, color = :white, markersize = 25, marker = 'o')
vlines!(cmapax, 0, color = :white, linewidth = 2)
hlines!(cmapax, 0, color = :white, linewidth = 2)

# show Closeup bifurcation diagram
using JLD2
lcarr = load("lyapunov_3color.jld2")["lcarr"] # .+ Makie.RGB(.5,.5,.5)
lcarr

start_p = [Plant.default_params...]
start_p[17] = 15 # Cashift
start_p[16] = -2.8 # xshift
end_p = [Plant.default_params...]
end_p[17] = 45 # Cashift
end_p[16] = -2.2 # xshift

resolution = 700 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)

image!(bifax, Ca_shifts, x_shifts, rotr90(lcarr))

points =[
    #(-2.6096227169036865, 28.127906799316406), # homoclinic to saddle
    (-2.6527276039123535, 28.78907774658203), # snpo 1
    (-2.627365827560425, 26.775069900512695), # cusp 1
    (-2.487631320953369, 22.474958419799805), # snpo 2
    (-2.553318500518799, 27.644296813964844), # cusp 2
    (-2.4775397777557373, 22.507494888305664), # snpo 3
]

scatter!(bifax, reverse.(points), color = :white, markersize = 10, marker = 'o')
