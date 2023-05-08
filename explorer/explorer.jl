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
initCa = @lift ($u0[5])
initx = @lift ($u0[1])
initV = @lift ($u0[6])

stepsize = Observable(5f0)
abstol = Observable(1e-6)
reltol = Observable(1e-3)
maxpoints = Observable(1000)

#use DynamicalSystems interface
dynsys = @lift CoupledODEs(Plant.melibeNew, $u0, $p, diffeq = (
    alg = BS3(), abstol = $abstol, reltol = $reltol
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

initctrlax = GridLayout(ctrlax[1,1], tellwidth = false)

Label(initctrlax[1,1], "V: ")
initctrlax[1,2] = initV_tb = Textbox(fig, validator = Float32, placeholder="$(initV[])", width=150)

on(initV) do newV
    initV_tb.displayed_string = string(newV)
end

Label(initctrlax[1,3], "Ca: ")
initctrlax[1,4] = initCa_tb = Textbox(fig, validator = Float32, placeholder="$(initCa[])", width=150)

on(initCa) do newCa
    initCa_tb.displayed_string = string(newCa)
end

Label(initctrlax[1,5], "x: ")
initctrlax[1,6] = initx_tb = Textbox(fig, validator = Float32, placeholder="$(initx[])", width=150)

on(initx) do newx
    initx_tb.displayed_string = string(newx)
end

initupdatebutton = Button(initctrlax[1,7], label = "update", buttoncolor = RGBf(.2,.2,.2))

on(initupdatebutton.clicks) do clicks
    newV = parse(Float32, initV_tb.displayed_string[])
    newCa = parse(Float32, initCa_tb.displayed_string[])
    newx = parse(Float32, initx_tb.displayed_string[])

    u0.val = (newx, u0[][2:4]..., newCa, newV, u0[][end])
    auto_dt_reset!(dynsys[].integ)
    u0[] = u0[]
    build_map!(map_prob, mapics[])
    empty!(traj[])
end

bifctrlax = GridLayout(ctrlax[1,2], tellwidth = false)

integctrlax = GridLayout(ctrlax[2,1:2], tellwidth = false)

Label(integctrlax[1,1], "step size: ")
integctrlax[1,2] = stepsize_tb = Textbox(fig, validator = Float32, placeholder="$(stepsize[])", width=150)

Label(integctrlax[1,3], "abstol: ")
integctrlax[1,4] = abstol_tb = Textbox(fig, validator = Float32, placeholder="$(abstol[])", width=150)

Label(integctrlax[1,5], "reltol: ")
integctrlax[1,6] = reltol_tb = Textbox(fig, validator = Float32, placeholder="$(reltol[])", width=150)

Label(integctrlax[1,7], "max points: ")
integctrlax[1,8] = maxpoints_tb = Textbox(fig, validator = Float32, placeholder="$(maxpoints[])", width=150)

integupdatebutton = Button(integctrlax[1,9], label = "update", buttoncolor = RGBf(.2,.2,.2))

on(integupdatebutton.clicks) do clicks
    stepsize[] = parse(Float32, stepsize_tb.displayed_string[]) 
    abstol[] = parse(Float32, abstol_tb.displayed_string[])
    reltol[] = parse(Float32, reltol_tb.displayed_string[])
    newmaxpoints = parse(Int64, maxpoints_tb.displayed_string[])

    if maxpoints[] != newmaxpoints
        newtraj = CircularBuffer{Point{3,Float32}}(newmaxpoints)

        if maxpoints[] < newmaxpoints
            append!(newtraj, traj[])
        else
            append!(newtraj, traj[][end-(newmaxpoints-1):end])
        end
        
        maxpoints[] = newmaxpoints
        traj[] = newtraj
    end
end

include("./trajectory.jl")
include("./bifurcation.jl")
include("./return_map.jl")

display(fig) # display
run_traj()