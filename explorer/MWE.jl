using OrdinaryDiffEq, DynamicalSystems, GLMakie, StaticArrays
using DataStructures: CircularBuffer

p = Observable(Float32[10,28,8/3])
u0 = rand(Float32, 3)

function lorenz(u,p,t)
    σ, ρ, β = p
    x, y, z = u
    SVector(σ*(y - x), x*(ρ - z) - y, x*y - β*z)
end

#use DynamicalSystems interface
dynsys = @lift CoupledODEs(lorenz, u0, $p, diffeq = (
    alg = BS3(), dt = .1f0
))

set_theme!(theme_black())
set_window_config!(framerate=60.0, focus_on_show=true, title = "MWE")

fig = Figure(resolution = (1500, 1000).*1.3);

trajax = Axis3(fig[1:2,1], azimuth = 5pi/13, elevation = pi/25)
trajax.title = "Trajectory"
trajax.xlabel = "x"
trajax.ylabel = "y"
trajax.zlabel = "z"

bifax = Axis(fig[1:2,2])
bifax.title = "Bifurcation Diagram"
bifax.xlabel = "r"
bifax.ylabel = "σ"

traceax = Axis(fig[3,1])
traceax.title = "Voltage Trace"
traceax.ylabel = "V"
traceax.xlabel = "t"

function progress_for_one_step!(solver, u)
    step!(solver[])
    solver[] = solver[]
    push!(u[],solver[].integ.u)
    u[] = u[]
end

#set length of stored trajectory
maxpoints = 25000
u = Observable(CircularBuffer{SVector{3,Float32}}(maxpoints))

# create initial trajectory
for i = 1:maxpoints[]
    progress_for_one_step!(dynsys, u)
end
traj = @lift map(x -> Point3f(x[[1,2,3]]...), $u)

lines!(trajax, traj, colormap = :devon, color = @lift 1:$maxpoints)
# z trace
trace = @lift [e[3] for e in $u]
lines!(traceax, trace)

#animate trajectory
isrunning = Observable(true)
delay = .0001

function run_traj()
    @async while isrunning[]
        isopen(fig.scene) || break # ensures computations stop if closed window
        progress_for_one_step!(dynsys, u)
        sleep(delay) # or `yield()` instead
    end
end

bifaxpoint = @lift Point2f($p[1], $p[2])
scatter!(bifax, bifaxpoint)

limits!(bifax, 10, 140, 0, 70)

bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    σ,ρ = pars
    p[] = [σ,ρ,p[][3]]
    bifaxpoint[] = Point2f(σ, ρ)
end

display(fig) # display
run_traj()