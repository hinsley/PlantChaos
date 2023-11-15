# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.
using Pkg
Pkg.activate("./lyapunov")
using GLMakie, StaticArrays, ProgressMeter, DynamicalSystems
using LinearAlgebra: norm
using OrdinaryDiffEq
include("../model/Plant.jl")
include("./my_lyapunov.jl")

# set up parameter space for scan
start_p = Float64[Plant.default_params...]
start_p[17] = -41
#start_p[17] = -70.0 # Cashift
start_p[16] = -1.4# # xshift
#start_p[16] = -3. # xshift
end_p = Float64[Plant.default_params...]
end_p[17] = -30.#150.0 # Cashift
#end_p[17] = 50.#10.0 # Cashift
end_p[16] = -1.15#1. # xshift
#end_p[16] = 1.#1. # xshift

resolution = 100 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

# set up systems
u0 = @SVector [
    0.8e0;     # x
    0e0; # y
    0.137e0;   # n
    0.389e0;   # h
    0.8e0;     # Ca
    -62.0e0;   # V
    0.0e0      # Isyn
]

function melibeNew(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6], u[7]),
        0.0e0
    ]
end

sys = CoupledODEs(melibeNew, u0, ps[1]; diffeq = (alg = RK4(), dtmax = 6.0, abstol = 1e-7, reltol = 1e-7))
d0 = 1e-6
_d1 = randn(7)
_d2 = _d1/norm(_d1)*d0
psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])
systems = [deepcopy(psys) for _ in 1:Threads.nthreads() - 1]
pushfirst!(systems, psys)

traj = trajectory(sys, 150000)
begin
    fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = "t", ylabel = "x")
    lines!(ax, collect(traj[2]), traj[1][:,5])
    fig
end

lyapunov(sys, 150000; Ttr = 50000, d0 = d0)

# perform scan
lyaparray2 = zeros(Float64, resolution, resolution)
include("lyapunov2.jl")
begin
    p = Progress(resolution^2)
    Threads.@threads for I in 1:resolution^2
        i = div(I - 1, resolution) + 1
        j = mod(I - 1, resolution) + 1
        system = systems[Threads.threadid()]
        #u0 = u00#rand(5)
        _d1 = randn(7)
        _d2 = _d1/norm(_d1)*d0
        set_parameter!(system, 16, x_shifts[i])
        set_parameter!(system, 17, Ca_shifts[j])
        reinit!(system, u0, [SVector{7}(u0), SVector{7}(u0+_d2)])
        lyaparray2[i,j] = lyapunov(system, 1000000; Ttr = 10000, d0 = 1e-6, Δt = .1)
        next!(p)
    end
    finish!(p)
end

# plot results
begin fig = Figure(resolution = (2000,2000))
    arr = map(lyaparray2') do x
        std = 1e-9
        x > 0 ? 1-exp(-x^2/std) : -1+exp(-x^2/std)
    end
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lyaparray2',
    colormap = Reverse(:RdBu), colorrange = (-.00005, .00005))
    Colorbar(fig[1,2], hm)
    fig
end

lpos = map(lyaparray2) do x
    x > 5e-9 ? x : NaN
end
lneg = map(lyaparray2) do x
    x < 5e-9 ? x : NaN
end

begin fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lpos';
    colormap = [:grey, :yellow, :orange, :red], colorrange = (0, .0008));
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lneg';
    colormap = [:black, :midnightblue, :steelblue])
    fig
end

"""using CSV
dir = "./toys/homoclinics"
i = 58
begin
    fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = "Ca", ylabel = "x")
    d = CSV.File("$dir/$i.csv")
    Ca = d.Ca
    x = d.x
    lines!(ax, Ca, x, color = :black, lw = 2)
    global i = i+1
    fig
end
i"""

#fig

#save(homedir() * "/Dropbox/lyapunov_pd.png",fig)

#using CSV
#CSV.write(homedir() * "/Dropbox/lyapunov_pd.csv", CSV.Tables.table(lyaparray2))
