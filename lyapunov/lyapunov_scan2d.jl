# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.
using Pkg
Pkg.activate("./lyapunov")
using GLMakie
using StaticArrays

include("../model/Plant.jl")

start_p = [Plant.default_params...]
start_p[17] = -70.0 # Cashift
start_p[16] = -4. # xshift
end_p = [Plant.default_params...]
end_p[17] = 100.0 # Cashift
end_p[16] = 1. # xshift

resolution = 5 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

u0 = @SVector Float32[
    0.2f0, # x
    Plant.default_state[2],
    Plant.default_state[3],
    Plant.default_state[4],
    1.0f0, # Ca
    Plant.default_state[6],
    Plant.default_state[7]
]

using DynamicalSystems, OrdinaryDiffEq
using LinearAlgebra: norm
sys = CoupledODEs(Plant.melibeNew, u0, ps[1]; diffeq = (alg = RK4(), adaptive = false, dt = .5f0))
d0 = 1f-5
_d1 = randn(Float32, length(u0))
_d2 = _d1/norm(_d1)*d0
psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])

λ = 0f0
begin
    d = λdist(psys)/d0
    λ += log(d)/.5f0
    println("λ: ",λ)
    println("d: ",d)
    λrescale!(psys, d)
    derr = λdist(psys)/d0
    println("derror: ", derr)
    step!(psys)
    nothing
end



step!(psys)

d = λdist(psys)/d0

λrescale!(psys, 1f0)

 #plot trajectory. It is a tonic spiker and should be stable.
 """let F = Figure()
    traj = trajectory(sys, 100000)
    ax = Axis(F[1,1], xlabel = "Time", ylabel = "v")
    lines!(ax, traj[1][:,6])
    F
end
"""
#lyapunov(psys, 10000f0; Ttr = 100000f0, d0 = d0, d0_upper = 100f0*d0, d0_lower = .01f0*d0, Δt = 1f0)
function λrescale!(pds::ParallelDynamicalSystem, a)
    u1 = current_state(pds, 1)
    u2 = current_state(pds, 2)
    if ismutable(u2) # if mutable we assume `Array`
        @. u2 = u1 + (u2 - u1)/a
    else # if not mutable we assume `SVector`
        u2 = @. u1 + (u2 - u1)/a
    end
    set_state!(pds, u2, 2)
end
function λdist(ds::ParallelDynamicalSystem)
    u1 = current_state(ds, 1)
    u2 = current_state(ds, 2)
    # Compute euclidean dinstace in a loop (don't care about static or not)
    d = zero(eltype(u1))
    @inbounds for i in eachindex(u1)
        d += (u1[i] - u2[i])^2
    end
    return sqrt(d)
end



systems = [deepcopy(psys) for _ in 1:Threads.nthreads() - 1]
pushfirst!(systems, psys)

lyaparray = zeros(Float32, resolution, resolution)
Threads.@threads for I in 1:resolution^2
    i = div(I - 1, resolution) + 1
    j = mod(I - 1, resolution) + 1
    system = systems[Threads.threadid()]
    set_parameter!(system, 16, x_shifts[i])
    set_parameter!(system, 17, Ca_shifts[j])
    lyaparray[i,j] = lyapunovspectrum(system, 1000, Ttr = 100, di,)
end



let fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lyaparray'; colorrange = (.0000,.00025), 
    colormap = [:midnightblue, :lightskyblue1, :white, :yellow, :orange, :red], lowclip= :black)
    Colorbar(fig[1,2], hm)
    fig
end