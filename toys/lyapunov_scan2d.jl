# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.

using Plots
using Plots.PlotMeasures
using StaticArrays

include("../model/Plant.jl")
include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -45.0 # Cashift
start_p[16] = -.5 # xshift
end_p = [Plant.default_params...]
end_p[17] = -20.0 # Cashift
end_p[16] = -1.5 # xshift

resolution = 100 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution) |> collect
x_shifts = LinRange(start_p[16], end_p[16], resolution) |> collect

u0 = @SVector Float32[
    0.2f0, # x
    Plant.default_state[2],
    Plant.default_state[3],
    1.0f0, # Ca
    Plant.default_state[5],
]

using DynamicalSystems
begin
    sys = CoupledODEs(Plant.melibeNew, u0, ps[1])
    systems = [deepcopy(sys) for i in 1:Threads.nthreads()-1]
    pushfirst!(systems, sys)
    lyaparray = Array{Float32}(undef,resolution,resolution)
    t = time()
    @time Threads.@threads for k in 1:resolution^2
        i = (k-1)%resolution+1
        j = ceil(Int, k/resolution)
        if k%1000 == 0
            println("$(k) out of $(resolution^2) at time $(round(time()-t, sigdigits = 6))")
        end
        system = systems[Threads.threadid()]
        set_parameter!(system, 17, Ca_shifts[i])
        set_parameter!(system, 16, x_shifts[j])
        lyaparray[i,j] = lyapunov(systems[Threads.threadid()], 1000000; Ttr = 10000,d0 = 1e-6, Δt = .1)
    end
end

heatmap(reverse(lyaparray, dims = 1),size = (1000,1000))
savefig("lyap_2d.png")