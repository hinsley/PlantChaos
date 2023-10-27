using GLMakie
using StaticArrays


start_p = [Plant.default_params...]
start_p[17] = -45.0 # Cashift
start_p[16] = -.5 # xshift
end_p = [Plant.default_params...]
end_p[17] = -20.0 # Cashift
end_p[16] = -1.5 # xshift

resolution = 100 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

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
    # TODO: REVERT THIS! u[1], u[2], u[3], u[4], u[5], u[6], u[7] = u

    # du1 = dx(p, u[1] V)
    # du2 = dy(y, V)
    # du3 = dn(n, V)
    # du4 = dh(h, V)
    # du5 = dCa(p, Ca, u[1] V)
    # du6 = dV(p, u[1] y, n, h, Ca, V, Isyn)
    # du7 = 0.0e0
    # return @SVector T[du1, du2, du3, du4, du5, du6, du7]

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


using DynamicalSystems
sys = CoupledODEs(melibeNew, u0, ps[1])

begin
    lyaparray = Array{Float64}(undef,resolution,resolution)
    for i in 1:resolution, j in 1:resolution
        sys = CoupledODEs(melibeNew, u0, ps[i,j])
        lyaparray[i,j] = lyapunov(sys, 1000000; Ttr = 10000,d0 = 1e-6, Δt = .1)
    end
    heatmap(x_shifts, Ca_shifts, lyaparray, xlabel="Ca_shift", ylabel="Lyapunov exponent")
end

F = Figure()
ax = Axis(F[1,1], xlabel = "x_shift", ylabel = "Ca_shift")
heatmap!(ax, x_shifts, Ca_shifts, lyaparray, colormap = :thermal)
F