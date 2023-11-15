using GLMakie, CUDA
using LinearAlgebra: norm

resolution = 100 # How many points to sample.

# generate parameter space
xspace = LinRange(-1.5, 1.5, resolution)
caspace = LinRange(-45, -20, resolution)
ps = [[xspace[i], caspace[j]] for i in 1:resolution, j in 1:resolution]

# generate initial conditions
d0 = 1e-7
u0 = let
    u0 = rand(Float32, 5)
    perturb = randn(5)
    perturb = perturb/norm(perturb)*d0
    u0 + perturb
end



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