
using Pkg; Pkg.activate("./lyapunov_2")
using GLMakie, CUDA, StaticArrays
using LinearAlgebra: norm

include("./lyapunov.jl")
include("./model.jl")

resolution = 100 # How many points to sample.

# generate parameter space
xspace = LinRange(-4, 4, resolution) |> collect |> cu
caspace = LinRange(-45, 100, resolution) |> collect |> cu

#initialize scan data
lyapunov_exponents = CUDA.fill(NaN32, (resolution, resolution))

T = 500000f0
TTr = 500000f0
dt = 1f0
d0 = 1f-3
rescale_dt = 10f0

block_size = (16, 16)
grid_size = (Int(ceil(resolution/block_size[1])), Int(ceil(resolution/block_size[2])))
@cuda threads=block_size blocks=grid_size lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

CUDA.synchronize()

let 
    f = Figure(resolution = (800, 600))
    ax = Axis(f[1,1], xlabel = "Ca", ylabel = "x")
    hm = heatmap!(ax, caspace, xspace, lyapunov_exponents)
    Colorbar(f[1,2], hm, label = "Lyapunov exponent")
    f
end


#tr = @cuda lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
#        TTr, dt, d0, rescale_dt)

#CUDA.registers(tr)
#CUDA.memory(tr)

#CUDA.@profile lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
#        TTr, dt, d0, rescale_dt)
