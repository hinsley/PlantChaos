
using Pkg; Pkg.activate("./lyapunov_2")
using GLMakie, CUDA, StaticArrays
using LinearAlgebra: norm

include("./lyapunov.jl")
include("./model.jl")

resolution = 100 # How many points to sample.

# generate parameter space
xspace = LinRange(-1.5, 1.5, resolution) |> collect |> cu
caspace = LinRange(-45, -20, resolution) |> collect |> cu

#initialize scan data
lyapunov_exponents = CUDA.zeros(resolution, resolution)

T = 2f0
TTr = 0f0
dt = 1f0
d0 = 1f-6
rescale_dt = 10f0

tr = @cuda lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

CUDA.registers(tr)
CUDA.memory(tr)

#CUDA.@profile lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
#        TTr, dt, d0, rescale_dt)

open("llvm_ir.txt", "w") do file
    @device_code_llvm(io = file, @cuda(lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)))
end

        
        
        
        
        """du = @MVector zeros(5)
u = @MVector rand(5)
k = @MMatrix zeros(3,5)
xshift = 0f0
cashift = 0f0

runge_kutta_step!(f!, du, u, k, xshift, cashift, dt)
"""