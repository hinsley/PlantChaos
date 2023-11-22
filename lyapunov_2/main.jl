
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

T = 100f0
TTr = 100f0
dt = 1f0
d0 = 1f-7
rescale_dt = Int32(10)

tr = @cuda lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

CUDA.registers(tr)
CUDA.memory(tr)

data = @cuda threads=(2,2) blocks=(1,1) lyapunov_kernel!(f!, xspace, caspace, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

        """du = @MVector zeros(5)
u = @MVector rand(5)
k = @MMatrix zeros(3,5)
xshift = 0f0
cashift = 0f0

runge_kutta_step!(f!, du, u, k, xshift, cashift, dt)
"""