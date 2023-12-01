# test f
begin
    u = @MVector rand(Float32, 5)
    du = @MVector zeros(Float32, 5)
    x_shift = -2f0
    Ca_shift = 0f0
    f!(du, u, x_shift, Ca_shift)

    trajectory = zeros(Float32, 1000000)
    q = @MVector zeros(Float32, 5)

    dt = 1f0
    for i in 1:1000000
        runge_kutta_step!(f!, du, u, q, x_shift, Ca_shift, dt)
        trajectory[i] = u[5]
    end
    lines(trajectory)
end

include("../model/Plant.jl")
using .Plant


du1 = melibeNew([u[1],0f0, u[2:end]...],Plant.default_params, 0f0)
f!(du, u, 0f0,0f0)
Plant.IT(Plant.default_params, u[1], u[5])

[du1[1], du1[3:end]...] - du

du1
du
