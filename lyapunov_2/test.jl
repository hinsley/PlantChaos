# test f
begin
    u = @MVector rand(Float32, 5)
    du = @MVector zeros(Float32, 5)
    x_shift = 0f0
    Ca_shift = 0f0
    f!(du, u, x_shift, Ca_shift)

    trajectory = zeros(Float32, 100000)
    q = @MVector zeros(Float32, 5)

    dt = 1f0
    for i in 1:100000
        runge_kutta_step!(f!, du, u, q, x_shift, Ca_shift, dt)
        trajectory[i] = u[5]
    end
    lines(trajectory)
end