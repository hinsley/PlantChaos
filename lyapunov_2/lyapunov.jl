
function runge_kutta_step(f, u, p, dt)
    k1 = dt * f(u, p)
    k2 = dt * f(u + 0.5f0 * k1, p)
    k3 = dt * f(u + 0.5f0 * k2, p)
    k4 = dt * f(u + k3, p)
    return u + (k1 + 2f0 * k2 + 2 * k3 + k4) / 6f0
end

function dist(u1, u2)
    sqrt(sum((u - u1).^2))
end

function lyapunov_kernel!(f, us, u1s, ps, lyapunov_exponents, T;
        TTr = 0f0, dt = 1f0, d0 = 1f-9, rescale_dt = 10)

    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if idx > length(ps) return end
    
    @inbounds begin
        p = ps[idx]
        u = us[idx]
        u1 = u1s[idx]
    end

    d = dist(u, u1)
    t0 = 0f0
    λ_total = 0f0

    for t = vcat(-TTr:dt:0,dt:dt:T) # TODO get rid of steprange
        rescale_count = 0
        while rescale_count < rescale_dt
            u = runge_kutta_step(f, u, p, dt)
            u1 = runge_kutta_step(f, u1, p, dt)
            rescale_count += dt
            t == 0f0 && break
        end
        #rescale
        dnew = dist(u, u1)
        λ_total += log(dnew/d)
        u1 = u1 + (u1 - u) / dnew * d0
        d = dist(u,u1) # calculated explicitly for numerical precision.
    end
    lyapunov_exponents[idx] = λ_total / T 
end

function lyapunov(f, ps, T, N, resolution; TTr = 0f0, dt = 1f0,
        d0 = 1f-9, d0_upper = d0*1f+3, d0_lower = d0*1f-3)
    N
    u = CUDA.rand(resolution, resolution)

    lyapunov_exponents = CUDA.fill(0.0f0, N)

    # Copy initial conditions to GPU
    CUDA.copyto!(u, initial_conditions)

    # Define grid and block sizes
    threads_per_block = 256
    blocks = ceil(Int, N / threads_per_block)

    # Launch the kernel
    @cuda blocks=blocks threads=threads_per_block lyapunov_kernel!(u, parameters, lyapunov_exponents, dt, total_time)

    # Copy results back to CPU
    lyapunov_exponents_cpu = Array(lyapunov_exponents)

    return lyapunov_exponents_cpu
end