
function runge_kutta_step!(f, du, u, k, x_shift, Ca_shift, dt)
    @inbounds begin
    f(du, u, x_shift, Ca_shift)
    @. k[1,:] = dt * du
    f(du, u + 0.5f0 * k[1,:], x_shift, Ca_shift)
    @. k[2,:] = dt * du
    f(du, u + 0.5f0 * k[2,:], x_shift, Ca_shift)
    @. k[3,:] = dt * du
    f(du, u + 0.5f0 * k[3,:], x_shift, Ca_shift)
    u = u + (k[1,:] + 2f0 * k[2,:] + 2f0 * k[3,:] + dt * du) / 6f0
    end
end

#TTr = 0f0, dt = 1f0, d0 = 1f-9, rescale_dt::Int32 = Int32(10)
function lyapunov_kernel!(f, xs, cas, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

    xix = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    caix = threadIdx().y + (blockIdx().y - Int32(1)) * blockDim().y
    if xix > length(xs) return end
    if caix > length(cas) return end

    #initial conditions
    u = @MVector rand(5)
    du = @MVector zeros(5)
    pert = @MVector randn(5)
    u1 = u + pert / norm(pert) * d0
    k = @MMatrix zeros(3,5)
    
    #integration
    d = norm(u-u1)
    @cuprintln("d $d")
    λ_total = 0f0
    t = -TTr
    while t < T
        rescale_count = 0f0
        while rescale_count < rescale_dt
            @inbounds runge_kutta_step!(f!, du, u, k, xs[xix], cas[caix], dt)
            @inbounds runge_kutta_step!(f!, du, u1, k, xs[xix], cas[caix], dt)
            rescale_count += dt
            t += dt
            t*(t-dt) <= 0f0 && break
            t > T && break
        end
        #rescale
        a = norm(u-u1)
        @cuprintln("dnew $dnew")
        if t>0f0
            λ_total += log(dnew/(d + eps(Float32)))
        end
        u1 = u1 + (u1 - u) / (dnew  / d0)
        d = norm(u-u1) # calculated explicitly for numerical precision.
    end
    @inbounds lyapunov_exponents[xix, caix] = λ_total / T
    return
end

function lyapunov(f, xs, cas, T, TTr, dt, d0, rescale_dt)
    # calculate batch size()

    lyapunov_exponents = CUDA.zeros(length(xs), length(cas))
    @cuda threads=(32,32) blocks=(1,1) lyapunov_kernel!(f, xs, cas, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)
    return lyapunov_exponents
end

function optimal_launch_configuration(kernel)
    device = CUDA.device()
    warp_size = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)
    num_multiprocessors = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    # threads
    max_threads_per_block = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_threads_per_multiprocessor = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    # blocks
    max_blocks_per_multiprocessor = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)
    # registers
    max_registers_per_block = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    max_registers_per_multiprocessor = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    max_registers_per_thread = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_THREAD)
    # shared memory
    max_shared_memory_per_block = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    max_shared_memory_per_multiprocessor = CUDA.attribute(device,
        CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)

    
    return blocks, threads_per_block
end