@inline function runge_kutta_step!(f, du, u, q, x_shift, Ca_shift, dt)
    @inbounds begin
    @. q = u
    f(du, u, x_shift, Ca_shift)
    @. q = q + dt*du/6f0
    f(du, u + 0.5f0/dt * du, x_shift, Ca_shift)
    @. q = q + dt*du/3f0
    f(du, u + 0.5f0/dt * du, x_shift, Ca_shift)
    @. q = q + dt*du/3f0
    f(du, u + dt*du, x_shift, Ca_shift)
    @. u = q + dt*du/6f0
    end
    return nothing
end

#TTr = 0f0, dt = 1f0, d0 = 1f-9, rescale_dt::Int32 = Int32(10)
function lyapunov_kernel!(f, xs, cas, lyapunov_exponents, T,
        TTr, dt, d0, rescale_dt)

    xix = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    caix = threadIdx().y + (blockIdx().y - Int32(1)) * blockDim().y
    if xix > length(xs) return end
    if caix > length(cas) return end

    #initial conditions
    u = MVector{5, Float32}(undef)
    i = Int32(1)
    while i <= Int32(5)
        @inbounds u[i] = rand(Float32)
        i += Int32(1)
    end
    du = MVector{5, Float32}(undef)
    pert = MVector{5, Float32}(undef)
    i = Int32(1)
    while i <= Int32(5)
        @inbounds pert[i] = randn(Float32)
        i += Int32(1)
    end
    u1 = u + pert / norm(pert) * d0
    #keeps track of actual distance, approximately d0
    d = norm(u1-u)
    # preallocated container for rk4
    q = MVector{5, Float32}(undef)

    λ_total = 0f0
    t = -TTr
    a = 0f0

    while t < T
        rescale_count = 0f0
        while rescale_count < rescale_dt
            @inbounds runge_kutta_step!(f!, du, u, q, xs[xix], cas[caix], dt)
            @inbounds runge_kutta_step!(f!, du, u1, q, xs[xix], cas[caix], dt)
            rescale_count += dt
            t += dt
            t*(t-dt) <= 0f0 && break
            t > T && break
        end
        #rescale
        dnew = norm(u1-u)
        if t>0f0
            λ_total += log(d/dnew)
        end
        @. u1 = u + (u1 - u) / (dnew/d0 + 1f-12)
        d = dnew
    end
    @inbounds lyapunov_exponents[caix, xix] = λ_total / T
    return nothing
end

"""function optimal_launch_configuration(kernel)
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
end"""