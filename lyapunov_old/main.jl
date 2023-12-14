using Pkg
Pkg.activate("../lyapunov_old/")
using GLMakie
using StaticArrays, ProgressMeter

include("../model/Plant.jl")

function melibeOld(u::AbstractArray{T}, p, t) where T
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

u0 = @SVector Float32[
    .2;     # x
    5.472e-46; # y
    0.137e0;   # n
    0.389e0;   # h
    1.0e0;     # Ca
    -62.0e0;   # V
    0.0e0      # Isyn
]

include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -70.0 # Cashift
start_p[16] = -4 # xshift
end_p = [Plant.default_params...]
end_p[17] = 100.0 # Cashift
end_p[16] = 2 # xshift

resolution = 100 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]


using DynamicalSystems
sys = CoupledODEs(melibeOld, u0, ps[1])

N = Threads.nthreads()

begin
    lyaparray = Array{Float64}(undef, resolution, resolution)
    
    # Initialize a progress bar
    p = Progress(resolution^2, 1, "Computing Lyapunov Exponents: ", 50)

    Threads.@threads for i in 1:resolution
        for j in 1:resolution
            sys = CoupledODEs(melibeOld, u0, ps[i,j])
            lyaparray[i,j] = lyapunov(sys, 1000000; Ttr = 100000, d0 = 1e-6, Δt = .1)
            next!(p)
        end
    end
end

begin
    f = Figure()
    ax = Axis(f[1,1])
    heatmap!(ax, Ca_shifts, x_shifts, lyaparray', colorrange = (-.00005,.00005))
    f
end


function upscale(array::Array{T}, res::Int) where T
    old_res1, old_res2 = size(array)
    results = Array{T}(undef, res, res)

    for i in 1:res
        for j in 1:res
            # Map the new coordinates to the old image scale
            x = ((i - 1) * (old_res1 - 1)) / (res - 1) + 1
            y = ((j - 1) * (old_res2 - 1)) / (res - 1) + 1

            # Determine the coordinates of the four surrounding pixels
            x1, y1 = floor(Int, x), floor(Int, y)
            x2, y2 = ceil(Int, x), ceil(Int, y)

            # Ensure the coordinates are within the bounds of the original image
            x1 = max(min(x1, old_res1), 1)
            y1 = max(min(y1, old_res2), 1)
            x2 = max(min(x2, old_res1), 1)
            y2 = max(min(y2, old_res2), 1)

            # Calculate the fractional parts of the new pixel's coordinates
            fx, fy = x - x1, y - y1

            # Calculate the bilinear interpolation
            result = (1 - fx) * (1 - fy) * array[x1, y1] +
                     fx * (1 - fy) * array[x2, y1] +
                     (1 - fx) * fy * array[x1, y2] +
                     fx * fy * array[x2, y2]

            results[i, j] = result
        end
    end

    return results
end
upscale_axis(ax, res) = LinRange(ax[1], ax[end], res)
function point_in_polygon(point, polygon)
    count = 0
    n = length(polygon)

    for i in 1:n
        p1 = polygon[i]
        p2 = polygon[i % n + 1]

        # Check if point is on the same y-level as the vertex
        if p1[2] < point[2] ≤ p2[2] || p2[2] < point[2] ≤ p1[2]
            # Calculate x-coordinate of the edge at point's y-level
            x_at_y = p1[1] + (point[2] - p1[2]) * (p2[1] - p1[1]) / (p2[2] - p1[2])

            # Increase count if the ray intersects
            count += point[1] < x_at_y
        end
    end

    return count % 2 == 1
end
function refine_image_within_polygon!(func::Function, array, polygon, ps)
    res = size(array)[1]
    pr = Progress(res^2, 1, "refining Lyapunov Exponents: ", 50)
    Threads.@threads for i in 1:res
        for j in 1:res
            p = ps[i,j]
            xs = p[16]
            cs = p[17]
            if point_in_polygon((cs,xs), polygon)
                array[i, j] = func(p)
            end
            next!(pr)
        end
    end
    return nothing
end

newres = 200
up_lyap = upscale(lyaparray, newres)
up_xs = upscale_axis(x_shifts, newres)
up_cas = upscale_axis(Ca_shifts, newres)
up_ps = [[Plant.default_params[1:15]; [up_xs[i], up_cas[j]]] for i in 1:newres, j in 1:newres]

ref_lyap = refine_image_within_polygon!(up_lyap, 
    [(-70,2), (-70,1.5), (-40,-2), (50, -3.2)], up_ps) do par
    sys = CoupledODEs(melibeOld, u0, par)
    lyapunov(sys, 1000000; Ttr = 100000, d0 = 1e-6, Δt = .1)
end


begin
    f = Figure()
    ax = Axis(f[1,1])
    heatmap!(ax, up_cas, up_xs,
        ref_lyap', colorrange = (-.0002,.0002))
    f
end

scatter(polygon)
point = (-59.8,-3.975)
point_in_polygon(point, polygon)