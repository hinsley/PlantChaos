using Pkg
Pkg.instantiate()

using CairoMakie, OrdinaryDiffEq, StaticArrays, FileIO,
    Roots, DelimitedFiles, NonlinearSolve,
    LinearAlgebra, ForwardDiff, Optim

include("../../../map_vis/return_map_utils.jl")
include("../../../model/Plant.jl")
using .Plant
using Peaks, Interpolations

ca_shift = -40
x_shift = -1.28
p = SVector{17}(vcat(Plant.default_params[1:15], [x_shift, ca_shift]))
u0 = convert.(Float64, Plant.default_state)
map_resolution = 1000
preimage_range = (0.3, 0) # Distance from the equilibrium.

set_theme!(theme_black())

cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)
ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
monteprob = EnsembleProblem(map_prob, output_func= output_func, safetycopy=false)

# calculate and unpack all data needed for plotting
result = calculate_return_map(monteprob, ics_probs, p, preimage_range..., resolution=map_resolution)
preimage = result[1]
xmap = result[2]
cass = result[3]
xss = result[4]
vss = result[5]
ln1 = result[6]
ln2 = result[7]
lerp = result[8]
eq = result[9]

function flat_maxima(xmap, flatness_threshold)
    # Find local maxima in xmap.
    maxima_indices, _ = Peaks.findmaxima(xmap)

    # Filter maxima based on flatness criteria.
    valid_maxima_indices = Int[]
    for idx in maxima_indices[2:end-1]
        try
            # Check if the maximum is flat.
            if all(abs(xmap[idx] - xmap[i]) <= flatness_threshold for i in (idx-1):(idx+1))
                push!(valid_maxima_indices, idx)
            end
        catch BoundsError
            continue
        end
    end

    return valid_maxima_indices
end

flatness_threshold = 1e-3  # Adjust this for the flatness threshold.

flatmaxes = flat_maxima(xmap, flatness_threshold)
flat_maxima_values = xmap[flatmaxes]

# Refine flat maxima.
# ...For calculating single points on the map for refining near maxima.
function xreturn(lerp,prob,x)
    # get initial conditions by linear interpolation
    ics = lerp(x)
    # solve the map
    prob = remake(prob, u0 = ics)
    sol = solve(prob, RK4(), abstol = 1e-8, reltol = 1e-8, callback = cb)
    # return the final value
    return sol[1,end]
end
# Sharpen the flat maxima.
for (flatmax_i, xmap_i) in enumerate(flatmaxes) # Maybe get rid of enumerate when doing continuation for speed?
    opt = optimize(x -> -xreturn(lerp, remake(map_prob, p=(p=p, eq=eq)), x), preimage[xmap_i+1], preimage[xmap_i-1])
    xmap[xmap_i] = -Optim.minimum(opt)
    flat_maxima_values[flatmax_i] = xmap[xmap_i]
end

# plot the map
begin
    fig = Figure(resolution=(1500, 1000).*1.3)
    
    mapax = Axis(fig[1,1], aspect=1)
    mapax.title = "x min map"
    mapax.xlabel = rich("x", subscript("n"))
    mapax.ylabel = rich("x", subscript("n+1"))

    lines!(mapax, preimage, xmap, color = range(0.,1., length=map_resolution), colormap = :thermal)
    lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)
    lines!(mapax, ln1, color = :white, linewidth = 1.0, linestyle = :dash)
    lines!(mapax, ln2, color = :pink, linestyle = :dash, linewidth = 1)

    display(fig)
end