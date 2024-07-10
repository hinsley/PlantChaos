
map_resolution = 150

include("./return_map_utils.jl")
include("./map_prob.jl")
#include("./upper_map.jl")

# calculate and unpack all data needed for plotting
_ans = @lift calculate_return_map(monteprob, ics_probs, $p, $(mapslider.sliders[1].value),
    $(mapslider.sliders[2].value), resolution = map_resolution)
preimage = @lift $_ans[1]
xmap = @lift $_ans[2]
cass = @lift $_ans[3]
xss = @lift $_ans[4]
vss = @lift $_ans[5]
ln1 = @lift $_ans[6]
ln2 = @lift $_ans[7]
lerp = @lift $_ans[8]
eq = @lift $_ans[9]

# plot the map
lines!(mapax, preimage, xmap, color = range(0.,1., length=map_resolution), colormap = :thermal)
lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)
lines!(mapax, ln1, color = :white, linewidth = 1.0, linestyle = :dash)
lines!(mapax, ln2, color = :pink, linestyle = :dash, linewidth = 1)
#lines!(mapax, ln3, color = :red, linestyle = :dash, linewidth = 1)

refine_prob = ODEProblem{false}(mapf, @SVector(zeros(BigFloat,6)), (BigFloat(0), BigFloat(1e5)), zeros(17))

# events for buttons
on(refinemin_button.clicks) do b
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    reset_limits!(mapax)
end
on(refinemax_button.clicks) do b
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage, true)
    reset_limits!(mapax)
end

# reset limits on changing section length
on(xss) do _
    reset_limits!(mapax)
    reset_limits!(trajax)
end

sadplot1, sadplot2, sadplot3 = nothing, nothing, nothing
on(show_unstable_button.clicks) do b
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    global sadplot1 =lines!(trajax, sad_upper, color = :white, linewidth = 1.0)
    global sadplot2 =lines!(trajax, sad_lower, color = :white, linewidth = 1.0)
    ln = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end]),
        (NaN,NaN),
        (preimage[][end], sad_lower[3,end]),
        (preimage[][1], sad_lower[3,end])
    ]
    global sadplot3 = lines!(mapax, ln, color = :white, linewidth = 1.0, linestyle = :dash)
    reset_limits!(mapax)
    reset_limits!(trajax)
end

colorrng = range(0, 1, length = length(cass[]))
lines!(trajax, cass, xss, vss, color = colorrng, colormap = :thermal,
    linewidth = .2)


## TODO
# important!! record last voltage max above threshold and last section crossing. 
# Only return the last section crossing if the next voltage max has a higher calcium val.