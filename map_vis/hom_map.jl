map_resolution = 200

include("./return_map_utils.jl")
include("./hom_map_prob.jl")


# calculate and unpack all data needed for plotting
_ans = @lift calculate_return_map(monteprob,ics_probs, $p, $(mapslider.sliders[1].value),
    $(mapslider.sliders[2].value), resolution = map_resolution)
c_preimage = @lift $_ans[1]
cmap = @lift $_ans[2]
c_cass = @lift $_ans[3]
c_xss = @lift $_ans[4]
c_vss = @lift $_ans[5]


# plot the map
lines!(mapax, preimage, xmap, color = range(0.,1., length=map_resolution), colormap = :thermal)
lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)

# plot the trajectories in phase space
colorrng = range(0, 1, length = length(cass[]))
lines!(trajax, cass, xss, vss, color = colorrng, colormap = :thermal,
    linewidth = .2)

refine_prob = ODEProblem{false}(mapf, @SVector(zeros(BigFloat,6)), (BigFloat(0), BigFloat(1e5)), zeros(17))

# reset limits on changing section length
on(xss) do _
    reset_limits!(mapax)
    reset_limits!(trajax)
end

## TODO
# important!! record last voltage max above threshold and last section crossing. 
# Only return the last section crossing if the next voltage max has a higher calcium val.