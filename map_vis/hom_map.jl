c_map_resolution = 50000

include("./return_map_utils.jl")
include("./hom_map_prob.jl")


# calculate and unpack all data needed for plotting
_ans = @lift calculate_circle_map(c_monteprob, c_ics_probs, $p, $(mapslider.sliders[5].value),
    $(mapslider.sliders[6].value), $(mapslider.sliders[4].value), resolution = c_map_resolution)
c_preimage = @lift $_ans[1]
cmap = @lift $_ans[2]
c_cass = @lift $_ans[3]
c_xss = @lift $_ans[4]
c_vss = @lift $_ans[5]


# plot the map
lines!(cmapax, c_preimage, cmap, color = range(0.,1., length=c_map_resolution), colormap = :thermal)
lines!(cmapax, c_preimage, c_preimage, color = :white, linestyle = :dash, linewidth = 2,)

# plot the trajectories in phase space
colorrng = range(0, 1, length = length(cass[]))

pcass = cass
pxss = xss
pvss = vss

cmap_trajectory = Observable(false)

on(switch_traj_button.clicks) do _
    if cmap_trajectory[]
        pcass[] = c_cass[]
        pxss[] = c_xss[]
        pvss[] = c_vss[]
    else
        pcass[] = cass[]
        pxss[] = xss[]
        pvss[] = vss[]
    end
    cmap_trajectory[] = !cmap_trajectory[]
end

# plot the trajectories in phase space
colorrng = range(0, 1, length = length(cass[]))
lines!(trajax, pcass, pxss, pvss, color = colorrng, colormap = :thermal,
    linewidth = .2)

# reset limits on changing section length
on(c_xss) do _
    reset_limits!(cmapax)
    reset_limits!(trajax)
end

## TODO
# important!! record last voltage max above threshold and last section crossing. 
# Only return the last section crossing if the next voltage max has a higher calcium val.