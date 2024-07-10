map_traj_len = last(mapslider.sliders[3].range[])
map_trajectory = Observable(fill(NaN, map_traj_len))
map_lerp = @lift linear_interpolation($preimage, $xmap)
map_traj_to_plot = Observable(fill(Point2f(NaN32,NaN32), map_traj_len*2))
on(p) do _
    map_trajectory[] = fill(NaN, map_traj_len)
    map_traj_to_plot[] = fill(Point2f(NaN32,NaN32), map_traj_len*2)
end

function generate_map_trajectory!(traj,N,mlerp)
    pre = mlerp.itp.knots[1]
    traj[1] = pre[ceil(Int, length(pre)/2)]
    for i=2:N
        traj[i] = mlerp(traj[i-1])
    end
    for i=N+1:length(traj)
        traj[i] = NaN
    end
end

function gen_map_plot_data!(data, traj)
    for i in 1:(length(traj)-1)
        data[i*2-1] = Point2f(traj[i], traj[i])
        data[i*2] = Point2f(traj[i], traj[i+1])
    end
end

@lift generate_map_trajectory!($map_trajectory, $(mapslider.sliders[3].value), $map_lerp)
@lift gen_map_plot_data!($map_traj_to_plot, $map_trajectory)

lines!(mapax, map_traj_to_plot, color = :white, linewidth = 1)

"""
period 2 blue sky
0.0003000000142492354
-2.4959588050842285
"""

function lyap(traj, preimage, map)
    n = length(traj)
    sum = 0.0
    last_idx = findfirst(isnan, traj) - 1
    for i in 1:last_idx
        j = findfirst(x -> x > traj[i], preimage)
        deriv = (map[j+1] - map[j])/(preimage[j+1] - preimage[j])
        sum += log(abs(deriv))
    end
    return sum/(n-1)
end

lyap(map_trajectory[], preimage[], xmap[])