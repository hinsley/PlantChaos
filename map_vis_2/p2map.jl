function iterate_map(xmap, preimage)
    newmap = similar(xmap)
    lerp = linear_interpolation(preimage, xmap)
    for i in 1:length(xmap)
        newmap[i] = lerp(xmap[i])
    end
    return newmap
end
function upscale_iterate_map(xmap, preimage, n, k)
    newmap = Array{Float64}(undef, n)
    lerp = scale(interpolate(xmap, BSpline(Quadratic(Reflect(OnCell())))), preimage)
    newpreimage = range(preimage[1], preimage[end], length = n)
    for i in 1:n
        newmap[i] = lerp(newpreimage[i])
        for j in 1:k-1
            newmap[i] = lerp(newmap[i])
        end
    end
    (collect(newpreimage), newmap) 
end

__ans = @lift upscale_iterate_map($xmap, range($preimage[1], $preimage[end], length = length($preimage)), 1000, 2)
p2pre = @lift $__ans[1]
p2map = @lift $__ans[2]
lines!(cmapax, preimage, preimage, color = :grey, linestyle = :dash, linewidth = 2,)
lines!(cmapax, p2pre, p2map, color = :white, linewidth = 2)