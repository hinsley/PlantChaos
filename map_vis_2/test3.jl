function calculate_hom_box(xmap, preimage)
    d = xmap.- preimage
    ixs = findall(i -> d[i]*d[i+1] < 0, 2:(length(d)-1))
    if length(ixs) < 2 
        return [[(NaN,NaN), (NaN,NaN)], [(NaN,NaN), (NaN,NaN)]]
    end
    (minx, mini) = findmin(xmap[1:ixs[end]])
    lns = []
    push!(lns, [(minx,minx), (preimage[mini], minx)])
    push!(lns, [(preimage[ixs[1]], preimage[ixs[1]]), (minx, preimage[ixs[1]])])
    return lns
end

primary_hom_line = @lift fill($xmap[1], map_resolution)
lines!(mapax, preimage, primary_hom_line, color = :white, linestyle = :dash, linewidth = 1)
lns = @lift calculate_hom_box($xmap, $preimage)
ln1 = @lift $lns[1]
ln2 = @lift $lns[2]