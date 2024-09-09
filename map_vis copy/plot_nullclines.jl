
function plot_nullclines!(trajax, sad_lower, sad_upper, p; ca_pad1 = 5, ca_pad2 = 0, x_pad1 = 20, xpad2 = 5)
    # ca nullcline
    vs = range(-80, -20, length = 500) |> collect
    xs = Plant.xinf.(Ref(p), vs)
    casca = p[13].*xs.*(p[12].-vs.+p[17])
    # select range
    camin = sad_lower[1,end]
    camax = sad_upper[1,1]
    xmin = minimum(sad_lower[2,:])
    xmax = maximum(sad_upper[2,:])
    ixbeg = findfirst(x -> x > xmin, xs)
    icabeg = findfirst(x -> x > camin, casca)
    ixend = findlast(x -> x < xmax, xs)
    icaend = findlast(x -> x < camax, casca)
    ibeg = max(ixbeg, icabeg) - ca_pad1
    iend = min(ixend, icaend) + ca_pad2
    casca = casca[ibeg:iend]
    xsca = xs[ibeg:iend]
    lines!(trajax, casca, xsca, color = :red, linewidth = 2, linestyle = :dash)

    # x nullcline
    Q = Plant.II.(Ref(p), Plant.hinf.(vs), vs) .+
        Plant.IK.(Ref(p), Plant.ninf.(vs), vs) .+
        Plant.IT.(Ref(p), xs, vs) .+
        Plant.Ileak.(Ref(p), vs)

    casx  = @. .5*Q/(p[7]*(-vs+p[9]) - Q)
    # select range
    icabeg = findfirst(x -> x > camin, casx)
    icaend = findlast(x -> x < 0, casx)
    icaend = findlast(x -> x > 0, casx[1:icaend])
    icaend = findlast(x -> x < camax, casx[1:icaend])
    ibeg = max(icabeg, ixbeg) - x_pad1
    iend = min(icaend, ixend) + xpad2
    casx2 = casx[ibeg:iend]
    xsx = xs[ibeg:iend]
    # separate stable from unstable
    xstable = Float64[]
    xunstable = Float64[]
    cstable = Float64[]
    cunstable = Float64[]
    eold = casx2[1]
    e2old = casx2[2]
    for i in 1:length(casx2)-1
        e = casx2[i]
        e2 = casx2[i+1]
        if e<e2
            push!(xunstable, xsx[i])
            push!(cunstable, casx2[i])
        else
            push!(xstable, xsx[i])
            push!(cstable, casx2[i])
        end
        if (eold-e2old)*(e-e2)<0
            push!(xstable,NaN)
            push!(cstable, NaN)
            push!(xunstable, NaN)
            push!(cunstable, NaN)
        end
        eold = e
        e2old = e2
    end
    lines!(trajax, cstable, xstable, color = :black, linewidth = 2)
    lines!(trajax, cunstable, xunstable, color = :black, linewidth = 2, linestyle = :dash)
    nothing
end