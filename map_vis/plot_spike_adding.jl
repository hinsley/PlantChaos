function calc_traj(xmap, preimage, x0)
    len = 100
    x = x0

    maptraj = fill(Point2f(NaN32,NaN32), len*2)
    lerp = linear_interpolation(preimage, xmap)
    maptraj[1] =  Point2f(x, x)
    maptraj[2] =  Point2f(x, lerp(x))
    for i=1:len-1
        x = lerp(x)
        p = Point2f(x, x)
        local p2
        try
            p2 = Point2f(x, lerp(x))
        catch
            break
        end
        maptraj[i*2+1] = p
        maptraj[i*2+2] = p2
    end
    maptraj
end

begin
    try close(sc3) 
    catch
        nothing
    end
    global sc3 = GLMakie.Screen(;resize_to = (1500, 800))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 30,
            yticklabelsize = 30,
            xlabelsize = 40,
            ylabelsize = 40,
        )
    ))
    fig = Figure()
    p[] = vcat(p[][1:15], [-1.078, -36.3154]);
    v0 = -48
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_lower[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax1 = Axis(fig[1,1], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(ax1, maptraj, color = :dodgerblue4, linewidth = 2)
    # plot return map
    lines!(ax1, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
    lines!(ax1, preimage[], xmap[], color = :black, linewidth = 2)
    # sf
    lines!(ax1, ln1[], color = :red, linewidth = 2.0, linestyle = :dot)
    # saddle
    ln_ = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end]),
        (NaN,NaN),
        (preimage[][end], sad_lower[3,end]),
        (preimage[][1], sad_lower[3,end])
    ]

    lines!(ax1, ln_, color = :green, linewidth = 2.0, linestyle = :dot)

    # plot fixed points
    ixs = findall(1:length(preimage[])-1) do i
        (preimage[][i]- xmap[][i])*(preimage[][i+1]- xmap[][i+1]) < 0
    end
    scatter!(ax1, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax1, eq[][6], eq[][6], color = :red, markersize = 10)

    # instide parabola
    p[] = vcat(p[][1:15], [-2.0403289794921875, -2.745530366897583]);
    v0 = -52
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    
    mapslider.sliders[1].value[] = sad_upper[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax2 = Axis(fig[1,2], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(ax2, maptraj, color = :dodgerblue4, linewidth = 2)
    # plot return map
    lines!(ax2, preimage[], preimage[], color = :dodgerblue4, linestyle = :dash, linewidth = 2,)
    lines!(ax2, preimage[], xmap[], color = :black, linewidth = 2)
    # sf
    lines!(ax2, ln1[], color = :red, linewidth = 2.0, linestyle = :dot)
    # saddle
    ln_ = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end])
    ]
    lines!(ax2, ln_, color = :green, linewidth = 2.0, linestyle = :dot)
    # saddle po
    ix = findlast(1:length(preimage[])) do i
        preimage[][i] > xmap[][i]
    end
    ln_ = [
        (preimage[][1], preimage[][ix]),
        (preimage[][ix], preimage[][ix]),
    ]
    lines!(ax2, ln_, color = :blue, linewidth = 2.0, linestyle = :dash)

    # plot fixed points
    ixs = findall(1:length(preimage[])-1) do i
        (preimage[][i]- xmap[][i])*(preimage[][i+1]- xmap[][i+1]) < 0
    end
    scatter!(ax2, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax2, eq[][6], eq[][6], color = :red, markersize = 10)

    # below parabola
    #p[] = vcat(p[][1:15], [-0.115, -0.115]);
    p[] = vcat(p[][1:15], [-1.5417746305465698, -23.116209030151367]);
    v0 = -51
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_upper[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax3 = Axis(fig[1,3], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(ax3, maptraj, color = :dodgerblue4, linewidth = 2)
    # plot return map
    lines!(ax3, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
    lines!(ax3, preimage[], xmap[], color = :black, linewidth = 2)
    # sf
    lines!(ax3, ln1[], color = :red, linewidth = 2.0, linestyle = :dot)
    # saddle
    ln_ = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end])
    ]
    lines!(ax3, ln_, color = :green, linewidth = 2.0, linestyle = :dot)
    # saddle po
    ix = findlast(1:length(preimage[])) do i
        preimage[][i] > xmap[][i]
    end
    ln_ = [
        (preimage[][1], preimage[][ix]),
        (preimage[][ix], preimage[][ix]),
    ]
    lines!(ax3, ln_, color = :blue, linewidth = 2.0, linestyle = :dash)

    # plot fixed points
    ixs = findall(1:length(preimage[])-1) do i
        (preimage[][i]- xmap[][i])*(preimage[][i+1]- xmap[][i+1]) < 0
    end
    scatter!(ax3, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax3, eq[][6], eq[][6], color = :red, markersize = 10)
    resize!(fig, round(Int,1000*1.5), round(Int, 860*1.5))
    resize_to_layout!(fig)
    display(sc3, fig)
end

