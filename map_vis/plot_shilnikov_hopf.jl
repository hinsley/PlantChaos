function calc_traj(xmap, preimage, x0, len = 100)
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

include("./plot_nullclines.jl")

_ps = [[-1.078, -36.3154],
    [-2.0403289794921875, -2.745530366897583],
    [-1.5417746305465698, -23.116209030151367],
    [-1.690864158630371, -16.559186935424805]]

begin
    mapslider.sliders[2].value[] = 1.0
    try close(sc3) 
    catch
        nothing
    end
    global sc3 = GLMakie.Screen(;resize_to = (1500, 800))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 20,
            yticklabelsize = 20,
            xlabelsize = 25,
            ylabelsize = 25,
        )
    ))
    fig = Figure()
    p[] = vcat(p[][1:15], _ps[1]);
    v0 = -46.15995
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_lower[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax1 = Axis(fig[1,2], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0, 10)
    lines!(ax1, maptraj, color = :grey, linewidth = 2)
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
    #scatter!(ax1, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax1, eq[][6], eq[][6], color = :red, markersize = 10)

    # instide parabola
    p[] = vcat(p[][1:15], _ps[2]);
    v0 = -52
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    
    mapslider.sliders[1].value[] = sad_upper[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax2 = Axis(fig[2,3], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(ax2, maptraj, color = :grey, linewidth = 2)
    # plot return map
    lines!(ax2, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
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
    #scatter!(ax2, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax2, eq[][6], eq[][6], color = :red, markersize = 10)

    # below parabola
    #p[] = vcat(p[][1:15], [-0.115, -0.115]);
    p[] = vcat(p[][1:15], _ps[3]);
    v0 = -51
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_upper[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax3 = Axis(fig[2,1], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(ax3, maptraj, color = :grey, linewidth = 2)
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
    #scatter!(ax3, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax3, eq[][6], eq[][6], color = :red, markersize = 10)

    # hom to po
    #p[] = vcat(p[][1:15], [-0.115, -0.115]);
    p[] = vcat(p[][1:15], _ps[4]);
    v0 = -48.34997
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_upper[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
    ax4 = Axis(fig[3,2], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    maptraj = calc_traj(xmap[], preimage[], v0, 30)
    lines!(ax4, maptraj, color = :grey, linewidth = 2)
    # plot return map
    lines!(ax4, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
    lines!(ax4, preimage[], xmap[], color = :black, linewidth = 2)
    # sf
    lines!(ax4, ln1[], color = :red, linewidth = 2.0, linestyle = :dot)
    # saddle
    ln_ = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end])
    ]
    lines!(ax4, ln_, color = :green, linewidth = 2.0, linestyle = :dot)
    ix = findlast(1:length(preimage[])) do i
        preimage[][i] > xmap[][i]
    end
    ln_ = [
        (preimage[][1], preimage[][ix]),
        (preimage[][ix], preimage[][ix]),
    ]
    lines!(ax4, ln_, color = :blue, linewidth = 2.0, linestyle = :dash)
    # saddle po
    ixs = findall(1:length(preimage[])-1) do i
        (preimage[][i]- xmap[][i])*(preimage[][i+1]- xmap[][i+1]) < 0
    end
    #scatter!(ax4, preimage[][ixs], preimage[][ixs], color = :transparent, strokewidth =1, markersize = 10, strokecolor = :blue)
    scatter!(ax4, eq[][6], eq[][6], color = :red, markersize = 10)

    resize!(fig, round(Int,1000*1.5), round(Int, 860*1.5))
    resize_to_layout!(fig)
    display(sc3, fig)

# middle panel and labels
    axmid = Axis(fig[2,2])
    hidedecorations!(axmid)
    xs = collect(-1.5:0.01:1.5)
    lines!(axmid, xs, -1 .* xs.^2, color = :orange, linewidth = 2)
    lines!(axmid, [(-1.7, 0.0), (1.7, 0)], color = :green, linewidth = 2, linestyle = :dash)
    lines!(axmid, [(0.0, 0.0), (0.0, .7)], color = :red, linewidth = 2)
    points = [(0,.5), (1.25, -.5), (1,-1), (.5, -1.25)]
    scatter!(axmid, points, color = :black, markersize = 15)
    ylims!(axmid, (-1.5, .6))
    xlims!(axmid, (-1.5, 1.5))
    labels = ["A", "B", "C", "D"]
    for (i, p) in enumerate(points)
        text!(axmid, p.- (-.075, .075), text = labels[i], color = :black, fontsize = 20)
    end
    scatter!(axmid, (0,0), color = :blue, markersize = 15, marker = :x)
    pos = (ax1.xaxis.attributes.limits[][1] -.23, ax1.yaxis.attributes.limits[][2] - .6)
    text!(ax1, pos, text = "A2", color = :black, fontsize = 25)
    pos = (ax2.xaxis.attributes.limits[][1] , ax2.yaxis.attributes.limits[][2] - .6)
    text!(ax2, pos, text = "B2", color = :black, fontsize = 25)
    pos = (ax3.xaxis.attributes.limits[][1] -.18 , ax3.yaxis.attributes.limits[][2] - .6)
    text!(ax3, pos, text = "D2", color = :black, fontsize = 25)
    pos = (ax4.xaxis.attributes.limits[][1] , ax4.yaxis.attributes.limits[][2] - .6)
    text!(ax4, pos, text = "C2", color = :black, fontsize = 25)
    text!(axmid, (0.0, 0.0), text = L"ShH", color = :blue, fontsize = 20)
    text!(axmid, (-1.0, 0.0), text = L"AH_{sub}", color = :green, fontsize = 20)
    text!(axmid, (0.0, 0.25), text = L"hom_{SF}", color = :red, fontsize = 20)
    text!(axmid, (-1.5, -1.0), text = L"hom_{PO}", color = :orange, fontsize = 20)


# trajectories
    a1 = Axis(fig[1,1], xlabel = "Ca", ylabel = "x")
    a2 = Axis(fig[1,3], xlabel = "Ca", ylabel = "x")
    a3 = Axis(fig[3,3], xlabel = "Ca", ylabel = "x")
    a4 = Axis(fig[3,1], xlabel = "Ca", ylabel = "x")
    axs = [a1, a2, a3, a4]


p[] = vcat(p[][1:15], _ps[1]);
plot_nullclines!(a1, sad_lower, sad_upper, p[])

ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.001, 0.0])
tspan = (0.0, 70000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol = solve(prob, RK4(), saveat = 1.0)
lines!(a1, sol[5,:], sol[1,:], color = :black, linewidth = 2,)
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.0015, 0.0])
tspan = (0.0, 70000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol2 = solve(prob, RK4(), saveat = 1.0)
lines!(a1, sol2[5,:], sol2[1,:], color = :grey, linewidth = 2,)
xlims!(a1, minimum(sol[5,:])-.02, maximum(sol[5,:])+.02)
ylims!(a1, minimum(sol[1,:])-.02, maximum(sol[1,:])+.02)

p[] = vcat(p[][1:15], _ps[2]);
plot_nullclines!(a2, sad_lower, sad_upper, p[])
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.037, 0.0])
tspan = (0.0, 1000000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol = solve(prob, RK4(), saveat = 1.0)
lines!(a2, sol[5,:], sol[1,:], color = :red, linewidth = .2,)
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.04, 0.0])
tspan = (0.0, 500000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol = solve(prob, RK4(), saveat = 1.0)
lst = 300000
lines!(a2, sol[5,:], sol[1,:], color = :grey, linewidth = 2,)
lines!(a2, sol[5,1:lst], sol[1,1:lst], color = :black, linewidth = 2,)
xlims!(a2, minimum(sol[5,:])-.02, maximum(sol[5,:])+.02)
ylims!(a2, minimum(sol[1,:])-.02, maximum(sol[1,:])+.02)

p[] = vcat(p[][1:15], _ps[3]);
plot_nullclines!(a4, sad_lower, sad_upper, p[])
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.1, 0.0])
tspan = (0.0, 1000000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol = solve(prob, RK4(), saveat = 1.0)
lines!(a4, sol[5,:], sol[1,:], color = :red, linewidth = 1,)
lst = 75000
lines!(a4, sol[5,1:lst], sol[1,1:lst], color = :black, linewidth = 2,)
xlims!(a4, minimum(sol[5,:])-.02, maximum(sol[5,:])+.02)
ylims!(a4, minimum(sol[1,:])-.02, maximum(sol[1,:])+.02)

p[] = vcat(p[][1:15], _ps[4]);
plot_nullclines!(a3, sad_lower, sad_upper, p[])
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.029, 0.0])
tspan = (0.0, 530000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol = solve(prob, RK4(), saveat = 1.0)
lines!(a3, sol[5,:], sol[1,:], color = :black, linewidth = 1,)
ics = SVector(eq[] .+ [-0.00096, 0.0, 0.0, 0.0, -0.025, 0.0])
tspan = (0.0, 1000000.0)
prob = ODEProblem(Plant.melibeNew, ics, tspan, p[])
sol2 = solve(prob, RK4(), saveat = 1.0)
lines!(a3, sol2[5,:], sol2[1,:], color = :red, linewidth = .4,)
xlims!(a3, minimum(sol[5,:])-.02, maximum(sol[5,:])+.02)
ylims!(a3, minimum(sol[1,:])-.02, maximum(sol[1,:])+.02)

# more labels
    text!(a1, (0.575, 0.85), text = "A1", color = :black, fontsize = 25)
    text!(a2, (0.655, 0.86), text = "B1", color = :black, fontsize = 25)
    text!(a3, (0.685, 0.86), text = "C1", color = :black, fontsize = 25)
    text!(a4, (0.7, 0.875), text = "D1", color = :black, fontsize = 25)
end

save("shilnikov_hopf.png", fig)