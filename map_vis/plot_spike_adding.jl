function calc_traj(xmap, preimage, x0)
    len = 200
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


points =[
    #(-2.6096227169036865, 28.127906799316406), # homoclinic to saddle
    (-2.6527276039123535, 28.78907774658203), # snpo 1
    (-2.627365827560425, 26.775069900512695), # cusp 1
    (-2.487631320953369, 22.474958419799805), # snpo 2
    (-2.553318500518799, 27.644296813964844), # cusp 2
    (-2.4775397777557373, 22.507494888305664), # snpo 3
]
vs = [
 -54, -54, -53.95, -54.3, -53.42
]
dots = [
    -53.7, -53.35, -53.395, -53.85, -53.4175
]
labels = fill("", 5)
begin

    mapslider.sliders[2].value[] = .81
    mapslider.sliders[1].value[] = .6
    try close(sc4) 
    catch
        nothing
    end
    global sc4 = GLMakie.Screen(;resize_to = (1000, 5000))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 14,
            yticklabelsize = 14,
            xlabelsize = 18,
            ylabelsize = 18,
            yticks = WilkinsonTicks(3),
            xticks = WilkinsonTicks(3),
        )
    ))
    fig = Figure()
    # bifurcation diagram
    bax = Axis(fig[1:2,1:2], xlabel = "ΔCa", ylabel = "Δx")

    for i in eachindex(points)
        println(i)
        p[] = vcat(p[][1:15], [points[i]...])
        v0 = vs[i]
        # calculate saddle trajectory
        sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
        # set map limit to saddle image
        #mapslider.sliders[1].value[] = sad_lower[1,end]/eq[][5]
        # refine local minima
        refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
        ax = Axis(fig[2+i , 1], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect(), title = labels[i])

        maptraj = calc_traj(xmap[], preimage[], v0)
        lines!(ax, maptraj, color = :grey, linewidth = 1)
        # plot return map
        lines!(ax, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
        lines!(ax, preimage[], xmap[], color = :black, linewidth = 2)
        # saddle
        sd = min(sad_upper[3,end], sad_lower[3,end])
        ln_ = [
            (preimage[][end], sd),
            (preimage[][1], sd),
        ]
        lines!(ax, ln_, color = :green, linewidth = 2.0, linestyle = :dot)
        # plot fixed points

        vmin = sd - .1
        vmax = maximum(xmap[]) + .1
        ylims!(ax, vmin, vmax)
        xlims!(ax, vmin, vmax)
        # display bif point
        scatter!(ax, (dots[i], dots[i]), color = :red, marker = '✶', markersize = 20)
    end

    p[] = vcat(p[][1:15], [points[1]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[3,2], xlabel = L"[Ca]", ylabel = L"x")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,20000:end], sol[1,20000:end], color = :red, linewidth = 1)

    p[] = vcat(p[][1:15], [points[2]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[4,2], xlabel = L"[Ca]", ylabel = L"x")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,19000:end], sol[1,19000:end], color = :red, linewidth = 1)

    p[] = vcat(p[][1:15], [points[3]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[5,2], xlabel = L"[Ca]", ylabel = L"x")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,22000:end], sol[1,22000:end], color = :red, linewidth = 1)

    p[] = vcat(p[][1:15], [points[4]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[6,2], xlabel = L"[Ca]", ylabel = L"x")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,24000:end], sol[1,24000:end], color = :red, linewidth = 1)

    p[] = vcat(p[][1:15], [points[5]...])
    u0 = @SVector [0.5052, 0.0, 0.0, 0.0, 1.02, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[7,2], xlabel = L"[Ca]", ylabel = L"x")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,500:2000], sol[1,500:2000], color = :red, linewidth = 1)
    ix = findfirst(x -> x> -53.4194, preimage[])
    x = eq[][1]*ix/map_resolution
    ca = eq[][5]*ix/map_resolution
    fastu = solve(
                remake(ics_probs[1], 
                    p = vcat(p[][1:15], x, ca),
                    u0 = eq[][[3,4,6]]
                ),
                NewtonRaphson()
            ).u
    scatter!(bax, reverse.(points), color = :black, marker = '✶', markersize = 23)
    scatter!(bax, reverse.(points), color = :red, marker = '✶', markersize = 20)
    scatter!(bax, reverse.(points), color = :black, markersize = 7)
    #add labels to bax
    panel_labs = ["A", "B", "C", "D", "E"]
    for i in 1:5
        text!(bax, reverse(points[i]), text = panel_labs[i], color = :black, fontsize = 20)
    end
    

    resize!(fig, 650, 2000)
    display(sc4, fig)
end

save("spike_adding_chaos.png", fig)
