
points =[
    (-2.6096227169036865, 28.127906799316406), # homoclinic to saddle
    (-2.6527276039123535, 28.78907774658203), # snpo 1
    (-2.627365827560425, 26.775069900512695), # cusp 1
    (-2.487631320953369, 22.474958419799805), # snpo 2
    (-2.553318500518799, 27.644296813964844), # cusp 2
    (-2.4775397777557373, 22.507494888305664), # snpo 3
]
vs = [
    -54, -54, -54, -54, -54.3, -54.1
]
dots = [
    -54.39775, -53.7, -53.35, -53.395, -53.85, -53.4175
]
labels = [
    "homoclinic to saddle",
    "SNPO 1",
    "unstable cusp of cycles",
    "SNPO 2",
    "stable cusp of cycles",
    "SNPO 3",
]
begin
    mapslider.sliders[2].value[] = .81
    mapslider.sliders[1].value[] = .6
    try close(sc4) 
    catch
        nothing
    end
    global sc4 = GLMakie.Screen(;resize_to = (1500, 1500))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 18,
            yticklabelsize = 18,
            xlabelsize = 22,
            ylabelsize = 22,
            yticks = WilkinsonTicks(3),
            xticks = WilkinsonTicks(3),
        )
    ))
    fig = Figure()
    for i in eachindex(points)
        p[] = vcat(p[][1:15], [points[i]...])
        v0 = vs[i]
        # calculate saddle trajectory
        sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
        # set map limit to saddle image
        #mapslider.sliders[1].value[] = sad_lower[1,end]/eq[][5]
        # refine local minima
        refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
        ax = Axis(fig[ceil(Int, (i+4)/5) ,((i+3)%5)+1], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect(), title = labels[i])

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
        # display homoclinic to saddle line
        if i == 1
            j = findall(x -> x > .1, diff(xmap[]))[3]
            ln = [
                (preimage[][j], preimage[][j]),
                (preimage[][j], xmap[][j])
            ]
            lines!(ax, ln, color = :red, linewidth = 2.0, linestyle = :dash)
        end
        # display bif point
        scatter!(ax, (dots[i], dots[i]), color = :red, marker = '✶', markersize = 20)
    end
    p[] = vcat(p[][1:15], [points[1]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[1,2], xlabel = L"[Ca]", ylabel = L"x", title = "chaos with homoclinic \n to saddle")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,22000:end], sol[1,22000:end], color = :red, linewidth = 1)



    p[] = vcat(p[][1:15], [points[4]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[1,3], xlabel = L"[Ca]", ylabel = L"x", title = "ghost of SNPO 2")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,24000:end], sol[1,24000:end], color = :red, linewidth = 1)

    p[] = vcat(p[][1:15], [points[5]...])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    ax = Axis(fig[1,4], xlabel = L"[Ca]", ylabel = L"x", title = "multistability near \n SNPO 3")
    lines!(ax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 1)
    lines!(ax, sol[5,24000:end], sol[1,24000:end], color = :red, linewidth = 1)
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
    #u0 = @SVector [xs[1], 0.0, fastu[1], fastu[2], cas[i], fastu[3]]
    #tspan = (0., 1e6)
    #prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    #sol = solve(prob, RK4())
    #lines!(ax, sol[5,1000:end], sol[1,1000:end], color = :green, linewidth = 1)
    #lines!(ax, sol[5,20000:end], sol[1,20000:end], color = :red, linewidth = 1)
    

    resize!(fig, 1500, 600)
    display(sc4, fig)
end

save("spike_adding_chaos.png", fig)
