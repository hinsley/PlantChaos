# refine
begin
    mx = maximum(xmap[])
    for i in 2:length(xmap[])-1
        e = xmap[][i]
        if e == mx
            xmap[] = vcat(xmap[][1:i-1], (xmap[][i-1]+xmap[][i+1])/2, xmap[][i+1:end])
        end
    end
end


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

# shilnikov hopf

include("./plot_nullclines.jl")

fig2 = let
    try close(sc2) 
    catch
        nothing
    end
    global sc2 = GLMakie.Screen(;resize_to = (1000, 1000))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 30,
            yticklabelsize = 30,
            xlabelsize = 40,
            ylabelsize = 40,
        )
    ))
    fig = Figure()

    # on the homoclinic
    p[] = vcat(p[][1:15], [-1.078, -36.3154]);
    v0 = -48 # for map
    
    # calculate saddle trajectory
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    # set map limit to saddle image
    mapslider.sliders[1].value[] = sad_lower[1,end]/eq[][5]
    # refine local minima
    refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)

    # plot return map
    mapax = Axis(fig[1:4,1], xlabel = L"V_n", ylabel = L"V_{n+1}", limits = ((-53.5,-46), (-53.5,-46)), aspect = DataAspect())
    colorrng = range(0, 1, length = length(xmap[])) |> collect
    lines!(mapax, preimage[], xmap[], color = colorrng, colormap = Reverse(:RdYlGn_10), linewidth = 4)
    lines!(mapax, preimage[], preimage[], color = :grey, linestyle = :dash, linewidth = 2,)
    # saddle focus
    lines!(mapax, ln1[], color = :red, linewidth = 2.0, linestyle = :dot)
    # image of saddle
    ln_ = [
        (preimage[][end], sad_upper[3,end]),
        (preimage[][1], sad_upper[3,end]),
        (NaN,NaN),
        (preimage[][end], sad_lower[3,end]),
        (preimage[][1], sad_lower[3,end])
    ]

    lines!(mapax, ln_, color = :green, linewidth = 2.0, linestyle = :dot)
   
    # map trajectory
    
    maptraj = calc_traj(xmap[], preimage[], v0)
    lines!(mapax, maptraj, color = :dodgerblue4, linewidth = 2)
    
    # ODE trajectory
    trajax = Axis(fig[1:4,2], xlabel = L"\text{[Ca]}", ylabel = L"x")
    tax = Axis(fig[5,:], ylabel = L"V(t)", xlabel = L"t", yticks = [-50,0])
    hidexdecorations!(tax)
    hidespines!(tax, :t,:r)
    tanax = Axis(fig[7,:], ylabel = L"\theta(t)", xlabel = L"t")
    caax = Axis(fig[6,:], ylabel = L"\text{[Ca]}", xlabel = L"t")
    hidexdecorations!(caax)
    hidespines!(tanax, :t,:r)
    hidespines!(caax, :t,:r)
    # calculate u0 from v0
    u0 = lerp[](v0)
    # solve trajectory
    prob = ODEProblem(Plant.melibeNew, u0, (0., 300000.), p[])
    sol = solve(prob, RK4())

    # plot nullclines
    plot_nullclines!(trajax, sad_lower, sad_upper, p[])

    # plot trajectories that generate map
    colorrng = range(0, 1, length = length(cass[])) |> collect
    lines!(trajax, cass[], xss[], color = colorrng, colormap = Reverse(:RdYlGn_10), linewidth = 1.0)
    # plot trajectory
    lines!(trajax, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = 2)
    # plot time series
    lines!(tax, sol.t, sol[6,:], color = :dodgerblue4, linewidth = 2)

    Θ = atan.(sol[1,:]./sol[5,:]).-atan(eq[][1]/eq[][5])
    lines!(tanax, sol.t, Θ, color = :dodgerblue4, linewidth = 2)
    lines!(caax, sol.t, sol[5,:], color = :dodgerblue4, linewidth = 2)
    

    crossings = Int[]
    θref = atan(eq[][1]/eq[][5])
    for i in 1:length(sol)-1
        x = sol[1,i]
        ca = sol[5,i]
        θ = atan(x/ca)
        
        x2 = sol[1,i+1]
        ca2 = sol[5,i+1]
        θ2 = atan(x2/ca2)

        # calculate zero crossings
        if ((θ - θref) * (θ2 - θref) < 0) && (θ < θ2)
            push!(crossings, i)
        end
    end
    # plot ca peaks on phase plane

    caps = (sol[5, crossings].+ sol[5, crossings.+1])./2
    xps = (sol[1, crossings].+ sol[1, crossings.+1])./2

    scatter!(trajax, caps, xps, color = :black, markersize = 12)
    scatter!(trajax, caps, xps, color = :red, markersize = 8)
    # plot ca peaks on time series
    
    scatter!(tanax, sol.t[crossings], Θ[crossings], color = :black, markersize = 12)
    scatter!(tanax, sol.t[crossings], Θ[crossings], color = :red, markersize = 8)
    scatter!(caax, sol.t[crossings], sol[5,crossings], color = :black, markersize = 12)
    scatter!(caax, sol.t[crossings], sol[5,crossings], color = :red, markersize = 8)
    # plot ca peaks on return map
    scatter!(mapax, [e[1] for e in maptraj[1:2:end-2]], [e[1] for e in maptraj[3:2:end]], color = :black, markersize = 12)
    scatter!(mapax, [e[1] for e in maptraj[1:2:end-2]], [e[1] for e in maptraj[3:2:end]], color = :red, markersize = 8)

    lines!(trajax, sad_upper, color = :green, linewidth = 2.0)
    lines!(trajax, sad_lower, color = :green, linewidth = 2.0)

    mapax.limits[] = ((preimage[][1] - 0.1, preimage[][end] + 0.1), (preimage[][1] - 0.1, preimage[][end] + 0.1))

    # inside the parabola
    #p[] = SVector{17}(vcat(p[][1:15], [-1.52, -21.]))

    # plot Equilibria
    scatter!(trajax, [(eq[][5], eq[][1])], color = :black, markersize = 38, marker = '♦')
    scatter!(trajax, [(eq[][5], eq[][1])], color = :red, markersize = 30, marker = '♦')
    scatter!(trajax, [(sad_lower[1,1], sad_lower[2,1])], color = :black, markersize = 38, marker = '★')
    scatter!(trajax, [(sad_upper[1,end], sad_upper[2,end])], color = :black, markersize = 38, marker = '★')
    scatter!(trajax, [(sad_upper[1,end], sad_upper[2,end])], color = :green, markersize = 30, marker = '★')

    scatter!(mapax, [(eq[][6], eq[][6])], color = :black, markersize = 38, marker = '♦')
    scatter!(mapax, [(eq[][6], eq[][6])], color = :red, markersize = 30, marker = '♦')
    scatter!(mapax, [(sad_lower[3,end], sad_lower[3,end])], color = :black, markersize = 38, marker = '★')
    scatter!(mapax, [(sad_upper[3,end], sad_upper[3,end])], color = :green, markersize = 30, marker = '★')

    hidedecorations!(trajax, ticks = false, label = false, ticklabels = false)
    hidedecorations!(mapax, ticks = false, label = false, ticklabels = false)
    hidedecorations!(tax, ticks = false, label = false, ticklabels = false)
    hidedecorations!(tanax, ticks = false, label = false, ticklabels = false)
    
    set_theme!(theme_black())
    display(sc2, fig)
    resize!(fig, round(Int,1000*1.5), round(Int, 860*1.5))
    resize_to_layout!(fig)
    fig
end

save("homoclinic_map.png", fig2)
