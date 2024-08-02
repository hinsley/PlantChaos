lcarr = load("lyapunov_3color.jld2")["lcarr"]
include("./plot_nullclines.jl")

start_p = [Plant.default_params...]
start_p[17] = 15 # Cashift
start_p[16] = -2.8 # xshift
end_p = [Plant.default_params...]
end_p[17] = 45 # Cashift
end_p[16] = -2.2 # xshift

resolution = 700 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)

# hand drawn curves
# pd curve
pd_curve = [
    (44.898616790771484, -2.5857670307159424),
    (42.10068313598633, -2.6179141998291016),
    (39.510860443115234, -2.631117582321167),
    (37.64164733886719, -2.6336562633514404),
    (35.76277160644531, -2.6273014545440674),
    (34.968807220458984, -2.6242096424102783),
    (33.055084228515625, -2.6132869720458984),
    (31.490755081176758, -2.599351406097412),
    (28.965070724487305, -2.5693960189819336),
    (28.869434356689453, -2.567809820175171), # cusp bg
]
snpog_curve = [
    (28.869434356689453, -2.567809820175171), # cusp bg
    (25.682607650756836, -2.5280463695526123),
    (20.857946395874023, -2.450077772140503), # multi snpo
    (17.44451332092285, -2.3881523609161377),
    (15.042675018310547, -2.3397936820983887)
]
snpoa_curve = [
    (18.417884826660156, -2.4305918216705322), # cusp ab
    (19.549236297607422, -2.4417343139648438),
    (20.857946395874023, -2.450077772140503), # multi snpo
    (22.6110897064209, -2.4533474445343018),
    (23.044034957885742, -2.454495429992676),
    (25.016345977783203, -2.4435882568359375),
    (26.67597198486328, -2.4223482608795166),
    (27.884729385375977, -2.3990349769592285),
    (29.310068130493164, -2.3621816635131836),
    (31.700477600097656, -2.2764272689819336),
    (33.274288177490234, -2.201303482055664),
]
snpob_curve = [
    (18.417884826660156, -2.4305918216705322), # cusp ab
    (22.496091842651367, -2.4877092838287354),
    (28.869434356689453, -2.567809820175171), # cusp bg
]

snpo0_curve = [
    (25.93706512451172, -2.6712090969085693), # Bautin
    (32.262168884277344, -2.7767200469970703),
    (33.71145248413086, -2.799856662750244),
]

ah_sup_curve = [
    (25.93706512451172, -2.6712090969085693), # Bautin
    (32.58976745605469, -2.799747943878174),
]
ah_sub_curve = [
    (25.93706512451172, -2.6712090969085693), # Bautin
    (15.005159378051758, -2.452031373977661),
]

points =[
    (18.23660659790039, -2.4285969734191895), # cusp ab
    (28.869434356689453, -2.567809820175171), # cusp bg
    (20.836591720581055, -2.4494824409484863), # multi snpo
    (25.93706512451172, -2.6712090969085693), # Bautin
    ]

function calc_traj(xmap, preimage, x0; len = 200)
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

function plotmap!(ax, v0; final = 0, len = 200)
        # calculate saddle trajectory
        sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
        # refine local minima
        refine_map!(remake(map_prob, p = (p = p[], eq = eq[])), lerp[], xmap, preimage)
        maptraj = calc_traj(xmap[], preimage[], v0, len = len)
        lines!(ax, maptraj, color = :grey, linewidth = 1)
        if final > 0
            lines!(ax, maptraj[end-final:end], color = :red, linewidth = 1)
        end
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
        nothing
end

p[] = vcat(p[][1:15], [reverse(points[1])...])


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
    bax = Axis(fig[1:2,1:2], xlabel = L"Δ_\text{[Ca]}", ylabel = L"Δ_x")
    image!(bax, Ca_shifts, x_shifts, rotr90(lcarr))
    lines!(bax, [pd_curve...], color = :white, linestyle = :dot, linewidth = 3)
    lines!(bax, [snpog_curve...], color = :orange, linewidth = 3)
    lines!(bax, [snpog_curve...], color = :black, linestyle = :dot, linewidth = 3)
    lines!(bax, [snpoa_curve...], color = :yellow, linewidth = 3)
    lines!(bax, [snpoa_curve...], color = :black, linestyle = :dot, linewidth = 3)
    lines!(bax, [snpo0_curve...], color = :red, linewidth = 3)
    lines!(bax, [snpob_curve...], color = :green, linewidth = 3)
    lines!(bax, [snpob_curve...], color = :black, linestyle = :dot, linewidth = 3)
    lines!(bax, [ah_sub_curve...], color = :grey, linestyle = :dash, linewidth = 3)
    lines!(bax, [ah_sup_curve...], color = :grey, linestyle = :dash, linewidth = 3)

    scatter!(bax, points[1], color = :lawngreen, marker = '+', markersize = 30)
    scatter!(bax, points[2], color = :black, marker = '*', markersize = 55)
    scatter!(bax, points[2], color = :white, marker = '*', markersize = 45)
    scatter!(bax, points[2], color = :olive, marker = '*', markersize = 43)
    scatter!(bax, points[3], color = :black, marker = 'o' , markersize = 25)
    scatter!(bax, points[3], color = :gold, marker = 'o' , markersize = 20)
    scatter!(bax, points[3], color = :black , markersize = 8)
    scatter!(bax, points[4], color = :red, marker = '□' , markersize = 25)
    text!(bax, pd_curve[5]..., text = L"\textbf{PD}", color = :white, fontsize = 16)
    text!(bax, snpog_curve[4]..., text = L"\textbf{SN_{PO}-\gamma / PD}", color = :white, fontsize = 16, rotation = -.7)
    text!(bax, snpog_curve[4]..., text = L"\textbf{SN_{PO}-\gamma}", color = :orange, fontsize = 16, rotation = -.7)
    text!(bax, snpoa_curve[7].+ (1.5,0.) ..., text = L"\textbf{SN_{PO}-\alpha}", color = :yellow, fontsize = 16, rotation = .95)
    text!(bax, snpob_curve[2].+ (-1,-.01) ..., text = L"\textbf{SN_{PO}-\beta}", color = :white, fontsize = 16, rotation = -.6)
    text!(bax, snpob_curve[2].+ (-1,-.01) ..., text = L"\textbf{SN_{PO}-\beta}", color = :green, fontsize = 16, rotation = -.6)
    text!(bax, snpo0_curve[1].+ (3,-.048) ..., text = L"\textbf{SN_{PO}-0}", color = :red, fontsize = 16, rotation = -.65)
    text!(bax, ah_sup_curve[1].+ (1.5,-.065) ..., text = L"\textbf{AH_{sup}}", color = :lightgrey, fontsize = 16, rotation = -.7)
    text!(bax, ah_sub_curve[1].+ (-5,.065) ..., text = L"\textbf{AH_{sub}}", color = :lightgrey, fontsize = 16, rotation = -.7)
    
    # alpha_beta cusp
    ax1 = Axis(fig[1,3], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    p[] = vcat(p[][1:15], [reverse(points[1])...])
    plotmap!(ax1, -54.1)
    scatter!(ax1, (-53.46,-53.46), color = :black, marker = '+', markersize = 45)
    scatter!(ax1, (-53.46, -53.46), color = :lawngreen, marker = '+', markersize = 40)

    ax2 = Axis(fig[1,4], xlabel = L"\text{[Ca]}", ylabel = L"x")

    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    plot_nullclines!(ax2, sad_lower, sad_upper, p[])

    u0 = @SVector [0.53, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 5e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    
    lines!(ax2, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = .4)
    lines!(ax2, sol[5,40000:50000], sol[1,40000:50000], color = :red, linewidth = 1)
    limits!(ax2, minimum(sol[5,:]) - .05, maximum(sol[5,:]) + .05, minimum(sol[1,:]) - .05, maximum(sol[1,:]) + .05)

    # beta gamma cusp
    ax3 = Axis(fig[2,3], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    p[] = vcat(p[][1:15], [reverse(points[2])...])
    plotmap!(ax3, -54.4)
    scatter!(ax3, (-53.94,-53.94), color = :black, marker = '*', markersize = 55)
    scatter!(ax3, (-53.94,-53.94), color = :olive, marker = '*', markersize = 50)

    ax4 = Axis(fig[2,4], xlabel = L"\text{[Ca]}", ylabel = L"x")
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    plot_nullclines!(ax4, sad_lower, sad_upper, p[])
    u0 = @SVector [0.5, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 1e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())
    
    lines!(ax4, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = .4)
    lines!(ax4, sol[5,end-2000:end], sol[1,end-2000:end], color = :red, linewidth = 1)
    limits!(ax4, minimum(sol[5,:]) - .05, maximum(sol[5,:]) + .05, minimum(sol[1,:]) - .05, maximum(sol[1,:]) + .05)

    # multi snpo
    ax5 = Axis(fig[3,3], xlabel = L"V_n", ylabel = L"V_{n+1}", aspect = DataAspect())
    p[] = vcat(p[][1:15], [reverse(points[3])...])
    plotmap!(ax5, -53.295)
    scatter!(ax5, (-53.83,-53.83), color = :black, marker = 'o', markersize = 25)
    scatter!(ax5, (-53.83,-53.83), color = :gold, marker = 'o', markersize = 20)
    scatter!(ax5, (-53.285,-53.285), color = :black, marker = 'o', markersize = 25)
    scatter!(ax5, (-53.285,-53.285), color = :gold, marker = 'o', markersize = 20)

    ax6 = Axis(fig[3,4], xlabel = L"\text{[Ca]}", ylabel = L"x")
    sad_upper, sad_lower = get_saddle_traj(remake(map_prob, p = (p = p[], eq = eq[])), p[])
    plot_nullclines!(ax6, sad_lower, sad_upper, p[])
    u0 = @SVector [0.5694, 0.0, 0.0, 0.0, 1.0, -53.5]
    tspan = (0., 4e6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p[])
    sol = solve(prob, RK4())

    lines!(ax6, sol[5,:], sol[1,:], color = :dodgerblue4, linewidth = .4)
    lines!(ax6, sol[5,end-4500:end-3500], sol[1,end-4500:end-3500], color = :red, linewidth = 1)
    lines!(ax6, sol[5,5000:7000], sol[1,5000:7000], color = :red, linewidth = 1)

    limits!(ax6, minimum(sol[5,:]) - .05, maximum(sol[5,:]) + .05, minimum(sol[1,:]) - .05, maximum(sol[1,:]) + .05)

    # stability of blue sky
    ax7 = Axis(fig[3,2], aspect = DataAspect())
    pt = (42.00076675415039, -2.9138572216033936)
    p[] = vcat(p[][1:15], [reverse(pt)...])
    plotmap!(ax7, -54.6, len = 20000, final = 5000)


    # bautin unfolding
    ax8 = Axis(fig[3,1])
    hidedecorations!(ax8)
    x1 = range(-1,0, length = 100)
    x2 = range(-1,2, length = 100)
    x3 = range(-2,2, length = 100)
    y1 = -x1.^2
    y2 = @. x2 - exp(-x2) + exp(1)
    lines!(ax8, x1, y1, color = :red, linestyle = :dot, linewidth = 3)
    lines!(ax8, x2, y2, color = :red, linewidth = 3)
    lines!(ax8, x3, fill(0.0, 100), color = :grey, linestyle = :dash, linewidth = 3)
    scatter!(ax8, [0.0], [0.0], color = :red, marker = '□', markersize = 20)
    scatter!(ax8, [(-1,-1)], color = :black, markersize = 15)
    xlims!(ax8, -2, 2)
    text!(ax8, (-1.5, 0.1), text = L"\textbf{AH_{sub}}", color = :grey, fontsize = 16)
    text!(ax8, (.5, 0.1), text = L"\textbf{AH_{sup}}", color = :grey, fontsize = 16)
    text!(ax8, (-.3, -.5), text = L"\textbf{Bautin}", color = :red, fontsize = 16)
    text!(ax8, (-.9, -1.1), text = L"\textbf{cusp}", color = :black, fontsize = 16)
    text!(ax8, (0, 1.8), text = L"\textbf{SN_{PO}-0}", color = :red, fontsize = 16, rotation = .85)
    text!(ax8, (-1.5, 2.5), text = L"↖Ch∀0s", color = :black, fontsize = 20)

    # panel labels
    text!(bax, (15.2, -2.225), text = L"\textbf{A}", color = :black, fontsize = 20)
    text!(ax8, (-1.9, 4.3), text = L"\textbf{B}", color = :black, fontsize = 20)
    text!(ax7, (-54.65, -53.33), text = L"\textbf{C}", color = :black, fontsize = 20)
    text!(ax1, (-54.13, -52.7), text = L"\textbf{D}", color = :black, fontsize = 20)
    text!(ax2, (0.65, 0.888), text = L"\textbf{E}", color = :black, fontsize = 20),
    text!(ax3, (-54.59, -53.9), text = L"\textbf{F}", color = :black, fontsize = 20)
    text!(ax4, (0.685, 0.887), text = L"\textbf{G}", color = :black, fontsize = 20)
    text!(ax5, (-54.27, -53.29), text = L"\textbf{H}", color = :black, fontsize = 20)
    text!(ax6, (0.665, 0.89), text = L"\textbf{I}", color = :black, fontsize = 20)
    
    resize!(fig, 1350, 1000)
    display(sc4, fig)
end

save("spike_adding_chaos.png", fig)