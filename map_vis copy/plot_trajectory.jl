function generate_trajectory(p)
    u0 = SVector{6}(.5, 0., 0., 0.0, .3, -30.)
    tspan = (0.0, 1000000.0)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p)
    sol = solve(prob, RK4(), saveat = 1.0)
    return (sol[5,:], sol[1,:], sol.t)
end



begin
    try close(sc3) 
    catch
        nothing
    end
    global sc3 = GLMakie.Screen(;resize_to = (1500, 800))
    set_theme!()
    fig = Figure()
    ax = Axis(fig[1:4,1], xlabel = "Ca", ylabel = "x")
    _tr_dat = @lift generate_trajectory($p)
    catr = @lift $_tr_dat[1]
    xtr = @lift $_tr_dat[2]
    ttr = @lift $_tr_dat[3]
    lines!(ax, catr, xtr, color = :black, linewidth = 2)
    catr_end = @lift $catr[end-100000:end]
    xtr_end = @lift $xtr[end-100000:end]
    lines!(ax, catr_end, xtr_end, color = :red, linewidth = 2)

    ax2 = Axis(fig[5,1], xlabel = "t", ylabel = "x")
    lines!(ax2, ttr, xtr, color = :black, linewidth = 2)

    display(sc3, fig)
end