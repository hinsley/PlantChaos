
begin
    sc = GLMakie.Screen()
    _u0 = SVector(.75, 0.5, 0.5, .5, 0.6, 0.0)
    tspan = (0.0, 100000.0)
    _prob = @lift ODEProblem(Plant.melibeNew, _u0, tspan, $p)
    sol = @lift solve($_prob, RK4(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0)
    _fig = Figure();
    ax = Axis3(_fig[1,1])
    x = @lift $sol[1,:]
    Ca = @lift $sol[5,:]
    v = @lift $sol[6,:]
    lines!(ax, Ca, x, v)
    display(sc,_fig)
end