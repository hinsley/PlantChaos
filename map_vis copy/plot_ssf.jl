function hom_f(u,p,t)
    p = p.p
    du = melibeNew(u,p,t)
    return du
end

function hom_condition(u,t,integrator)
    radius = .01
    eq = integrator.p.eq
    return t > 200 ? norm(u .- eq) - radius : 1.0
end

hom_cb = ContinuousCallback(hom_condition, affect!, affect_neg! = nothing)

_ps = [[0.6248946785926819,-54.0794792175293]]
p[] = vcat(p[][1:15], _ps[1])

begin
    try close(ssf_screen) 
    catch
        nothing
    end
    global ssf_screen = GLMakie.Screen(;resize_to = (1500, 800))
    set_theme!()
    ssf_fig = Figure()
    # saddle case above saddle-saddle focus point
    #p[] = vcat(p[][1:15], _ps[1])
    phaseax = Axis3(ssf_fig[1:4,1:2], xlabel = "Ca", ylabel = "x", zlabel = "V", 
        azimuth = 1.6π, elevation = .1π)
    tax = Axis(ssf_fig[5,1], xlabel = "t", ylabel = "V")
    eigax = Axis(ssf_fig[5,2], xlabel = "Re", ylabel = "Im")
    thetaslider = Slider(ssf_fig[6,1:2], range = 4.71:0.0000001:4.7136, startvalue = 0)
    #thetaslider = Slider(ssf_fig[6,1:2], range = 4.711157657:0.000000001:4.711157688, startvalue = 0) # for burst of 17 to 16
    Label(ssf_fig[7,1:2], "θ on unstable manifold around saddle", tellwidth = false)
    display(ssf_screen, ssf_fig)

    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p[], v),
        Plant.xinfinv(p[], 0.99e0), Plant.xinfinv(p[], 0.01e0))
    v_eq = v_eqs[2]
    Ca_eq = Equilibria.Ca_null_Ca(p[], v_eq)
    x_eq = Plant.xinf(p[], v_eq)
    saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> melibeNew(u,p[],0), saddle)
    vals,vecs = eigen(jac)
    θ = thetaslider.value
    ssf_u0 = @lift SVector{6}(
        .001*(sin($θ) * vecs[:,5]+ cos($θ) * vecs[:,6]) + saddle
    )
    ssf_p = (p = p[], eq = SVector{6}(saddle))
    _ssf_prob = ODEProblem(hom_f, SVector{6}(zeros(6)), (0.0, 5e4), ssf_p)
    ssf_prob = @lift remake(_ssf_prob, u0 = $ssf_u0)
    ssf_sol = @lift solve($ssf_prob, RK4(), abstol = 1e-10, reltol = 1e-10, saveat = .2)

    pltsol = @lift [Point3f(x[5], x[1], x[6]) for x in $ssf_sol.u]
    colors = @lift map($ssf_sol.u) do e
        norm(e .- saddle)
    end
    lines!(phaseax, pltsol, color = :blue)
    pltt = @lift [Point2f($ssf_sol.t[i], $ssf_sol.u[i][6]) for i in eachindex($ssf_sol)]
    lines!(tax, pltt, color = :blue)

    plteigs = [(real(e), imag(e)) for e in vals if real(e) != 0.0][1:5]
    scatter!(eigax, plteigs, markersize = 15, color = :black, marker = 'o')
    vlines!(eigax, 0, color = :black, linewidth = 2)
    hlines!(eigax, 0, color = :black, linewidth = 2)

end

record(ssf_fig, "emergingburst17.mp4", thetaslider.range[]; framerate = 30) do theta
    θ[] = theta
end