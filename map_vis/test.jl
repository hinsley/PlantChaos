cais = findall(isnan, cass[])
xis = findall(isnan, xss[])
_cas = vcat([cass[][1]],[cass[][i+1] for i in cais[1:end-1]])
_xs = vcat([xss[][1]],[xss[][i+1] for i in xis[1:end-1]])

radius = mapslider.sliders[4].value

mapics = generate_ics_circle!(ics_probs, p[], eq[], radius[], map_resolution)
[e[5] for e in a] |> lines
prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
monteprob = remake(monteprob, prob_func = prob_func,
    output_func = output_func, p = (p = p[], eq = eq[], radius = radius[], menu_i = map_switch_menu.i_selected[]))

mapsol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=map_resolution,
    callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)

xmap = map(mapsol) do e
    Ca = e[1,end]
    x = e[2,end]
    arg = (Ca-eq[][5])/radius[]
    a = abs(arg) > 1 ? NaN : asin(arg)
    if x > eq[][1]
        a
    else
        pi-a
    end
end

mapsol.u[1].t

caends = [cass[][i-1] for i in cais[1:end]]
xends = [xss[][i-1] for i in xis[1:end]]

distances = [sqrt(a[1]]