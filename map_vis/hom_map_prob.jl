mutable struct Pars
    p::Vector{Float64}
    eq::SVector{6,Float64}
    radius::Float64
    v_saddle::Float64
    spiking::Bool
end

function spike_condition(u, t, integrator)
    -mapf(u, integrator.p, 0.0)[6]
end

function spike_affect!(integrator)
    v = integrator.u[6]
    if integrator.p.spiking
        dist = sqrt(abs2(integrator.u[1]-integrator.p.eq[1]) + abs2(integrator.u[5]-integrator.p.eq[5]))
        if (dist > integrator.p.radius) && (v < integrator.p.v_saddle)
            terminate!(integrator)
        end
    else
        if v > integrator.p.v_saddle
            integrator.p.spiking = true
        end
    end
end
spikecb = ContinuousCallback(spike_condition, spike_affect!, affect_neg! = nothing)

function c_condition(u,t,integrator)
    if (t < 50)
        1
    else
        sqrt(abs2(u[1]-integrator.p.eq[1]) + abs2(u[5]-integrator.p.eq[5])) - integrator.p.radius
    end
end
affect!(integrator) = terminate!(integrator) # Stop the solver
c_cb = ContinuousCallback(c_condition, affect!, affect_neg! = nothing)

cbset = CallbackSet(spikecb, c_cb)

c_ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
c_map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
c_monteprob = EnsembleProblem(c_map_prob, safetycopy=false)

function calculate_circle_map(monteprob, ics_probs, p, slider1, slider2, radius; resolution = 100)
    eq = SVector{6}(Equilibria.eq(p))

    preimage = range(slider1, slider2, length = resolution)
    mapics = generate_ics_circle!(ics_probs, p, eq, preimage, radius, resolution)

    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_saddle = v_eqs[3]

    ps = Pars(p, eq, radius, v_saddle, false)
    prob = monteprob.prob
    ts = range(0, prob.tspan[2], length = 1000)
    mapsol = Array{OrdinaryDiffEq.RecursiveArrayTools.DiffEqArray{Float64, 2, Vector{Vector{Float64}},
        StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},
        OrdinaryDiffEq.SciMLBase.SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing},
         typeof(OrdinaryDiffEq.SciMLBase.DEFAULT_OBSERVED),Pars}}(undef, resolution)
    Threads.@threads for i = 1:resolution
        prob = remake(prob, p = ps, u0 = mapics[i])
        sol = solve(prob, RK4(), abstol = 1e-8, reltol = 1e-8, saveat = 3e0)
        mapsol[i] = sol(ts, idxs = [5,1,6])
    end
    
    cmap = map(mapsol) do e
        Ca = e[1,end]
        x = e[2,end]
        arg = (Ca-eq[5])/radius
        a = abs(arg) > 1 ? NaN : asin(arg)
        if x > eq[1]
            a
        else
            pi-a
        end
    end

    cass, xss, vss = glue_trajs(mapsol)
    return (preimage, cmap, cass, xss, vss)
end 