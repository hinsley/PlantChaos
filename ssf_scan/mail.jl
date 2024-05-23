using Pkg
Pkg.activate("./ssf_scan")
using GLMakie, OrdinaryDiffEq, Roots, StaticArrays, LinearAlgebra, ForwardDiff

include("../model/Plant.jl")
function melibe5(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        Plant.dx(p, u[1], u[5]),
        Plant.dn(u[2], u[5]),
        Plant.dh(u[3], u[5]),
        Plant.dCa(p, u[4], u[1], u[5]),
        Plant.dV(p, u[1], 0.0, u[2], u[3], u[4], u[5], 0.0),
    ]
end
include("../tools/equilibria.jl")

mutable struct Params
    p::Vector{Float64}
    count::Int
    v_eq::Float64
    sf::Vector{Float64}
    check1::Bool
    check2::Bool
end

function make_space(space, xs)
    space = collect(space)
    sz = size(space)
    data = Array{Params}(undef, sz...)
    u0s = Array{SVector{6,Float64}}(undef, size(space)...)
    Threads.@threads for i in 1:sz[1]
        for j in 1:sz[2]
            p = vcat(Plant.default_params[1:15], xs, space[i,j][2])
            v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
            local u0
            v_upper = v_eqs[3]
            v_sf = v_eqs[2]
            Ca_eq = Equilibria.Ca_null_Ca(p, v_sf)
            x_eq = Plant.xinf(p, v_sf)
            eq = [x_eq, Plant.default_state[2], Plant.ninf(v_sf), Plant.hinf(v_sf), Ca_eq, v_sf]
            #jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
            #vals,vecs = eigen(jac)
            #eps = .01
            #k = argmax(real.(vecs)[6,:])
            #e1 = real.(vecs)[:,1]
            #e2 = real.(vecs)[:,2]
            #theta = space[i,j][1]
            #vec = eps*(cos(theta) * e1 + sin(theta) * e2)
            eps = .002
            e1 = SVector{6}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            e2 = SVector{6}(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            vec = eps*(cos(space[i,j][1]) * e1 + sin(space[i,j][1]) * e2)

            u0 = SVector{6}(eq - eps * vec)
            u0s[i,j] = u0
            data[i,j] = Params(p, 0, v_upper, u0, false, false)
        end
    end
    (data, u0s)
end

function f(u,p,t)
    p = p.p
    Plant.melibeNew(u,p,t)
end

function spike_condition(u, t, integrator)
   -f(u, integrator.p, 0.0)[6]
end
function spike_affect!(integrator)
    v = integrator.u[6]
    if v < integrator.p.v_eq
        if integrator.p.check1
            if integrator.p.check2
                terminate!(integrator)
            end
        else 
            integrator.p.check1 = true
        end
    else
        integrator.p.count += 1
        integrator.p.check2 = true
    end
end
cb = ContinuousCallback(spike_condition, spike_affect!, nothing)

function prob_func(prob, i, repeat)
    remake(prob, p = space[i], u0 = u0s[i])
end
function output_func(sol, i)
    prb = sol.prob
    count = prb.p.count
    last_pt = (sol.u[end][1], sol.u[end][5])
    eq = (prb.p.sf[1], prb.p.sf[5])
    dist = sqrt(sum((last_pt .- eq).^2))
    return ((count, dist), false)
end


## Closeup
resolution = 2000
thsp = range(0.0, 2pi, length = resolution)
csp = range(-36.025, -36.02, length = resolution)
xs = -1.1
_space = Iterators.product(thsp, csp)
global space, u0s = make_space(_space, xs)

function compute()
    for e in space
        e.check1 = false
        e.check2 = false
        e.count = 0
    end
    prob = ODEProblem(f, u0s[1], (0.0, 300000.0), space[1])
    scanprob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func, safetycopy = false)
    sol = solve(scanprob, BS3(), EnsembleThreads(), trajectories = resolution^2,
    callback = cb, save_everystep = false, reltol = 1e-9, abstol = 1e-9)

    distances = reshape([e[2] for e in sol.u], resolution, resolution)
    counts = reshape([e[1] for e in sol.u], resolution, resolution)
    
    return (distances, counts)
end

distances, counts = compute()
heatmap(counts)
ds2 = map(distances) do x
    x > .00001 ? .00001 : x
end
begin
    try close(ssf_screen) 
    catch
        nothing
    end
    global ssf_screen = GLMakie.Screen(;resize_to = (1000, 1000))
    set_theme!()
    ssf_fig = Figure(size = (1000, 800))
    display(ssf_screen, ssf_fig)
    ssf_ax = Axis(ssf_fig[1,2], xlabel = "ΔCa", ylabel = "θ",  yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    pl = heatmap!(ssf_ax, csp, thsp, counts', colormap = Makie.Categorical(:lighttest))
    Colorbar(ssf_fig[1,3],pl, label = "spikes")
    ax2 = Axis3(ssf_fig[1,1], xlabel = "ΔCa", ylabel = "θ", zlabel = "dist * 10e5", aspect = (1,1,1), yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))

    surface!(ax2, csp,thsp,10^5 .* ds2', color = log.(counts'), colormap = :lighttest)
    ax3 = Axis(ssf_fig[2,1], xlabel = "ΔCa", ylabel = "dist * 10e5")
    mins = reshape(reduce(min, 10^5 .* ds2, dims = 1), resolution)
    lines!(ax3, csp, mins, color = :black, linewidth = 2)
    lines!(ax2, [Point3f(csp[i], 0, mins[i]) for i in 1:resolution], color = :black, linewidth = 2)
    ax4 = Axis(ssf_fig[2,2], xlabel = "θ", ylabel = "spikes", yreversed = true, xticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    spks = Int[]
    for i in 1:resolution
        count = 0
        for j in 1:resolution
            if counts[i,j] > count
                if distances[i,j] < .00001
                    count = counts[i,j]
                end
            end
        end
        push!(spks, count)
    end
    spks
    lines!(ax4, thsp, spks, color = spks, colormap = :lighttest)
end
#save("ssf_scan_closeup.png", ssf_fig)


# zoom out
resolution = 500
thsp = range(0.0, 2pi, length = resolution)
csp = range(-40, -30, length = resolution)
xs = -1.1
_space = Iterators.product(thsp, csp)
global space, u0s = make_space(_space, xs)


distances, counts = compute()
ds2 = map(distances) do x
    x > .05 ? .05 : x
end
begin
    try close(ssf_screen) 
    catch
        nothing
    end
    global ssf_screen = GLMakie.Screen(;resize_to = (1000, 1000))
    set_theme!()
    ssf_fig = Figure(size = (1000, 450))
    display(ssf_screen, ssf_fig)
    ssf_ax = Axis(ssf_fig[1,2], xlabel = "ΔCa", ylabel = "θ",  yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    pl = heatmap!(ssf_ax, csp, thsp, counts', colormap = Makie.Categorical(:lighttest))
    Colorbar(ssf_fig[1,3],pl, label = "spikes")
    ax2 = Axis3(ssf_fig[1,1], xlabel = "ΔCa", ylabel = "θ", zlabel = "dist", aspect = (1,1,1), yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))

    surface!(ax2, csp,thsp, ds2', color = log.(counts'), colormap = :lighttest)
    mins = reshape(reduce(min, ds2, dims = 1), resolution)
    lines!(ax2, [Point3f(csp[i], 0, mins[i]) for i in 1:resolution], color = :black, linewidth = 2)
end

save("ssf_scan_zoomout.png", ssf_fig)