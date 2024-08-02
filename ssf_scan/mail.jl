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
space, u0s = make_space(_space, xs)

function compute(space, u0s, resolution)

    function prob_func(prob, i, repeat)
        remake(prob, p = space[i], u0 = u0s[i])
    end

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

distances, counts = compute(space, u0s, resolution)
heatmap(counts)
ds2 = map(distances) do x
    x > .00001 ? .00001 : x
end

resolution2 = 1000
thsp2 = range(0.0, 2pi, length = resolution2)
csp2 = range(-40, -30, length = resolution2)
_space2 = Iterators.product(thsp2, csp2)
space2, u0s2 = make_space(_space2, xs)


distances2, counts2 = compute(space2, u0s2, resolution2)
ds22 = map(distances2) do x
    x > .05 ? .05 : x
end

hom = -36.0233
begin
    try close(ssf_screen) 
    catch
        nothing
    end
    global ssf_screen = GLMakie.Screen(;resize_to = (1000, 1000))
    set_theme!()
    ssf_fig = Figure(size = (1000, 1200))
    display(ssf_screen, ssf_fig)
    ssf_ax = Axis(ssf_fig[2,2], xlabel = "ΔCa", ylabel = "θ",  yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    pl = heatmap!(ssf_ax, csp, thsp, counts', colormap = Makie.Categorical(:lighttest))
    Colorbar(ssf_fig[2,3],pl, label = "spikes")
    ax2 = Axis3(ssf_fig[2,1], xlabel = "ΔCa", ylabel = "θ", zlabel = "dist * 1e5", aspect = (1,1,1),
     yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))

    surface!(ax2, csp,thsp,1e5 .* ds2', color = log.(counts'), colormap = :lighttest)
    ax3 = Axis(ssf_fig[3,1], xlabel = "ΔCa", ylabel = "dist * 1e5")
    mins = reshape(reduce(min, 1e5 .* ds2, dims = 1), resolution)
    lines!(ax3, csp, mins, color = :black, linewidth = 2)
    lines!(ax2, [Point3f(csp[i], 0, mins[i]) for i in 1:resolution], color = :black, linewidth = 2)
    ax4 = Axis(ssf_fig[3,2], xlabel = "θ", ylabel = "dist * 1e5", xticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    spks = Int[]
    dsts = Float64[]
    for i in 1:resolution
        count = 0
        dst = 1e-5
        for j in 1:resolution
            if counts[i,j] > count
                if distances[i,j] < .00001
                    count = counts[i,j]
                    dst = distances[i,j]
                end
            end
        end
        push!(spks, count)
        push!(dsts, dst)
    end
    lines!(ax4, thsp, dsts .* 1e5, color = spks, colormap = :lighttest)
    #hom
    lines!(ssf_ax, [hom, hom], [0, 2pi], color = :red, linewidth = 2)
    lines!(ax2, [hom, hom], [0, 2pi], [0, 0], color = :red, linewidth = 2)
    scatter!(ax3, [hom],[0], color = :red, markersize = 10)

    #zoom out
    ssf_ax2 = Axis(ssf_fig[1,2], xlabel = "ΔCa", ylabel = "θ",  yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))
    pl = heatmap!(ssf_ax2, csp2, thsp2, counts2', colormap = Makie.Categorical(:lighttest))
    Colorbar(ssf_fig[1,3],pl, label = "spikes")
    ax22 = Axis3(ssf_fig[1,1], xlabel = "ΔCa", ylabel = "θ", zlabel = "dist", aspect = (1,1,1), yticks = ([0, pi, 2pi], ["0",  "π", "2π"]))

    surface!(ax22, csp2,thsp2, ds22', color = log.(counts2'), colormap = :lighttest)
    mins = reshape(reduce(min, ds22, dims = 1), resolution2)
    lines!(ax22, [Point3f(csp2[i], 0, mins[i]) for i in 1:resolution2], color = :black, linewidth = 2)
    # hom
    lines!(ssf_ax2, [hom, hom], [0, 2pi], color = :red, linewidth = 2)
    lines!(ax22, [hom, hom], [0, 2pi], [0, 0], color = :red, linewidth = 2)

    #panel letters
    #text!(ssf_fig[1,1], "A", position = (-0.9, .8), fontsize = 20, space = :clip)
    Label(ssf_fig[1,1, Left()], "A", padding = (0,1,0,0), valign = :top, fontsize = 25)
    Label(ssf_fig[1,2, Left()], "B", padding = (0,30,0,0), valign = :top, fontsize = 25)
    Label(ssf_fig[2,1, Left()], "C", padding = (0,1,0,0), valign = :top, fontsize = 25)
    Label(ssf_fig[2,2, Left()], "D", padding = (0,30,0,0), valign = :top, fontsize = 25)
    Label(ssf_fig[3,1, Left()], "E", padding = (0,30,0,0), valign = :top, fontsize = 25)
    Label(ssf_fig[3,2, Left()], "F", padding = (0,30,0,0), valign = :top, fontsize = 25)
    # resize
    resize!(ssf_fig, 1000, 1200)
    display(ssf_screen, ssf_fig)
end

save("ssf_scan.png", ssf_fig)