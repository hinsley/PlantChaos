using Pkg
Pkg.instantiate()

using BifurcationKit
using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Plots
using Printf
using StaticArrays

include("../../../model/Plant.jl")
include("../../../tools/equilibria.jl")
include("../../../tools/solve.jl")

function cont()
    # Shilnikov-Hopf bifurcation
    p = Vector{Float64}([[Plant.default_params[i] for i in 1:15]..., -1.4, -38.0])

    # Equilibrium
    u0 = Vector{Float64}(Equilibria.eq(p))
    
    f(x, p) = [entry for entry in Plant.melibeNew(x, p, 0.0)] # We don't want an SVector here.

    # record a few components / indicators about x 
    #record(x, p) = (x = x)

    prob = BifurcationProblem(f, u0, p, (@lens _[16]))#, record_from_solution=record)

    opts_br = ContinuationPar(
        p_min = -5.0,
        p_max = -1.0,
        ds = 0.002,
        dsmax = 0.05,
        nev = 4,
        n_inversion = 6,
        max_bisection_steps = 25,
        max_steps = 200,
        detect_bifurcation = 3,
        newton_options = NewtonPar(tol = 1e-9, verbose=true)
    )

    br = @time continuation(
        prob,
        PALC(),
        opts_br;
        normC = norminf,
        bothside = true,
        #verbosity = 3
    )

    scene = plot(br, plotfold=false, markersize=4, legend=:topleft)
    savefig(scene, "SH_branch.png")
    
    return br
end
result = cont()

function cont_hopf(br, ind_hopf)
    p = Vector{Float64}([[Plant.default_params[i] for i in 1:15]..., br.specialpoint[ind_hopf].param, -38.0])

    opts_br = ContinuationPar(
        p_min = -50.0,
        p_max = 100.0,
        ds = 0.002,
        dsmax = 0.05,
        nev = 4,
        n_inversion = 6,
        max_bisection_steps = 25,
        max_steps = 1000,
        detect_bifurcation = 3,
        newton_options = NewtonPar(tol = 1e-9, verbose=true)
    )

    hopf_br = @time continuation(
        br,
        ind_hopf,
        (@lens _[17]),
        opts_br;
        bothside = true,
        normC = norminf,
        #update_minaug_every_step = 1,
        detect_codim2_bifurcation = 2,
        start_with_eigen = true,
        #verbosity = 3
    )

    scene = plot(hopf_br, plotfold=false, markersize=4, legend=:topleft)
    savefig(scene, "SH_hopf.png")
    
    return hopf_br
end
hopf_result = cont_hopf(result, 4)

# Continue the branch of periodic orbits from the Hopf bifurcation.
function cont_po(br, ind_hopf)
    # Hopf bifurcation
    p = Vector{Float64}([[Plant.default_params[i] for i in 1:15]..., br.specialpoint[ind_hopf].param, -38.0])

    # Equilibrium
    u0 = Vector{Float64}(Equilibria.eq(p))

    # Continuation parameters
    opts_po_cont = ContinuationPar(
        p_min = br.specialpoint[ind_hopf].param-0.3,
        p_max = br.specialpoint[ind_hopf].param,
        ds = -0.002,
        dsmax = 0.1,
        nev = 4,
        n_inversion = 6,
        max_bisection_steps = 25,
        max_steps = 1000,
        detect_bifurcation = 3,
        newton_options = NewtonPar(tol = 1e-9, verbose=true)
    )

    # Newton parameters
    optn_po = NewtonPar(tol = 1e-8,  max_iterations = 12)

    # Arguments for periodic orbits
    # One function to record information and one
    # function for plotting
    args_po = (	record_from_solution = (x, p) -> begin
            xtt = get_periodic_orbit(p.prob, x, p.p)
            return (u0 = x,
                    xtt = xtt,
                    max = maximum(xtt[1,:]),
                    min = minimum(xtt[1,:]),
                    period = getperiod(p.prob, x, p.p))
        end,
        plot_solution = (x, p; k...) -> begin
            xtt = get_periodic_orbit(p.prob, x, p.p)
            arg = (marker = :d, markersize = 1)
            plot!(xtt.t, xtt[1,:]; label = "x", arg..., k...)
            plot!(xtt.t, xtt[2,:]; label = "y", arg..., k...)
            plot!(xtt.t, xtt[3,:]; label = "n", arg..., k...)
            plot!(xtt.t, xtt[4,:]; label = "h", arg..., k...)
            plot!(xtt.t, xtt[5,:]; label = "Ca", arg..., k...)
            plot!(xtt.t, xtt[6,:]; label = "V", arg..., k...)
            plot!(br; subplot = 1, putspecialptlegend = false)
            end,
        # We use the supremum norm
        normC = norminf)

    Mt = 250 # Number of time sections

    # Create the ODE problem.
    probsh = ODEProblem(Plant.melibeNew!, u0, (0.0, 1e5), p; abstol = 1e-12, reltol = 1e-10)

    # Create the periodic orbit problem.
    po_prob = ShootingProblem(50, probsh, Rodas5(), parallel = true)

    # Continue the branch of periodic orbits from the Hopf point.
    br_po = continuation(br, ind_hopf, opts_po_cont, po_prob; args_po..., callback_newton = BifurcationKit.cbMaxNorm(10))

    # Plot the periodic orbit solution.
    scene = plot(br, br_po, markersize=3)
    plot!(scene, br_po.param, br_po.min, label="min")
    plot!(scene, br_po.param, br_po.max, label="max")

    # Save scene plot.
    #savefig(scene, "SH_po.png")

    return br_po
end
po_result = cont_po(result, 4)

# Show special points in phase space.
for special_point in result.specialpoint
    if special_point.type == :endpoint
        continue
    end
    # Run a solution.
    p = SVector{17, Float32}([[Plant.default_params[i] for i in 1:15]..., special_point.param, -38.0f0])
    u0 = SVector{6, Float32}(special_point.x)
    tspan = (0.0f0, 1.0f6)
    prob = ODEProblem(Plant.melibeNew, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-4, abstol=1e-4, saveat=0.1f0)
    # Plot the solution.
    plt = plot(sol, vars=(5, 1, 6), label="", color=:black, lw=2, legend=false)
    # Plot the special point.
    scatter!([special_point.x[1]], [special_point.x[6]], label="", color=:black, ms=4)
    
    # Save the plt.
    savefig(plt, "SH_$(special_point.type)_$(special_point.param).png")
end

# Show periodic orbit in phase space.
function show_po(po_result, ind_param, x=nothing)
    # Run a solution.
    #p = SVector{17, Float64}([[Plant.default_params[i] for i in 1:15]..., po_result.param[ind_param], -38.0f0])
    #if isnothing(x)
    #    u0 = SVector{6, Float64}(po_result[ind_param].u0[1:6])
    #else
    #    u0 = SVector{6, Float64}(x)
    #end
    #tspan = (0.0f0, 1.0f5)
    #prob = ODEProblem(Plant.melibeNew, u0, tspan, p)
    #sol = solve(prob, Tsit5(), reltol=1e-4, abstol=1e-4, saveat=0.1f0)
    #Cas = [u[5] for u in sol.u]
    #xs = [u[1] for u in sol.u]
    #Vs = [u[6] for u in sol.u]

    #####
    # New
    Cas = po_result[ind_param].xtt[5,:]
    xs = po_result[ind_param].xtt[1,:]
    Vs = po_result[ind_param].xtt[6,:]
    #
    #####

    # Plot the solution.
    plt = plot(
        #sol,
        Cas,
        xs,
        Vs,
        vars=(5, 1, 6),
        title=@sprintf("\$\\Delta x\$ = %.5f", po_result[ind_param].param),
        label="",
        color=:black,
        lw=2,
        xlabel="Ca",
        ylabel="x",
        zlabel="V",
        legend=false,
        xlims=(0.85, 0.95),
        ylims=(0.5, 1.0),
        zlims=(-50.0, -40.0)
        #xlims=(minimum(Cas), maximum(Cas)),
        #ylims=(minimum(xs), maximum(xs)),
        #zlims=(minimum(Vs), maximum(Vs))
    )

    # Plot the special point.
    #scatter!([po_result[ind_param].u0[5]], [po_result[ind_param].u0[1]], [po_result[ind_param].u0[6]], label="", color=:black, ms=4)
    
    # Save the plt.
    # Be careful if doing a lot of these, e.g. in a @gif loop! 
    #savefig(plt, "SH_po_$(po_result[ind_param].param).png")
end
@gif for i in 1:length(po_result)
    show_po(po_result, i)
end fps=30
@gif for specialpoint in po_result.specialpoint
    if specialpoint.type != :endpoint
        show_po(po_result, specialpoint.idx, specialpoint.x[1:6])
    end
end fps=5
