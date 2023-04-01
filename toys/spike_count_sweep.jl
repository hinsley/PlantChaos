using DifferentialEquations
using DiffEqGPU
using JLD2
using LinearAlgebra
using Printf
using Roots
using StaticArrays
using Statistics

include("../model/Plant.jl")

state = Plant.default_state
u0 = state

# TODO: Use the helper functions from Plant.jl instead of redefining them here.
Vs(V) = (127.0f0*V+8265.0f0)/105.0f0
ah(V) = 0.07f0*exp((25.0f0-Vs(V))/20.0f0)
bh(V) = 1.0f0/(1.0f0+exp((55.0f0-Vs(V))/10.0f0))
hinf(V) = ah(V)/(ah(V)+bh(V))
am(V) = 0.1f0*(50.0f0-Vs(V))/(exp((50.0f0-Vs(V))/10.0f0)-1.0f0)
bm(V) = 4.0f0*exp((25.0f0-Vs(V))/18.0f0)
minf(V) = am(V)/(am(V)+bm(V))
an(V) = 0.01f0*(55.0f0-Vs(V))/(exp((55.0f0-Vs(V))/10.0f0)-1.0f0)
bn(V) = 0.125f0*exp((45.0f0-Vs(V))/80.0f0)
ninf(V) = an(V)/(an(V)+bn(V))
xinf(p, V) = 1.0f0 / (1.0f0 + exp(0.15f0 * (p[16] - V - 50.0f0)))
IKCa(p, V) = p[2]*hinf(V)*minf(V)^3.0f0*(p[8]-V) + p[3]*ninf(V)^4.0f0*(p[9]-V) + p[6]*xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)
xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end

function Ca_null_Ca(p, v)
    return p[13]*xinf(p, v)*(p[12]-v+p[17])
end

# The function which must be minimized to find the equilibrium voltage.
function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

# Finds the equilibrium in the slow subsystem.
function Ca_x_eq(p)
    v_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
    Ca_eq = Ca_null_Ca(p, v_eq)
    x_eq = xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

function evalue_dilate(re, im, dilation)
    if re == 0.0f0
        return (re, im)
    end
    return (re*(re^2)^((1-dilation)/(2*dilation)), im)
end

function countSpikes(sol, p, debug=false)
    # Obtain burst reset times.
    resets = []
    Ca = 5
    x = 1
    V = 6
    v_eq = 0.0f0
    Ca_eq = 0.0f0
    x_eq = 0.0f0
    try
        v_eq, Ca_eq, x_eq = Ca_x_eq(p)
    catch e
        if debug
            print("No equilibrium found.")
        end
        return [0]
    end

    for i in 2:length(sol)
        x = 1
        Ca = 5
        if sol[i-1][x] < x_eq && sol[i][x] < x_eq && sol[i][Ca] <= Ca_eq < sol[i-1][Ca]
            push!(resets, i)
        end
    end

    if debug
        print("$(length(resets)) burst resets observed.")
    end

    V_threshold = 0.0

    # Obtain spike counts per burst.
    spike_counts = []
    for i in 1:length(resets)-1
        spike_count = 0
        for j in resets[i]:resets[i+1]
            if sol[j-1][V] < V_threshold < sol[j][V]
                spike_count += 1
            end
        end
        if spike_count > 0
            push!(spike_counts, spike_count)
        end
    end

    if length(spike_counts) < 2
        return [0]
    else
        return spike_counts[2:end]
    end
end

function mmoSymbolics(sol, p, debug=false)
    # Obtain a mixed mode oscillation symbolic sequence.
    # 0 = sub-threshold oscillation
    # 1 = spike

    cutoff_STOs = 2 # Truncate the head of the symbolic sequence by this many STOs.

    STOs_observed = 0
    symbols = []
    Ca = 5
    x = 1
    V = 6
    v_eq = 0.0f0
    Ca_eq = 0.0f0
    x_eq = 0.0f0
    try
        v_eq, Ca_eq, x_eq = Ca_x_eq(p)
    catch e
        if debug
            print("No equilibrium found.")
        end
        return [0]
    end

    V_threshold = -40.0

    for i in 2:length(sol)
        x = 1
        Ca = 5
        if sol[i-1][V] < V_threshold < sol[i][V]
            if STOs_observed > cutoff_STOs
                push!(symbols, 1) # Spike
            end
        elseif sol[i-1][x] < x_eq && sol[i][x] < x_eq && sol[i][Ca] <= Ca_eq < sol[i-1][Ca]
            if STOs_observed > cutoff_STOs
                push!(symbols, 0) # Sub-threshold oscillation
            end
            STOs_observed += 1
        end
    end

    if debug
        print("$(length(symbols)) symbolic events captured.")
    end

    return symbols
end

function cleanup(mmo_symbolic_sequence)
    # Remove all solitary STOs.
    if length(mmo_symbolic_sequence) < 2
        return mmo_symbolic_sequence
    end
    cleaned = []
    for i in 1:length(mmo_symbolic_sequence)
        if mmo_symbolic_sequence[i] == 0
            if i == 1
                if mmo_symbolic_sequence[i+1] == 0
                    push!(cleaned, 0)
                end
            elseif i == length(mmo_symbolic_sequence)
                if mmo_symbolic_sequence[i-1] == 0
                    push!(cleaned, 0)
                end
            else
                if mmo_symbolic_sequence[i-1] == 0 || mmo_symbolic_sequence[i+1] == 0
                    push!(cleaned, 0)
                end
            end
        else
            push!(cleaned, 1)
        end
    end
    return cleaned
end

function spikeCounts(mmo_symbolic_sequence)
    # Obtain the number of spikes per burst.
    # In order to discard convergence to the equilibrium after
    # an initial transient train of spikes, we check whether a
    # sequence "spike, STO, spike" occurs (not necessarily
    # consecutively). If so, we return the number of spikes per
    # burst. Otherwise, we return NaN. This is what we use
    # transient_stage for.
    spikes = 0
    transient_stage = 0
    spikes_per_burst = []
    for i in 1:length(mmo_symbolic_sequence)
        if mmo_symbolic_sequence[i] == 1 # Spike
            if transient_stage == 1
                transient_stage += 1
            elseif transient_stage == 3
                spikes += 1
            end
        else # STO
            if transient_stage == 0 || transient_stage == 2
                transient_stage += 1
            end
            if transient_stage == 3 && spikes > 0
                push!(spikes_per_burst, spikes)
                spikes = 0
            end
        end
    end
    if length(spikes_per_burst) < 2
        return [0]
    else
        return spikes_per_burst[2:end]
    end
end

function maxSTOsPerBurst(mmo_symbolic_sequence)
    # Obtain the maximum number of STOs per burst.
    # In order to discard convergence to the equilibrium after
    # an initial transient train of spikes, we check whether a
    # sequence "spike, STO, spike" occurs (not necessarily
    # consecutively). If so, we return the maximum number of STOs
    # per burst. Otherwise, we return NaN. This is what we use
    # transient_stage for.
    STOs = 0
    transient_stage = 0
    STOs_per_burst = []
    max_spikes_per_burst = 0
    spikes_in_burst = 0
    for i in 1:length(mmo_symbolic_sequence)
        if mmo_symbolic_sequence[i] == 0 # STO
            spikes_in_burst = 0
            if transient_stage == 1
                transient_stage += 1
            elseif transient_stage == 3
                STOs += 1
            end
        else # Spike
            if transient_stage == 0 || transient_stage == 2
                transient_stage += 1
            end
            if transient_stage == 3 
                spikes_in_burst += 1
                if spikes_in_burst > max_spikes_per_burst
                    max_spikes_per_burst = spikes_in_burst
                end
                if STOs > 0
                    push!(STOs_per_burst, STOs)
                    STOs = 0
                end
            end
        end
    end

    if STOs_per_burst == [] || max_spikes_per_burst < 2
        return NaN
    end
    return maximum(STOs_per_burst)
end

function transitionMap(spike_counts)
    plt = scatter(1, markeralpha=0.2, legend=false, aspect_ratio=:equal, size=(600, 600), xticks=0:maximum(spike_counts), yticks=0:maximum(spike_counts), xlims=(-0.5, maximum(spike_counts) + 0.5), ylims=(-0.5, maximum(spike_counts) + 0.5))
    plot!(plt, [0, maximum(spike_counts)], [0, maximum(spike_counts)], linealpha=0.2)
    for i in 1:length(spike_counts)-1
        push!(plt, (spike_counts[i], spike_counts[i+1]))
    end
    return plt
end

function markovChain(spike_counts)
    if length(spike_counts) == 0
        return zeros(0, 0)
    end
    size = max(spike_counts...)
    chain = zeros(size, size)
    for i in 1:length(spike_counts)-1
        chain[spike_counts[i], spike_counts[i+1]] += 1
    end
    for row in 1:size
        # The following does yield the correct Markov chain, but is not
        # very useful for chaos scans.
        #
        # if max(chain[row, :]...) == 0.0
        #     # If we don't know what comes next, we consider all
        #     # outcomes equiprobable.
        #     chain[row, :] = ones(size)
        # end
        # Normalize rows to have total probability 1.
        # chain[row, :] = normalize!(chain[row, :], 1)

        if max(chain[row, :]...) != 0.0
            # If this row is nonzero, normalize it to have probability 1.
            chain[row, :] = normalize!(chain[row, :], 1)
        end
    end

    return chain
end

function makeParams(ΔCa, Δx)
    return @SVector Float32[
        Plant.default_params[1],  # Cₘ
        Plant.default_params[2],  # gI
        Plant.default_params[3],  # gK
        Plant.default_params[4],  # gₕ
        Plant.default_params[5],  # gL
        Plant.default_params[6],  # gT
        Plant.default_params[7],  # gKCa
        Plant.default_params[8],  # EI
        Plant.default_params[9],  # EK
        Plant.default_params[10], # Eₕ
        Plant.default_params[11], # EL
        Plant.default_params[12], # ECa
        Plant.default_params[13], # Kc
        Plant.default_params[14], # τₓ
        Plant.default_params[15], # ρ
        Δx,                       # Δx
        ΔCa                       # ΔCa
    ]
end

ΔCa_min = -42.5
ΔCa_max = -33.5
ΔCa_resolution = 1600
Δx_min = -1.6
Δx_max = 0.5
Δx_resolution = Int(ΔCa_resolution/2)
chunk_proportion = 1/8

tspan = (0.0f0, 1.0f5)

for chunk in 0:Int(1/chunk_proportion)^2-1
    println("Beginning chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    params = []
    chunk_ΔCa_min = ΔCa_min + (ΔCa_max - ΔCa_min)*chunk_proportion*trunc(Int, chunk*chunk_proportion)
    chunk_ΔCa_max = ΔCa_min + (ΔCa_max - ΔCa_min)*(chunk_proportion*(trunc(Int, chunk*chunk_proportion)+1)-1/ΔCa_resolution)
    chunk_Δx_min = Δx_min + (Δx_max - Δx_min)*chunk_proportion*(chunk%(1/chunk_proportion))
    chunk_Δx_max = Δx_min + (Δx_max - Δx_min)*(chunk_proportion*(chunk%(1/chunk_proportion)+1) - 1/Δx_resolution) # What exactly is that -1/Δx_resolution doing? It's preventing overlap.
    for ΔCa in range(chunk_ΔCa_min, chunk_ΔCa_max, length=Int(ΔCa_resolution*chunk_proportion))
        append!(params, [makeParams(ΔCa, Δx) for Δx in range(chunk_Δx_min, chunk_Δx_max, length=Int(Δx_resolution*chunk_proportion))])
        ranges = Dict(
            "ΔCa_min" => chunk_ΔCa_min,
            "ΔCa_max" => chunk_ΔCa_max,
            "Δx_min" => chunk_Δx_min,
            "Δx_max" => chunk_Δx_max
        )
        @save "toys/output/chunk_$(chunk+1)_ranges.jld2" ranges
    end

    function initial_conditions(p)
        try
            v_eq, Ca_eq, x_eq = Ca_x_eq(p)
            u0 = @SVector Float32[x_eq, state[2], state[3], state[4], Ca_eq-0.2, v_eq, state[7]]
        catch e
            # This should trigger when the Ca_x_eq function fails to converge.
            u0 = state
        end
        return u0
    end

    # Out-of-place.
    prob = ODEProblem{false}(Plant.melibeNew, u0, tspan, params[1])
    # In-place.
    # prob = ODEProblem{true}(Plant.melibeNew!, u0, tspan, params[1])
    prob_func(prob, i, repeat) = remake(prob, u0=initial_conditions(params[trunc(Int, i)]), p=params[trunc(Int, i)]) # Why are we getting Floats here?

    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    #@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUArray(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=3.0f0, saveat=range(tspan[1], tspan[2], length=1500))
    @time sol = solve(monteprob, Tsit5(), EnsembleThreads(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=1.0f0, saveat=range(tspan[1], tspan[2], length=1500), verbose=false)

    println("Post-processing chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    # TODO: Vectorize this so it doesn't take so long.
    # results = []
    # @time for i in 1:length(sol)
    #     push!(results, mmoSymbolics(sol[i], params[i]))
    # end

    println("Saving chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    # @save "toys/output/chunk_$(chunk+1).jld2" results
    @save "toys/output/chunk_$(chunk+1).jld2" sol
    println("Finished chunk $(chunk+1) of $(Int(1/chunk_proportion)^2): $(round(100*(chunk+1)*chunk_proportion^2, digits=2))%")
end

using Plots
using Plots.PlotMeasures

using FiniteDiff

########## Run a single solution and plot a multi-figure diagram.
@gif for ΔCa in range(-50.0, 175.0, length=250)
#begin
    #ΔCa = -39.5
    Δx = -1.8
    voltage_tspan = (0.0f0, 1.0f5) # The full trace.
    #voltage_tspan = (0.0f0, 2.0f4) # Comment this out to show the whole voltage trace.
    tspan = (0.0f0, 1.0f5)
    margin = 0.01f0
    params = makeParams(ΔCa, Δx)
    # Start below the equilibrium point.
    v_eq, Ca_eq, x_eq = Ca_x_eq(params)
    state = @SVector Float32[0.8, 0.0, 0.137, 0.389, 0.8, v_eq, 0.0]
    #state = @SVector Float32[x_eq, state[2], state[3], state[4], Ca_eq, v_eq, state[7]]
    prob = ODEProblem{false}(Plant.melibeNew, state, tspan, params)
    monteprob=EnsembleProblem(prob)
    sol = solve(monteprob, Tsit5(), EnsembleThreads(), trajectories=1, adaptive=false, dt=1.0f0, verbose=false)
    fig1 = plot(
        sol,
        idxs=(5, 1),
        lw=2.0,
        legend=false,
        xlims=(0.55, 1.25),
        ylims=(0.1, 0.95),
        #xlims=(min([v[5] for v in sol[1].u]...)-margin, max([v[5] for v in sol[1].u]...)+margin),
        #ylims=(min([v[1] for v in sol[1].u]...)-margin, max([v[1] for v in sol[1].u]...)+margin),
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx)
    )
    v_eq, Ca_eq, x_eq = Ca_x_eq(params)
    sol[1]
    V_range = nothing
    try
        V_range = range(xinfinv(params, min_x), xinfinv(params, max_x), length=1000)
    catch e
        V_range = range(-70, 20, length=1000)
    end
    plot!(fig1, [Ca_null_Ca(params, V) for V in V_range], [Plant.xinf(params, V) for V in V_range])
    plot!(fig1, [x_null_Ca(params, V) for V in V_range], [Plant.xinf(params, V) for V in V_range])
    scatter!(fig1, [Ca_eq], [x_eq])
    # Eigenvalue plot.
    function du(u)
        static_du = Plant.melibeNew(u, params, 0f0)
        return Float32[static_du[1], static_du[2], static_du[3], static_du[4], static_du[5], static_du[6], static_du[7]]
    end
    equilibrium = Float32[x_eq, state[2], state[3], state[4], Ca_eq, v_eq, state[7]]
    jacobian = FiniteDiff.finite_difference_jacobian(du, equilibrium)
    eigenvalues = eigvals(jacobian)
    eigenvalue_margin = 0.2 # Margin proportional to range.
    eigenvalue_dilation::Int = 2 # Dilation factor for eigenvalues - helpful for seeing stability transitions.
    evalue_real_imag_pairs = [(real(evalue), imag(evalue)) for evalue in eigenvalues]
    transformed_evalues = [evalue_dilate(evalue[1], evalue[2], eigenvalue_dilation) for evalue in evalue_real_imag_pairs]
    real_bounds = (min([evalue[1] for evalue in transformed_evalues]...), max([evalue[1] for evalue in transformed_evalues]...))
    real_range = real_bounds[2] - real_bounds[1]
    if real_range == 0
        real_bounds = (real_bounds[1] - eigenvalue_margin, real_bounds[2] + eigenvalue_margin)
    else
        real_bounds = (real_bounds[1] - eigenvalue_margin*real_range/2, real_bounds[2] + eigenvalue_margin*real_range/2)
    end
    imag_bounds = (min([evalue[2] for evalue in transformed_evalues]...), max([evalue[2] for evalue in transformed_evalues]...))
    imag_range = imag_bounds[2] - imag_bounds[1]
    if imag_range == 0
        imag_bounds = (imag_bounds[1] - eigenvalue_margin, imag_bounds[2] + eigenvalue_margin)
    else
        imag_bounds = (imag_bounds[1] - eigenvalue_margin*imag_range/2, imag_bounds[2] + eigenvalue_margin*imag_range/2)
    end
    fig2=scatter(
        [evalue[1] for evalue in transformed_evalues],
        [evalue[2] for evalue in transformed_evalues],
        legend=false,
        xlabel=@sprintf("\$\\textrm{sign}(\\textrm{Re}(\\lambda)) \\cdot \\sqrt[%d]{|\\textrm{Re}(\\lambda)|}\$", eigenvalue_dilation),
        ylabel="\$\\textrm{Im}(\\lambda)\$",
        xlims=real_bounds,
        ylims=imag_bounds,
        color=:black,
        markersize=6.0
    )
    plot!(fig2, [real_bounds...], [0, 0], color=:black, lw=1.0)
    plot!(fig2, [0, 0], [imag_bounds...], color=:black, lw=1.0)
    fig3=plot(
        sol,
        idxs=(6),
        legend=false,
        xlims=voltage_tspan,
        ylims=(-70, 30),
        xlabel="t",
        ylabel="V"
    )
    plt = plot(
        fig1,
        fig2,
        fig3,
        layout=@layout[a{0.6h}; b c],
        size=(1440, 1280),
        left_margin=1cm,
        right_margin=1cm
    )
    display(plt)
end
##########

function paramsToChunkAndIndex(ΔCa, Δx)
    # Returns the chunk number and index within that chunk for a given ΔCa and Δx parameter value.
    # Also returns the ΔCa and Δx values of the actual point within the chunk after snapping to the
    # correct discretized value.
    true_ΔCa_max = ΔCa_max - (ΔCa_max - ΔCa_min)/ΔCa_resolution
    true_Δx_max = Δx_max - (Δx_max - Δx_min)/Δx_resolution
    ΔCa_normed = (ΔCa - ΔCa_min)/(true_ΔCa_max - ΔCa_min)
    Δx_normed = (Δx - Δx_min)/(true_Δx_max - Δx_min)
    ΔCa_index = round(ΔCa_normed*(ΔCa_resolution - 1) + 1, RoundNearestTiesUp)
    Δx_index = round(Δx_normed*(Δx_resolution - 1) + 1, RoundNearestTiesUp)
    chunk_ΔCa_index = ceil(ΔCa_index/ΔCa_resolution/chunk_proportion)
    chunk_Δx_index = ceil(Δx_index/Δx_resolution/chunk_proportion)
    chunk_index = (chunk_ΔCa_index-1)/chunk_proportion + chunk_Δx_index
    interior_ΔCa_index = ΔCa_index - ΔCa_resolution * chunk_proportion * (chunk_ΔCa_index - 1)
    interior_Δx_index = Δx_index - Δx_resolution * chunk_proportion * (chunk_Δx_index - 1)
    interior_index = (interior_ΔCa_index - 1) * Δx_resolution * chunk_proportion + interior_Δx_index
    solution_ΔCa = ΔCa_min + (ΔCa_index-1)*(ΔCa_max - ΔCa_min)/ΔCa_resolution
    solution_Δx = Δx_min + (Δx_index-1)*(Δx_max - Δx_min)/Δx_resolution

    return (trunc(Int, chunk_index), trunc(Int, interior_index), solution_ΔCa, solution_Δx)
end

function plotCaX(ΔCa, Δx; lw=0.5, dpi=500, size=(1280, 720), Ca_lims=(0.6, 1.0))
    chunk, index, true_ΔCa, true_Δx = paramsToChunkAndIndex(ΔCa, Δx)
    @load "toys/output/chunk_$(chunk).jld2" sol

    plt = plot(
        sol[index],
        idxs=(5, 1),
        lw=lw,
        legend=false,
        xlims=(Ca_lims[1], Ca_lims[2]),
        ylims=(0.0, 1.0),
        dpi=dpi,
        size=size,
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx)
    )

    p = makeParams(true_ΔCa, true_Δx)

    v_eq, Ca_eq, x_eq = Ca_x_eq(p)
    V_range = nothing
    try
        V_range = range(xinfinv(p, min_x), xinfinv(p, max_x), length=1000)
    catch e
        V_range = range(-70, 20, length=1000)
    end
    
    plot!(plt, [Ca_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range])
    plot!(plt, [x_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range])
    scatter!(plt, [Ca_eq], [x_eq])

    return plt
end

function plotV(ΔCa, Δx; lw=0.5, dpi=500, size=(1280, 720), Ca_lims=(0.6, 1.0))
    chunk, index, true_ΔCa, true_Δx = paramsToChunkAndIndex(ΔCa, Δx)
    @load "toys/output/chunk_$(chunk).jld2" sol

    plt = plot(
        sol[index],
        idxs=(6),
        lw=lw,
        legend=false,
        dpi=dpi,
        size=size,
        xlabel="t",
        ylabel="mV",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx)
    )

    return plt
end

function animatedWalk(initial, final, frames; fps=10, lw=0.5, dpi=500, size=(1280, 720), Ca_lims=(0.6, 1.0), verbose=true)
    # Initial and final should each be 2-tuples of (ΔCa, Δx).
    ΔCa = range(initial[1], final[1], length=frames)
    Δx = range(initial[2], final[2], length=frames)
    anim = @animate for i in 1:frames
        if verbose
            println("Plotting frame $(i) of $(frames).")
        end
        plotCaX(ΔCa[i], Δx[i], lw=lw, dpi=dpi, size=size, Ca_lims=Ca_lims)
    end
    gif(anim, "toys/output/walk_$(initial)_$(final).gif", fps=fps)
end

plt = heatmap(
    xlabel="\$\\Delta_{Ca}\$",
    ylabel="\$\\Delta_x\$",
    #xlim=(-41, -38),
    #ylim=(-1.75, -1.1),
    title="Slow manifold revolutions per burst",
    size=(1000, 750),
    dpi=1000
)

for i in 1:Int(1/chunk_proportion)^2
    println("Plotting chunk $(i) of $(Int(1/chunk_proportion)^2).")
    @load "toys/output/chunk_$(i)_ranges.jld2" ranges
    @load "toys/output/chunk_$(i).jld2" sol

    ΔCa_range = range(ranges["ΔCa_min"], ranges["ΔCa_max"], length=Int(ΔCa_resolution*chunk_proportion))
    Δx_range = range(ranges["Δx_min"], ranges["Δx_max"], length=Int(Δx_resolution*chunk_proportion))
    
    params = []
    for ΔCa in ΔCa_range
        for Δx in Δx_range
            push!(params, makeParams(ΔCa, Δx))
        end
    end

    heatmap!(
        plt,
        ΔCa_range,
        Δx_range,
        reshape([maxSTOsPerBurst(cleanup(mmoSymbolics(sol[i], params[i]))) for i in 1:length(sol)], Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion)),
        color=cgrad(:amp, scale=:exp)
    );
end

display(plt)

chunk, index, true_ΔCa, true_Δx = paramsToChunkAndIndex(-40.99, -1.57796)
@load "toys/output/chunk_$(chunk).jld2" sol
maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx))))
plotCaX(true_ΔCa, true_Δx)
println(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx))))

plt2 = plot(
    sol[index],
    idxs=(6),
    lw=0.5,
    legend=false,
    title="Voltage trace",
    size=(1000, 750)
)
display(plt2)

for Ca in range(-40.0, -39.0, length=10)
    # Get max STOs per burst.
    chunk, index, true_ΔCa, true_Δx = paramsToChunkAndIndex(Ca, -1.2)
    @load "toys/output/chunk_$(chunk).jld2" sol
    if maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx)))) > 1
        println("Ca = $(Ca): $(maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx)))))")
    end
end
