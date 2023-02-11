using DifferentialEquations
using DiffEqGPU
using JLD2
using LinearAlgebra
using Roots
using StaticArrays

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

    V_threshold = 0.0

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

function maxSTOsPerBurst(mmo_symbolic_sequence)
    # Obtain the maximum number of STOs per burst.
    max_STOs = 0
    STOs = 0
    for i in 1:length(mmo_symbolic_sequence)
        if mmo_symbolic_sequence[i] == 0
            STOs += 1
            if STOs > max_STOs
                max_STOs = STOs
            end
        else
            STOs = 0
        end
    end
    return max_STOs
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

ΔCa_min = -46.0
ΔCa_max = -5.0
ΔCa_resolution = 1000
Δx_min = -3.0
Δx_max = 0.0
Δx_resolution = Int(ΔCa_resolution/2)
chunk_proportion = 1/5

tspan = (0.0f0, 1.0f5)

for chunk in 0:Int(1/chunk_proportion)^2-1
    println("Beginning chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    params = []
    chunk_ΔCa_min = ΔCa_min + (ΔCa_max - ΔCa_min)*chunk_proportion*trunc(Int, chunk*chunk_proportion)
    chunk_ΔCa_max = ΔCa_min + (ΔCa_max - ΔCa_min)*(chunk_proportion*(trunc(Int, chunk*chunk_proportion)+1)-1/ΔCa_resolution)
    for ΔCa in range(chunk_ΔCa_min, chunk_ΔCa_max, length=Int(ΔCa_resolution*chunk_proportion))
        chunk_Δx_min = Δx_min + (Δx_max - Δx_min)*chunk_proportion*(chunk%(1/chunk_proportion))
        chunk_Δx_max = Δx_min + (Δx_max - Δx_min)*(chunk_proportion*(chunk%(1/chunk_proportion)+1) - 1/Δx_resolution) # What exactly is that -1/Δx_resolution doing? It's preventing overlap.
        for Δx in range(chunk_Δx_min, chunk_Δx_max, length=Int(Δx_resolution*chunk_proportion))
            p = @SVector Float32[
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
                Δx,                          # Δx
                ΔCa                          # ΔCa
            ]
            push!(params, p)
        end
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
    # prob = ODEProblem{false}(Plant.melibeNew!, u0, tspan, params[1])
    prob_func(prob, i, repeat) = remake(prob, u0=initial_conditions(params[trunc(Int, i)]), p=params[trunc(Int, i)]) # Why are we getting Floats here?

    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    #@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUArray(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=3.0f0, saveat=range(tspan[1], tspan[2], length=1500))
    @time sol = solve(monteprob, Tsit5(), EnsembleThreads(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=1.0f0, saveat=range(tspan[1], tspan[2], length=1500), verbose=false)

    println("Post-processing chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    # TODO: Vectorize this so it doesn't take so long.
    results = []
    @time for i in 1:length(sol)
        push!(results, mmoSymbolics(sol[i], params[i]))
    end

    println("Saving chunk $(chunk+1) of $(Int(1/chunk_proportion)^2).")
    @save "toys/output/chunk_$(chunk+1).jld2" results
    println("Finished chunk $(chunk+1) of $(Int(1/chunk_proportion)^2): $(round(100*(chunk+1)*chunk_proportion^2, digits=2))%")
end

using Plots

plt = heatmap(
    xlabel="\$\\Delta_{Ca}\$",
    ylabel="\$\\Delta_x\$",
    xlim=(-46, -5),
    ylim=(-3, 0),
    title="Slow manifold revolutions per burst",
    color=:thermal,
    size=(1000, 750),
    dpi=1000
)

for i in 1:Int(1/chunk_proportion)^2
    @load "toys/output/chunk_$(i)_ranges.jld2" ranges
    @load "toys/output/chunk_$(i).jld2" results

    heatmap!(
        plt,
        range(ranges["ΔCa_min"], ranges["ΔCa_max"], length=Int(ΔCa_resolution*chunk_proportion)),
        range(ranges["Δx_min"], ranges["Δx_max"], length=Int(Δx_resolution*chunk_proportion)),
        reshape([maxSTOsPerBurst(sequence) for sequence in results], Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion)),
        color=:thermal,
        size=(1000, 750),
        dpi=1000
    );
end

display(plt)
