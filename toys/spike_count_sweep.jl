using DifferentialEquations
using DiffEqGPU
using JLD2
using LinearAlgebra
using Roots
using StaticArrays

include("../model/GPUPlant.jl")

state = GPUPlant.default_state
u0 = state

# The function which must be minimized to find the equilibrium voltage.
function Ca_difference(p, v)
    Vs(V) = (127.0f1*V+8265.0f1)/105.0f1
    ah(V) = 0.07f1*exp((25.0f0-Vs(V))/20.0f1)
    bh(V) = 1.0f1/(1.0f1+exp((55.0f1-Vs(V))/10.0f1))
    hinf(V) = ah(V)/(ah(V)+bh(V))
    am(V) = 0.1f1*(50.0f1-Vs(V))/(exp((50.0f1-Vs(V))/10.0f1)-1.0f1)
    bm(V) = 4.0f1*exp((25.0f1-Vs(V))/18.0f1)
    minf(V) = am(V)/(am(V)+bm(V))
    an(V) = 0.1f1*(55.0f1-Vs(V))/(exp((55.0f1-Vs(V))/10.0f1)-1.0f1)
    bn(V) = 0.125f1*exp((45.0f1-Vs(V))/80.0f1)
    ninf(V) = an(V)/(an(V)+bn(V))
    IKCa = p[2]*hinf(v)*minf(v)^3*(p[8]-v) + p[3]*ninf(v)^4*(p[9]-v) + p[6]*xinf(p, v)*(p[8]-v) + p[4]*(p[10]-v)/((1.0f1+exp(10.0f1*(v-50.0f1)))*(1.0f1+exp(-(v-63.0f1)/7.8f1))^3) + p[5]*(p[11]-v)
    x_null_Ca = 0.5f1*IKCa/(p[7]*(v-p[9]) - IKCa)
    Ca_null_Ca = p[13]*xinf(p, v)*(p[12]-v+p[17])
    return x_null_Ca - Ca_null_Ca
end

# Finds the equilibrium in the slow subsystem.
function Ca_x_eq(p, min_V=(p[11]+p[9])/2)
    # Not very DRY of me.
    xinf(p, V) = 1.0f0 / (1.0f0 + exp(0.15f0 * (p[16] - V - 50.0f0)))
    v_eq = find_zero(v -> Ca_difference(p, v), min_V)
    Ca_eq = p[13]*xinf(p, v_eq)*(p[13]-v_eq+p[17])
    x_eq = xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

function countSpikes(sol, p, debug=false)
    # Obtain burst reset times.
    resets = []
    Ca = 5
    x = 1
    v_eq = 0.0f1
    Ca_eq = 0.0f1
    x_eq = 0.0f1
    try
        v_eq, Ca_eq, x_eq = Ca_x_eq(p, min(sol(sol.t, idxs=(6))...))
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
            if sol[j-1][6] < V_threshold < sol[j][6]
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
        if max(chain[row, :]...) == 0.0
            # If we don't know what comes next, we consider all
            # outcomes equiprobable.
            chain[row, :] = ones(size)
        end
        # Normalize rows to have total probability 1.
        chain[row, :] = normalize!(chain[row, :], 1)
    end

    return chain
end

ΔCa_min = -60.0
ΔCa_max = 35.0
ΔCa_resolution = 4000
Δx_min = -2.5
Δx_max = 1.0
Δx_resolution = Int(ΔCa_resolution/2)
chunk_proportion = 1/50

tspan = (0, 1.0f5)

for chunk in 1:Int(1/chunk_proportion)^2
    params = []
    chunk_ΔCa_min = ΔCa_min + (ΔCa_max - ΔCa_min)*chunk_proportion*trunc(Int, chunk*chunk_proportion)
    chunk_ΔCa_max = ΔCa_min + (ΔCa_max - ΔCa_min)*(chunk_proportion*(trunc(Int, chunk*chunk_proportion)+1)-1/ΔCa_resolution)
    for ΔCa in range(chunk_ΔCa_min, chunk_ΔCa_max, length=Int(ΔCa_resolution*chunk_proportion))
        chunk_Δx_min = Δx_min + (Δx_max - Δx_min)*chunk_proportion*(chunk%(1/chunk_proportion)-1)
        chunk_Δx_max = Δx_min + (Δx_max - Δx_min)*(chunk_proportion*(chunk%(1/chunk_proportion)) - 1/Δx_resolution)
        for Δx in range(chunk_Δx_min, chunk_Δx_max, length=Int(Δx_resolution*chunk_proportion))
            p = @SVector Float32[
                GPUPlant.default_params[1],  # Cₘ
                GPUPlant.default_params[2],  # gI
                GPUPlant.default_params[3],  # gK
                GPUPlant.default_params[4],  # gₕ
                GPUPlant.default_params[5],  # gL
                GPUPlant.default_params[6],  # gT
                GPUPlant.default_params[7],  # gKCa
                GPUPlant.default_params[8],  # EI
                GPUPlant.default_params[9],  # EK
                GPUPlant.default_params[10], # Eₕ
                GPUPlant.default_params[11], # EL
                GPUPlant.default_params[12], # ECa
                GPUPlant.default_params[13], # Kc
                GPUPlant.default_params[14], # τₓ
                GPUPlant.default_params[15], # ρ
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
        @save "toys/output/chunk_$(chunk)_ranges.jld2" ranges
    end
    function initial_conditions(p)
        try
            v_eq, Ca_eq, x_eq = Ca_x_eq(p)
            u0 = @SVector Float32[x_eq, state[2], state[3], state[4], Ca_eq-0.2, v_eq, state[7]]
        catch e
            u0 = state
        end
        return u0
    end
    prob = ODEProblem{false}(GPUPlant.melibeNew, u0, tspan, params[1])
    prob_func(prob, i, repeat) = remake(prob, u0=initial_conditions(params[trunc(Int, i)]), p=params[trunc(Int, i)]) # Why are we getting Floats here?

    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=1f-1, saveat=range(tspan[1], tspan[2], length=1500));

    # TODO: Vectorize this so it doesn't take so long.
    results = []
    for i in 1:length(sol)
        spike_counts = countSpikes(sol[i], params[i])
        if length(spike_counts) < 2
            push!(results, 0.0f1)
        else
            push!(results, Float32{norm(markovChain(spike_counts))})
        end
    end

    @save "toys/output/chunk_$(chunk).jld2" results
    println("Finished chunk $chunk of $(Int(1/chunk_proportion)^2): $(round(100*chunk*chunk_proportion^2, digits=2))%")
end
