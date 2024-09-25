using Pkg
Pkg.activate("./../symbolic_scan")
Pkg.instantiate()

# Imports.
using Colors
using FiniteDiff
using JLD2
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf
using Roots
using StaticArrays
using Statistics

begin # Setup.
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
    function Ca_x_eq(p; which_root=Nothing)
        v_eqs = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))
        if which_root == Nothing
            v_eq = length(v_eqs) > 1 ? v_eqs[2] : v_eqs[1]
        else
            v_eq = v_eqs[which_root]
        end
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

    function spikeAmplitudeVariance(sol, p; debug=false)
        # Obtain the variance of the spike amplitudes.

        V = 6 # Index in state variables.

        V_threshold = -40.0

        spike_amplitudes = []
        above_threshold = false
        V_max_in_spike = -Inf
        for i in 1:length(sol)
            if sol[i][V] >= V_threshold
                if !above_threshold
                    above_threshold = true
                    V_max_in_spike = sol[i][V]
                else
                    V_max_in_spike = max(V_max_in_spike, sol[i][V])
                end
            elseif above_threshold
                above_threshold = false
                push!(spike_amplitudes, V_max_in_spike)
                # There's no need to reset V_max_in_spike to -Inf.
            end
        end

        return isempty(spike_amplitudes) ? NaN : var(spike_amplitudes)
    end

    function ISIs(u, t, p; debug=false)
        # Get a vector of inter-spike intervals.

        V = 6 # Index in state variables.
        V_threshold = -40.0

        # Obtain the solution indices for when we initially exceed the threshold.
        spike_indices = []
        above_threshold = false
        for i in 1:length(u)
            if u[i][V] >= V_threshold
                if !above_threshold
                    above_threshold = true
                    push!(spike_indices, i)
                end
            elseif above_threshold
                above_threshold = false
            end
        end

        # Obtain the inter-spike intervals in terms of solution times.
        ISIs = []
        for i in 1:length(spike_indices)-1
            push!(ISIs, t[spike_indices[i+1]] - t[spike_indices[i]])
        end

        return ISIs
    end

    function interSpikeIntervalVariance(u, t, p)
        # Obtain the variance of the inter-spike intervals.
        intervals = ISIs(u, t, p)
        return isempty(intervals) ? NaN : var(intervals)
    end

    function minimumDistanceToEquilibrium(u, p)
        # Obtain the minimum distance to the equilibrium.
        # This is the minimum distance to the equilibrium
        # of the state variables (not including the
        # synaptic variables).
        try
            v_eq, Ca_eq, x_eq = Ca_x_eq(p)
            minimum_distance = Inf
            for i in 1:length(u)
                dCa = u[i][5] - Ca_eq
                dx = u[i][1] - x_eq
                distance = sqrt(dCa^2 + dx^2)
                if distance < minimum_distance
                    minimum_distance = distance
                end
            end
            return minimum_distance
        catch BoundsError
            return NaN
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

    function blockEntropy(str,m)
        # Compute the block entropy of a string.
        # str: string
        # m: block length
        blocks = [str[i:i+m-1] for i in 1:length(str)-m-1]
        # Block occurrence probability.
        psm = [count(==(b),blocks) for b in unique(blocks)]./length(blocks)
        # Return block entropy with a finite-size correction (source: Jack's paper).
        return -sum([p*log(p) for p in psm]) - (length(unique(blocks)) - 1)/(2*length(str))
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
end

begin # Create the plot.
    plt = heatmap(
        axis=nothing,
        grid=nothing,
        legend=false,
        ticks=nothing,
        title="",
        # Margins for frameless plot
        #left_margin=-1.6mm,
        #right_margin=-1.6mm,
        #top_margin=-2mm,
        #bottom_margin=-2mm,
        # Margins for plot with frame
        left_margin=40mm,
        bottom_margin=32mm,
        guidefontsize=48,
        xlabel="\$\\Delta_{Ca}\$",
        ylabel="\$\\Delta_x\$",
        #xlabel="",
        #ylabel="",
        size=(3840, 2160),
        # Full poster sample
        xlim=(-130, 100),
        ylim=(-12, 15)
    )
end

begin # Plot the ISI variance.
    ΔCa_min = -130.0
    ΔCa_max = 100.0
    ΔCa_resolution = 2000
    Δx_min = -12.0
    Δx_max = 15.0
    Δx_resolution = Int(ΔCa_resolution/2)
    chunk_proportion = 1/8

    tspan = (0.0f0, 1.0f5)

    scan_directory = "toys/output/Full poster sample"

    for i in 1:Int(1/chunk_proportion)^2
        println("Plotting chunk $(i) of $(Int(1/chunk_proportion)^2).")
        @load "$(scan_directory)/chunk_$(i)_ranges.jld2" ranges
        
        ΔCa_range = range(ranges["ΔCa_min"], ranges["ΔCa_max"], length=Int(ΔCa_resolution*chunk_proportion))
        Δx_range = range(ranges["Δx_min"], ranges["Δx_max"], length=Int(Δx_resolution*chunk_proportion))
        
        params = []
        for ΔCa in ΔCa_range
            for Δx in Δx_range
                push!(params, makeParams(ΔCa, Δx))
            end
        end
    
        # If the simulation didn't run long enough, use the previous trajectory. DUCT TAPE.
        #for i in 1:length(sol.u)
        #    if sol.u[i].t[end] < tspan[2]*0.8
        #        sol.u[i] = sol.u[i-1]
        #    end
        #end
    
        # Truncate first 20% of simulations.
        #percent_to_skip = 0.2
        #first_timestep = Int(round(percent_to_skip*length(sol.u[i].u)))+1
    
        # Measure: Inter-spike interval variance.
        @load "$(scan_directory)/chunk_$(i)_metricVector.jld2" metricVector
        heatmap!(
            plt,
            ΔCa_range,
            Δx_range,
            reshape(metricVector, Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion)),
            c=cgrad([
                Colors.RGB(6/255, 70/255, 53/255),
                Colors.RGB(6/255, 70/255, 53/255),
                Colors.RGB(135/255, 158/255, 131/255),
                Colors.RGB(204/255, 222/255, 209/255),
                Colors.RGB(0/255, 50/255, 0/255),
                Colors.RGB(150/255, 160/255, 160/255)
            ], [0.45, 0.45, 0.73, 0.68, 0.8, 0.7, 0.91])
        );
    end

    display(plt)
    savefig("ISI variance sweep.png")
end

begin # Plot the max-STO block entropy.
    begin # Full poster sample.
        ΔCa_min = -130.0
        ΔCa_max = 100.0
        ΔCa_resolution = 2000
        Δx_min = -12.0
        Δx_max = 15.0
        Δx_resolution = Int(ΔCa_resolution/2)
        chunk_proportion = 1/8

        tspan = (0.0f0, 1.0f5)

        scan_directory = "toys/output/Full poster sample"

        for i in 1:Int(1/chunk_proportion)^2
            println("Plotting chunk $(i) of $(Int(1/chunk_proportion)^2).")
            @load "$(scan_directory)/chunk_$(i)_ranges.jld2" ranges
            #@load "$(scan_directory)/chunk_$(i).jld2" sol
            
            ΔCa_range = range(ranges["ΔCa_min"], ranges["ΔCa_max"], length=Int(ΔCa_resolution*chunk_proportion))
            Δx_range = range(ranges["Δx_min"], ranges["Δx_max"], length=Int(Δx_resolution*chunk_proportion))
            
            params = []
            for ΔCa in ΔCa_range
                for Δx in Δx_range
                    push!(params, makeParams(ΔCa, Δx))
                end
            end
        
            # If the simulation didn't run long enough, use the previous trajectory. DUCT TAPE.
            #for i in 1:length(sol.u)
            #    if sol.u[i].t[end] < tspan[2]*0.8
            #        sol.u[i] = sol.u[i-1]
            #    end
            #end
        
            # Truncate first 20% of simulations.
            #percent_to_skip = 0.2
            #first_timestep = Int(round(percent_to_skip*length(sol.u[i].u)))+1
        
            # Measure: Block entropy of STOs per burst.
            # We don't truncate here since it's already being done by the measure computation functions.
            @load "$(scan_directory)/chunk_$(i)_metricVector.jld2" metricVector
            ISImetricVector = metricVector
            @load "$(scan_directory)/chunk_$(i)_STO_block_entropy_metricVector.jld2" metricVector
            if any(x -> !isinf(x) && !isnan(x), metricVector)
                # Only draw block entropy pixel if has well-defined ISI variance too.
                for j in 1:length(metricVector)
                    if isinf(ISImetricVector[j]) || isnan(ISImetricVector[j])
                        metricVector[j] = NaN
                    end
                end
                scalar = 12 # 10 is okay too, but a little more lackluster by comparison. Maybe necessary if doing higher resolution.
                rescaled_metricVector = scalar*metricVector
                heatmap!(
                    plt,
                    ΔCa_range,
                    Δx_range,
                    reshape(rescaled_metricVector, Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion)),
                    c=:thermal,
                    #clims=(0.0, 0.3)
                );
            end
        end
        
        display(plt)
        savefig("Max STO block entropy.png")
    end
    
    begin # Subcritical AH ridge.
        ΔCa_min = -42.5
        ΔCa_max = -33.5
        ΔCa_resolution = 1600
        Δx_min = -1.6
        Δx_max = 0.5
        Δx_resolution = Int(ΔCa_resolution/2)
        chunk_proportion = 1/8

        tspan = (0.0f0, 1.0f5)

        scan_directory = "toys/output/Subcritical AH ridge"

        for i in 1:Int(1/chunk_proportion)^2
            println("Plotting chunk $(i) of $(Int(1/chunk_proportion)^2).")
            @load "$(scan_directory)/chunk_$(i)_ranges.jld2" ranges
            #@load "$(scan_directory)/chunk_$(i).jld2" sol
            
            ΔCa_range = range(ranges["ΔCa_min"], ranges["ΔCa_max"], length=Int(ΔCa_resolution*chunk_proportion))
            Δx_range = range(ranges["Δx_min"], ranges["Δx_max"], length=Int(Δx_resolution*chunk_proportion))
            
            params = []
            for ΔCa in ΔCa_range
                for Δx in Δx_range
                    push!(params, makeParams(ΔCa, Δx))
                end
            end
        
            # If the simulation didn't run long enough, use the previous trajectory. DUCT TAPE.
            #for i in 1:length(sol.u)
            #    if sol.u[i].t[end] < tspan[2]*0.8
            #        sol.u[i] = sol.u[i-1]
            #    end
            #end
        
            # Truncate first 20% of simulations.
            #percent_to_skip = 0.2
            #first_timestep = Int(round(percent_to_skip*length(sol.u[i].u)))+1
        
            # Measure: Block entropy of STOs per burst.
            # We don't truncate here since it's already being done by the measure computation functions.
            @load "$(scan_directory)/chunk_$(i)_metricVector.jld2" metricVector
            ISImetricVector = metricVector
            @load "$(scan_directory)/chunk_$(i)_STO_block_entropy_metricVector.jld2" metricVector
            if any(x -> !isinf(x) && !isnan(x), metricVector)
                # Only draw block entropy pixel if has well-defined ISI variance too.
                for j in 1:length(metricVector)
                    if isinf(ISImetricVector[j]) || isnan(ISImetricVector[j])
                        metricVector[j] = NaN
                    end
                end
                scalar = 12 # 10 is okay too, but a little more lackluster by comparison. Maybe necessary if doing higher resolution.
                rescaled_metricVector = scalar*metricVector
                heatmap!(
                    plt,
                    ΔCa_range,
                    Δx_range,
                    reshape(rescaled_metricVector, Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion)),
                    c=:thermal,
                    #clims=(0.0, 0.3)
                );
            end
        end
        
        display(plt)
        savefig("Max STO block entropy enhanced.png")
    end
end
