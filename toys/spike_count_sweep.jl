using DifferentialEquations # Don't run this if accessing parameter sweeps.
using DiffEqGPU # Don't run this if accessing parameter sweeps.
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

ΔCa_min = -130.0
ΔCa_max = 100.0
ΔCa_resolution = 2000
Δx_min = -12.0
Δx_max = 15.0
Δx_resolution = Int(ΔCa_resolution/2)
chunk_proportion = 1/8

tspan = (0.0f0, 1.0f5)

scan_directory = "toys/output/Full poster sample"

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
        @save "$(scan_directory)/chunk_$(chunk+1)_ranges.jld2" ranges
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
    # @save "$(scan_directory)/chunk_$(chunk+1).jld2" results
    @save "$(scan_directory)/chunk_$(chunk+1).jld2" sol
    println("Finished chunk $(chunk+1) of $(Int(1/chunk_proportion)^2): $(round(100*(chunk+1)*chunk_proportion^2, digits=2))%")
end

using Plots
using Plots.PlotMeasures
using Colors

using FiniteDiff

# Precompute x dune mesh.
Ca_min = -1.0
Ca_max = 4.0
Ca_resolution = 1000
x_resolution = 1000
Δx_min = -15
Δx_max = 30
Δx_resolution = 10000
V_min = -49 # For visualizations of the dune.
V_min = -70
V_max = -30 # For visualizations of the dune.
V_max = -20
V_resolution = 1000
azimuth = -60
elevation = 30

V_range = range(V_min, V_max, length=V_resolution)
Δxs = range(Δx_min, Δx_max, length=Δx_resolution)
x_nullclines = []
plt = plot(
    legend=false,
    xlabel="\$Ca\$",
    ylabel="\$x\$",
    zlabel="\$V\$",
    xlims=(Ca_min, Ca_max),
    zlims=(V_min, V_max),
    camera=(azimuth, elevation)
)
for Δx in Δxs
    x_nullcline = [] # Tuples (Ca, x, V)
    params = makeParams(0.0, Δx)
    for V in V_range
        Ca = x_null_Ca(params, V)
        x = xinf(params, V)
        push!(x_nullcline, (Ca, x, V))
    end
    push!(x_nullclines, x_nullcline)
    plot!(plt, [point[1] for point in x_nullcline], [point[2] for point in x_nullcline], [point[3] for point in x_nullcline], lw=3.0)
end
# Draw a point in the $(x, V)$ plane projected onto the x dune.
function x_dune_Ca(x, V)
    V_prior = findlast(v -> v <= V, V_range)
    V_posterior = V_prior + 1
    function left_of_x_nullcline(Δx)
        params = makeParams(0.0, Δx)
        x_prior = xinf(params, V_range[V_prior])
        x_posterior = xinf(params, V_range[V_posterior])
        # Linearly interpolate.
        x_interpolated = x_prior + (x_posterior - x_prior)*(V - V_range[V_prior])/(V_range[V_posterior] - V_range[V_prior])
        return x <= x_interpolated
    end
    # Do a linear interpolation.
    Δx_prior = findlast(left_of_x_nullcline, Δxs)
    Δx_posterior = Δx_prior + 1
    params_prior = makeParams(0.0, Δxs[Δx_prior])
    params_posterior = makeParams(0.0, Δxs[Δx_posterior])
    x_prior = xinf(params_prior, V_range[V_prior])
    x_posterior = xinf(params_posterior, V_range[V_posterior])
    t = (x - x_prior)/(x_posterior - x_prior)
    Δx = Δxs[Δx_prior] + t*(Δxs[Δx_posterior] - Δxs[Δx_prior])
    params = Tuple(i == 16 ? Δx : Plant.default_params[i] for i in 1:length(Plant.default_params))
    Ca = x_null_Ca(params, V)
    return Ca
end
display(plt)

ΔCa = 150.0
Δx = -3.0
tspan = (0, 700)
u0 = Float64[Plant.default_state...]
p = makeParams(ΔCa, Δx)
V_eq, Ca_eq, x_eq = Ca_x_eq(p)
u0[1] = x_eq-0.05
u0[6] = V_eq
u0[5] = x_dune_Ca(u0[1], u0[6])
prob = ODEProblem(Plant.melibeNew!, u0, tspan, p)
function g(resid,u,p,t)
    resid[1] = u[5] - x_dune_Ca(u[1], u[6])
    resid[2:end] .= 0
end
cb = ManifoldProjection(g)
@time sol = solve(prob,Tsit5(),save_everystep=false,callback=cb,abstol=1e-6,reltol=1e-6)
#@gif for azimuth in range(0, 360, length=100)
plt = plot(
    sol,
    idxs=(5,1,6),
    legend=false,
    title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
    xlabel="\$Ca\$",
    ylabel="\$x\$",
    zlabel="\$V\$",
    camera=(30, 30)#azimuth, elevation)
)
# Plot in slow subsystem (Ca, x).
slowplt = plot(
    legend=false,
    title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
    xlabel="\$Ca\$",
    ylabel="\$x\$"
)
V_range = range(-70, 20, length=1000)
plot!(slowplt, [Ca_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range], lw=3.0)
plot!(slowplt, [x_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range], lw=3.0)
plot!(
    slowplt,
    sol,
    idxs=(5, 1),
    xlims=(0, 1.6),
    ylims=(0, 1)
)
scatter!(
    slowplt,
    [u0[5]],
    [u0[1]]
)
# Voltage plot vs time.
vplt = plot(
    [point[6] for point in sol.u],
    legend=false,
    title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
    xlabel="\$t\$",
    ylabel="\$V\$"
)
#end
Ca0 = 0.488
params = makeParams(-30, -0.91)
V0 = find_zero((V) -> Ca_null_Ca(params, V) - Ca0, -40)
########## Run a single solution and plot a multi-figure diagram.
#@gif for ΔCa in range(-50.0, 175.0, length=500)
begin
    # Arbitrary point
    ΔCa = 97.6410942 # Comment this out if making a gif.
    Δx = -1.4
    # Bogdanov-Takens
    #ΔCa = -10.2359
    #Δx = -10.8111392512278
    # Lower Bautin Point (GH)
    #ΔCa = 38.098
    #Δx = -2.70199569136383
    # Upper Bautin Point (GH)
    #ΔCa = -45.1575230179832
    #Δx = 11.944
    voltage_tspan = (0.0f1, 1.0f6) # The full trace.
    #voltage_tspan = (0.0f0, 2.0f4) # Comment this out to show the whole voltage trace.
    tspan = (0.0f1, 1.0f6)
    margin = 0.1f0
    titlefontsize=24
    guidefontsize=16
    tickfontsize=12
    which_root = 1 # Which root to use for the equilibrium point.

    params = makeParams(ΔCa, Δx)
    # Start below the equilibrium point.
    v_eq, Ca_eq, x_eq = Ca_x_eq(params, which_root=2)
    state = @SVector Float32[0.25, 0.0, 0.137, 0.389, 0.67, v_eq, 0.0] # Use this most of the time.
    #state = @SVector Float32[0.9, 0.0, 0.137, 0.389, 1.4, v_eq, 0.0] # For showing bistability in BT.
    #state = @SVector Float32[x_eq, state[2], state[3], state[4], Ca_eq, v_eq, state[7]]
    
    prob = ODEProblem(Plant.melibeNew, state, tspan, params)
    monteprob = EnsembleProblem(prob)

    sol = solve(monteprob, Tsit5(), EnsembleThreads(), trajectories=1, adaptive=false, dt=1.0f0, verbose=false)
    sol[1] = sol[1][1:end] # Remove transient.
    fig1 = plot(
        sol,
        idxs=(5, 1),
        lw=2.0,
        legend=false,
        xlims=(0.55, 1.25),
        ylims=(0.05, 0.97),
        #xlims=(min([v[5] for v in sol[1].u]...)-margin, max([v[5] for v in sol[1].u]...)+margin),
        #ylims=(min([v[1] for v in sol[1].u]...)-margin, max([v[1] for v in sol[1].u]...)+margin),
        xlabel="\$Ca\$",
        ylabel="\$x\$",
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        titlefontsize=titlefontsize
    )
    V_range = nothing
    try
        V_range = range(xinfinv(params, min_x), xinfinv(params, max_x), length=1000)
    catch e
        V_range = range(-70, 20, length=1000)
    end
    plot!(fig1, [Ca_null_Ca(params, V) for V in V_range], [Plant.xinf(params, V) for V in V_range], lw=3.0)
    plot!(fig1, [x_null_Ca(params, V) for V in V_range], [Plant.xinf(params, V) for V in V_range], lw=3.0)
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
        markersize=6.0,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize
    )
    plot!(fig2, [real_bounds...], [0, 0], color=:black, lw=1.0)
    plot!(fig2, [0, 0], [imag_bounds...], color=:black, lw=1.0)
    fig3=plot(
        sol,
        idxs=(6),
        legend=false,
        xlims=voltage_tspan,
        #xlims=(0, 10000),
        #ylims=(-70, 30),
        ylims=(-65, 30),
        xlabel="\$t\$",
        ylabel="\$V\$",
        xticks=nothing,
        lw=3.0,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        bottom_margin=4mm
    )
    plt = plot(
        fig1,
        #fig2,
        fig3,
        #layout=@layout[a{0.6h}; b c],
        layout=@layout[a{0.75h}; b],
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
    @load "$(scan_directory)/chunk_$(chunk).jld2" sol

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
    @load "$(scan_directory)/chunk_$(chunk).jld2" sol

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
    gif(anim, "$(scan_directory)/walk_$(initial)_$(final).gif", fps=fps)
end

##########
# Produce a scan.
# With labels
#plt = heatmap(
#    xlabel="\$\\Delta_{Ca}\$",
#    ylabel="\$\\Delta_x\$",
#    xlim=(-50, 100),
#    ylim=(-5, 1),
#    title="Max STO per burst - block entropy",
#    size=(1000, 750),
#    dpi=1000,
#    margin=2mm
#)
# Without any labels
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
    # Full parameter plane
    xlim=(-130, 100),
    ylim=(-12, 15)
    # Zoomed in
    #xlim=(-100, 100),
    #ylim=(-5, 5)
    # Horizontal AH strip
    #xlim=(-50, 100),
    #ylim=(-5, 1)
)

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

    # Measure: Max number of STOs per burst.
    # We don't truncate here since it's already being done by the measure computation functions.
    #heatmap!(
    #    plt,
    #    ΔCa_range,
    #    Δx_range,
    #    reshape([maxSTOsPerBurst(cleanup(mmoSymbolics(sol.u[i].u, params[i]))) for i in 1:length(sol.u)], Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion))
    #);

    # Measure: Block entropy of STOs per burst.
    # We don't truncate here since it's already being done by the measure computation functions.
    #block_size = 3
    #metricVector = [blockEntropy(cleanup(mmoSymbolics(sol.u[i].u, params[i])), block_size) for i in 1:length(sol.u)]
    #metricVector = [value == 0 ? NaN : value for value in metricVector]
    #heatmap!(
    #    plt,
    #    ΔCa_range,
    #    Δx_range,
    #    reshape(metricVector, Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion))
    #);

    # Measure: Spike voltage amplitude variance.
    #heatmap!(
    #    plt,
    #    ΔCa_range,
    #    Δx_range,
    #    reshape([spikeAmplitudeVariance(sol.u[i].u[first_timestep:end], params[i]) for i in 1:length(sol.u)], Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion))
    #) ;

    # Measure: Inter-spike interval variance.
    #metricVector = [log(1+interSpikeIntervalVariance(sol.u[i].u[first_timestep:end], sol.u[i].t[first_timestep:end], params[i])) for i in 1:length(sol.u)]
    #@save "$(scan_directory)/chunk_$(i)_metricVector.jld2" metricVector
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

    # Measure: Minimum distance in the slow projection to the equilibrium point.
    #heatmap!(
    #    plt,
    #    ΔCa_range,
    #    Δx_range,
    #    reshape([log(1+1/minimumDistanceToEquilibrium(sol.u[i].u[first_timestep:end], params[i])) for i in 1:length(sol.u)], Int(Δx_resolution*chunk_proportion), Int(ΔCa_resolution*chunk_proportion))
    #);
end

include("./bif_diagram.jl")
plot_bif_diagram!(plt)

display(plt)
##########

##########
# Return maps.
begin
    # Arbitrary point.
    ΔCa = -38.087
    Δx = -1.073
    # Lower Bautin Point (GH)
    #ΔCa = 38.098
    #Δx = -2.70199569136383
    # Upper Bautin Point (GH)
    #ΔCa = -45.1575230179832
    #Δx = 11.944
    map_resolution = 10000
    fill_ins = 0
    fill_in_resolution = 10
    V_threshold = -40 # Spike threshold.
    x_offset = 1f-2 # Offset from xinf to avoid numerical issues.
    V_margin = 1f-1 # Margin around V_eq to avoid numerical issues.
    min_V = -57 # Minimum V value to sample.
    max_V = -45 # Maximum V value to sample.
    # Plotting font sizes
    titlefontsize=24
    guidefontsize=20
    tickfontsize=16
    tspan=(0.0f0, 1.0f5)
    p = makeParams(ΔCa, Δx)
    V_eq, Ca_eq, x_eq = Ca_x_eq(p)
    V_range = range(-70, -15, length=100)

    ##########
    # Ca maximum return map.

    # Generate initial conditions along the Ca nullcline.
    V0 = collect(range(V_eq+V_margin, max_V, length=map_resolution))
    # Discontinuity nbhd only.
    #V0 = collect(range(-38.43, -36, length=map_resolution))
    Ca0 = [Ca_null_Ca(p, V) for V in V0]
    x0 = [xinf(p, V)-x_offset for V in V0]
    u0 = [@SVector Float32[x0[i], state[2], state[3], state[4], Ca0[i], V0[i], state[7]] for i in 1:length(V0)]
    Ca_null = [Ca_null_Ca(p, V) for V in V_range]

    # Solve.
    prob = ODEProblem{false}(Plant.melibeNew, u0[1], tspan, p)
    prob_func(prob, i, repeat) = remake(prob, u0=u0[i])
    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    function condition(u, t, integrator)
        p = integrator.p
        # TODO: Correct this comment.
        # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
        if u[6] > -20 || u[5] < Ca_eq return 1f0 end
        (t < 50) ? 1f0 : -p[15] * (p[13] * u[1] * (p[12] - u[6] + p[17]) - u[5])
    end
    affect!(integrator) = terminate!(integrator) # Stop the solver
    cb = ContinuousCallback(condition, affect!, affect_neg! = nothing) # Define the callback
    #@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUArray(), trajectories=trunc(Int, ΔCa_resolution*Δx_resolution*chunk_proportion^2), adaptive=false, dt=3.0f0, saveat=range(tspan[1], tspan[2], length=1500))
    @time sol = solve(monteprob, Tsit5(), EnsembleThreads(), callback=cb, trajectories=map_resolution, adaptive=false, dt=1.0f0, saveat=range(tspan[1], tspan[end], length=2000), verbose=false)
    Ca_initial = [sol[i][1][5] for i in 1:length(sol)]
    Ca_final = [sol[i][end][5] for i in 1:length(sol)]

    # Compute spike counts.
    spike_counts = []
    for i in 1:length(sol)
        spikes = 0
        for j in 1:length(sol[i])-1
            if sol[i][j][6] < V_threshold < sol[i][j+1][6]
                spikes += 1
            end
        end
        push!(spike_counts, spikes)
    end

    # Plot the phase portrait.
    plt = plot(
        xlabel="\$Ca\$",
        ylabel="\$x\$",
        xlims=(0.7, 1.3),
        ylims=(0.1, 0.95),
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        size=(1000, 750),
        c=spike_counts,
        legend=false,
        margin=5mm,
        titlefontsize=titlefontsize,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize
    )

    # Nullclines.
    plot!(plt, [Ca_null_Ca(p, V) for V in V_range], [xinf(p, V) for V in V_range], label="Ca nullcline")
    plot!(plt, [x_null_Ca(p, V) for V in V_range], [xinf(p, V) for V in V_range], label="x nullcline")
    # Equilibrium point.
    scatter!(plt, [Ca_eq], [x_eq], label="Equilibrium point", color=:red, ms=5)
    # Trajectories.
    for spike_count in reverse(unique(spike_counts))
        for i in 1:length(sol)
            if spike_counts[i] == spike_count
                Cas = [sol[i][j][5] for j in 1:length(sol[i].u)]
                xs = [sol[i][j][1] for j in 1:length(sol[i].u)]
                plot!(plt, Cas, xs, label=@sprintf("\$%d\$", spike_count), c=spike_count, ms=2, alpha=0.3)
            end
        end
    end

    display(plt)

    # Plot the return map.
    plt = plot(
        xlabel="\$Ca_n\$",
        ylabel="\$Ca_{n+1}\$",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        size=(1000, 1000),
        legend=false,
        margin=5mm,
        titlefontsize=titlefontsize,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        aspect_ratio=:equal,
        #xlim=(0, 1),
        #ylim=(0, 1)
    )

    # Fixed point line.
    plot!(plt, [Ca0[1], Ca0[end]], [Ca0[1], Ca0[end]], label="Fixed point line", color=:red)
    # Return map points.
    scatter!(plt, Ca_initial, Ca_final, label="Return map", c=spike_counts, markerstrokecolor=spike_counts, ms=2)

    display(plt)
    ##########

    ##########
    # Ca minimum return map.
    
    # Generate initial conditions along the Ca nullcline.
    V0 = collect(range(min_V, V_eq-V_margin, length=map_resolution))
    Ca0 = [Ca_null_Ca(p, V) for V in V0]
    x0 = [xinf(p, V)+x_offset for V in V0]
    u0 = [@SVector Float32[x0[i], state[2], state[3], state[4], Ca0[i], V0[i], state[7]] for i in 1:length(V0)]
    Ca_null = [Ca_null_Ca(p, V) for V in V_range]

    # Solve.
    prob = ODEProblem{false}(Plant.melibeNew, u0[1], tspan, p)
    prob_func(prob, i, repeat) = remake(prob, u0=u0[i])
    monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
    function condition(u, t, integrator)
        p = integrator.p
        # TODO: Correct this comment.
        # Return the distance between u and the Ca nullcline in x if to the left of the equilibrium.
        if u[1] >= x_eq || u[5] >= Ca_eq || u[6] >= V_eq return 1f0 end
        (t < 100) ? 1f0 : p[15] * (p[13] * u[1] * (p[12] - u[6] + p[17]) - u[5])
    end
    affect!(integrator) = terminate!(integrator) # Stop the solver
    cb = ContinuousCallback(condition, affect!, affect_neg! = nothing) # Define the callback
    @time sol = solve(monteprob, Tsit5(), EnsembleThreads(), callback=cb, trajectories=map_resolution, adaptive=false, dt=1.0f0, saveat=range(tspan[1], tspan[end], length=2000), verbose=false)
    Ca_initial = [sol[i][1][5] for i in 1:length(sol)]
    Ca_final = [sol[i][end][5] for i in 1:length(sol)]

    # Compute spike counts.
    spike_counts = []
    for i in 1:length(sol)
        spikes = 0
        for j in 1:length(sol[i])-1
            if sol[i][j][6] < V_threshold < sol[i][j+1][6]
                spikes += 1
            end
        end
        push!(spike_counts, spikes)
    end

    # Plot the phase portrait.
    plt = plot(
        xlabel="\$Ca\$",
        ylabel="\$x\$",
        xlims=(0.35, 0.95),
        ylims=(0.1, 0.95),
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        size=(1000, 750),
        c=spike_counts,
        legend=false,
        margin=5mm,
        titlefontsize=titlefontsize,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize
    )

    # Nullclines.
    plot!(plt, [Ca_null_Ca(p, V) for V in V_range], [xinf(p, V) for V in V_range], label="Ca nullcline")
    plot!(plt, [x_null_Ca(p, V) for V in V_range], [xinf(p, V) for V in V_range], label="x nullcline")
    # Equilibrium point.
    scatter!(plt, [Ca_eq], [x_eq], label="Equilibrium point", color=:red, ms=5)
    # Trajectories.
    for spike_count in unique(spike_counts)
        for i in 1:length(sol)
            if spike_counts[i] == spike_count
                Cas = [sol[i][j][5] for j in 1:length(sol[i].u)]
                xs = [sol[i][j][1] for j in 1:length(sol[i].u)]
                plot!(plt, Cas, xs, label=@sprintf("\$%d\$", spike_count), c=spike_count, ms=2, alpha=0.3)
            end
        end
    end

    display(plt)

    # Plot the return map.
    plt = plot(
        xlabel="\$Ca_n\$",
        ylabel="\$Ca_{n+1}\$",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        size=(1000, 1000),
        legend=false,
        margin=5mm,
        titlefontsize=titlefontsize,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        aspect_ratio=:equal,
        #xlim=(0.83, 0.832),
        #ylim=(0.487, 0.49)
    )

    # Fixed point line.
    plot!(plt, [Ca0[1], Ca0[end]], [Ca0[1], Ca0[end]], label="Fixed point line", color=:red)
    # Return map points.
    scatter!(plt, Ca_initial, Ca_final, label="Return map", c=spike_counts, markerstrokecolor=spike_counts, ms=2)

    display(plt)
    ##########
end
##########
plt = plot(
        xlabel="\$Ca_n\$",
        ylabel="\$Ca_{n+1}\$",
        title=@sprintf("\$\\Delta_{Ca} = %.3f, \\Delta_x = %.3f\$", ΔCa, Δx),
        size=(1000, 1000),
        legend=false,
        margin=5mm,
        titlefontsize=titlefontsize,
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        aspect_ratio=:equal,
        xlim=(0.72, 0.735),
        ylim=(0.72, 0.735)
)
plot!(plt, [Ca0[1], Ca0[end]], [Ca0[1], Ca0[end]], label="Fixed point line", color=:red)
# Return map points.
scatter!(plt, Ca_initial, Ca_final, label="Return map", c=spike_counts, markerstrokecolor=spike_counts, lw=3)
cobweb = [0.7237, 0.73175, 0.72605, 0.72494, 0.72437]
cobweb_priors = []
cobweb_posteriors = []
for i in 1:length(cobweb)-1
    push!(cobweb_priors, cobweb[i])
    push!(cobweb_posteriors, cobweb[i])
    push!(cobweb_priors, cobweb[i])
    push!(cobweb_posteriors, cobweb[i+1])
end
push!(cobweb_priors, cobweb[end])
push!(cobweb_posteriors, cobweb[end])
push!(cobweb_priors, cobweb[end])
push!(cobweb_posteriors, cobweb[1])
push!(cobweb_priors, cobweb[1])
push!(cobweb_posteriors, cobweb[1])
plot!(
    plt,
    cobweb_priors,
    cobweb_posteriors,
    color=:black
)


println(Ca_initial)

chunk, index, true_ΔCa, true_Δx = paramsToChunkAndIndex(-40.99, -1.57796)
@load "$(scan_directory)/chunk_$(chunk).jld2" sol
maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx))))
plotCaX(true_ΔCa, true_Δx)
println(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx))))

@load "$(scan_directory)/chunk_1_ranges.jld2" ranges
println(ranges)

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
    @load "$(scan_directory)/chunk_$(chunk).jld2" sol
    if maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx)))) > 1
        println("Ca = $(Ca): $(maxSTOsPerBurst(cleanup(mmoSymbolics(sol[index], makeParams(true_ΔCa, true_Δx)))))")
    end
end
