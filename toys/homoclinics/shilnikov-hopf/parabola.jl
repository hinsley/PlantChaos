using Pkg
Pkg.instantiate()

using CairoMakie, OrdinaryDiffEq, StaticArrays, FileIO,
    Roots, DelimitedFiles, NonlinearSolve,
    LinearAlgebra, ForwardDiff, Optim

include("../../../map_vis/return_map_utils.jl")
include("../../../model/Plant.jl")
include("../../../tools/equilibria.jl")
using .Plant
using Peaks, Interpolations

function xreturn(prob, x)
    # get initial conditions by linear interpolation
    ics = SVector{6}(Equilibria.dune(prob.p.p, x, prob.p.eq[5]))
    # solve the map
    prob = remake(prob, u0 = ics)
    sol = solve(prob, RK4(), abstol = 1e-8, reltol = 1e-8, callback = cb)
    # return the final value
    return sol[1,end]
end

Ca_shift = -40
x_shift = -1.33
function compute_full_map(Ca_shift, x_shift)
    p = SVector{17}(vcat(Plant.default_params[1:15], [x_shift, Ca_shift]))
    u0 = convert.(Float64, Plant.default_state)
    map_resolution = 1000
    preimage_range = (0.3, 0) # Distance from the equilibrium.

    set_theme!(theme_black())

    ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
    map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
    monteprob = EnsembleProblem(map_prob, output_func= output_func, safetycopy=false)

    # calculate and unpack all data needed for plotting
    result = calculate_return_map(monteprob, ics_probs, p, preimage_range..., resolution=map_resolution)
    preimage = result[1]
    xmap = result[2]
    cass = result[3]
    xss = result[4]
    vss = result[5]
    ln1 = result[6]
    ln2 = result[7]
    lerp = result[8]
    eq = result[9]

    function flat_maxima(xmap, flatness_threshold)
        # Find local maxima in xmap.
        maxima_indices, _ = Peaks.findmaxima(xmap)

        # Filter maxima based on flatness criteria.
        valid_maxima_indices = Int[]
        for idx in maxima_indices[2:end-1]
            try
                # Check if the maximum is flat.
                if all(abs(xmap[idx] - xmap[i]) <= flatness_threshold for i in (idx-1):(idx+1))
                    push!(valid_maxima_indices, idx)
                end
            catch BoundsError
                continue
            end
        end

        return valid_maxima_indices
    end

    flatness_threshold = 1e-3  # Adjust this for the flatness threshold.

    flatmaxes = flat_maxima(xmap, flatness_threshold)
    flat_maxima_values = xmap[flatmaxes]

    # Sharpen the flat maxima.
    for (flatmax_i, xmap_i) in enumerate(flatmaxes) # Maybe get rid of enumerate when doing continuation for speed?
        opt = optimize(x -> -xreturn(remake(map_prob, p=(p=p, eq=eq)), x), preimage[xmap_i+1], preimage[xmap_i-1])
        xmap[xmap_i] = -Optim.minimum(opt)
        flat_maxima_values[flatmax_i] = xmap[xmap_i]
    end

    # Refine the saddle PO preimage height.
    begin
        saddle_po_preimage = calculate_hom_box(xmap, preimage)
        old_saddle_po = nothing
        saddle_po_refinement_iterates = 0
        while saddle_po_preimage != old_saddle_po # Iterate until convergence.
            old_saddle_po = saddle_po_preimage
            # Solve the saddle PO.
            saddle_po_xmap = xreturn(remake(map_prob, p=(p=p, eq=eq)), saddle_po_preimage)
            # Insert (saddle_po_preimage, saddle_po_xmap) into the map.
            # Get the index of the last preimage value less than saddle_po_preimage.
            insertion_idx = findfirst(x -> x < saddle_po_preimage, preimage)
            insert!(preimage, insertion_idx, saddle_po_preimage)
            insert!(xmap, insertion_idx, saddle_po_xmap)
            saddle_po_preimage = calculate_hom_box(xmap, preimage)
            saddle_po_refinement_iterates += 1
        end
        #println("Saddle PO location converged after $(saddle_po_refinement_iterates-1) refinements.")
    end

    return preimage,
           xmap,
           cass,
           xss,
           vss,
           ln1,
           ln2,
           lerp,
           eq,
           flatmaxes,
           flat_maxima_values,
           saddle_po_preimage,
           map_prob
end

preimage, xmap, cass, xss, vss, ln1, ln2, lerp, eq, flatmaxes, flat_maxima_values, saddle_po_preimage, map_prob = compute_full_map(Ca_shift, x_shift)

# Plot the map
function plot_map(preimage, xmap, ln1, ln2)
    fig = Figure(resolution=(1500, 1000).*1.3)
    
    mapax = Axis(fig[1,1], aspect=1)
    mapax.title = "x min map"
    mapax.xlabel = rich("x", subscript("n"))
    mapax.ylabel = rich("x", subscript("n+1"))

    lines!(mapax, preimage, xmap, color = range(0.,1., length=length(xmap)), colormap = :thermal)
    lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)
    lines!(mapax, ln1, color = :white, linewidth = 1.0, linestyle = :dash)
    lines!(mapax, ln2, color = :pink, linestyle = :dash, linewidth = 1)

    display(fig)
end
plot_map(preimage, xmap, ln1, ln2)

##### Locate an initial point on the Shilnikov-Hopf homoclinic parabola.

# Initial parameters p must be beneath the Shilnikov-Hopf parabola (in the quiescent region).

function recompute_map_locally(Ca_shift, x_shift, eq, critical_point, critical_point_recompute_radius, saddle_po_preimage, saddle_po_recompute_radius)
    p = SVector{17}(vcat(Plant.default_params[1:15], [x_shift, Ca_shift]))
    ##### Recompute the map around the saddle PO and the critical point without computing the entire map.
    # Generate a new preimage collection for the saddle PO in the map -- this is needed for the refinement algorithm to work with.
    saddle_po_nbhd_preimage = [
        saddle_po_preimage + saddle_po_recompute_radius,
        saddle_po_preimage - saddle_po_recompute_radius
    ]
    # Recompute the heights in the saddle PO neighborhood (nbhd).
    try
        saddle_po_nbhd_xmap = [xreturn(remake(map_prob, p=(p=p, eq=eq)), x) for x in saddle_po_nbhd_preimage]
        # Compute the new saddle PO.
        saddle_po_preimage = calculate_hom_box(saddle_po_nbhd_xmap, saddle_po_nbhd_preimage)
        # Refine the saddle PO until convergence.
        begin
            old_saddle_po = nothing
            saddle_po_refinement_iterates = 0
            while saddle_po_preimage != old_saddle_po # Iterate until convergence.
                old_saddle_po = saddle_po_preimage
                # Solve the saddle PO.
                saddle_po_xmap = nothing
                try
                    saddle_po_xmap = xreturn(remake(map_prob, p=(p=p, eq=eq)), saddle_po_preimage)
                catch DomainError # Dune solver for fast subsystem ICs failed.
                    break
                end
                # Insert (saddle_po_preimage, saddle_po_xmap) into the map.
                # Get the index of the last preimage value less than saddle_po_preimage.
                insertion_idx = findfirst(x -> x < saddle_po_preimage, saddle_po_nbhd_preimage)
                insert!(saddle_po_nbhd_preimage, insertion_idx, saddle_po_preimage)
                insert!(saddle_po_nbhd_xmap, insertion_idx, saddle_po_xmap)
                saddle_po_preimage = calculate_hom_box(saddle_po_nbhd_xmap, saddle_po_nbhd_preimage)
                saddle_po_refinement_iterates += 1
            end
            # println("Saddle PO converged at $(saddle_po_preimage) after $(saddle_po_refinement_iterates-1) refinements.")
        end
    catch DomainError
        nothing
    end
    # Refine the critical point.
    opt = optimize(x -> -xreturn(remake(map_prob, p=(p=p, eq=eq)), x), critical_point-critical_point_recompute_radius, critical_point+critical_point_recompute_radius)
    critical_point = Optim.minimizer(opt)
    critical_value = -Optim.minimum(opt)
    #println(critical_value)

    return critical_point, critical_value, saddle_po_preimage
end

function refine_x_shift(Ca_shift, x_shift, eq, flatmaxes, preimage, xmap, critical_point_index, saddle_po_preimage)
    # critical_point_index is which critical point to use for finding the parabola.
    # Selected by index, starting with 1 at the far right.

    step_size = 1e-3 # Initial step size for xshift.
    saddle_po_recompute_radius = 1e-3 # How far to recompute the map around the saddle PO.
    critical_point_recompute_radius = 1e-4 # How far to recompute the map around the critical point.

    critical_point = preimage[flatmaxes[critical_point_index]] # The critical point on the chosen ``finger.''
    critical_value = xmap[flatmaxes[critical_point_index]] # The height of the critical point on the chosen ``finger.''
    while critical_value > saddle_po_preimage # Move up until passing the Shilnikov-Hopf parabola.
        # Move the finger up.
        x_shift += step_size
        critical_point, critical_value, saddle_po_preimage = recompute_map_locally(
            Ca_shift,
            x_shift,
            eq,
            critical_point,
            critical_point_recompute_radius,
            saddle_po_preimage,
            saddle_po_recompute_radius
        )
    end

    last_direction_up = true # Whether the last step was up or down.
    old_x_shift = nothing
    while critical_value != saddle_po_preimage && x_shift != old_x_shift
        old_x_shift = x_shift
        step_size /= 2
        if last_direction_up
            x_shift -= step_size
        else
            x_shift += step_size
        end
        last_direction_up = !last_direction_up
        critical_point, critical_value, saddle_po_preimage = recompute_map_locally(
            Ca_shift,
            x_shift,
            critical_point,
            critical_point_recompute_radius,
            saddle_po_preimage,
            saddle_po_recompute_radius
        )
    end
    
    return x_shift
end
x_shift = refine_x_shift(Ca_shift, x_shift, eq, flatmaxes, preimage, xmap, 1, saddle_po_preimage)

for Ca_shift in -40.0:0.01:-30.0
    preimage, xmap, cass, xss, vss, ln1, ln2, lerp, eq, flatmaxes, flat_maxima_values, saddle_po_preimage, map_prob = compute_full_map(Ca_shift, -1.33)
    x_shift = refine_x_shift(Ca_shift, x_shift, eq, flatmaxes, preimage, xmap, 1, saddle_po_preimage)
    #println("Shilnikov-Hopf parabola contains ($(Ca_shift), $(x_shift)).")
    println("$(Ca_shift), $(x_shift)")
end
