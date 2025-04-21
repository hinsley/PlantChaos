using Pkg
Pkg.activate("../lyapunov_old/")
Pkg.instantiate()
using GLMakie, StaticArrays, OrdinaryDiffEq, LinearAlgebra, ForwardDiff, ImageShow
using DynamicalSystems, ProgressMeter, JLD2
import LinearAlgebra

# Include model definition
include("../model/Plant.jl")
function melibeNew(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        Plant.dx(p, u[1], u[6]),
        Plant.dy(u[2], u[6]),
        Plant.dn(u[3], u[6]),
        Plant.dh(u[4], u[6]),
        Plant.dCa(p, u[5], u[1], u[6]),
        Plant.dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

# Initial conditions
u0 = @SVector Float64[
    .2;     # x
    .1;     # y
    0.137e0;   # n
    0.389e0;   # h
    1.0e0;     # Cadisp
    -62.0e0;   # V
]

# Parameter space setup
resolution = 1000
start_Ca_shift = -50.0
end_Ca_shift = -20.0
start_x_shift = -1.6
end_x_shift = 0.4
Ca_shifts = LinRange(start_Ca_shift, end_Ca_shift, resolution)
x_shifts = LinRange(start_x_shift, end_x_shift, resolution)

# Define all gh values to scan over
gh_values = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.02]

# Helper functions for visualization
function shape(v, mn, mx, bot, top)
    v = v > top ? top : v
    v = v < bot ? bot : v
    mx == mn && return 0.0
    return (v - bot) / (top - bot) 
end

# Function to compute Lyapunov exponents for a given gh value
function compute_lyapunov_for_gh(gh, resolution)
    # Set up parameters
    ps = [[Plant.default_params[1:3];[gh];Plant.default_params[5:15]; [x_shifts[i], Ca_shifts[j]]] 
         for i in 1:resolution, j in 1:resolution]
    
    # Allocate space for Lyapunov exponents
    lyap_vals = zeros(4, resolution, resolution)
    
    # Setup progress bar
    total_points = resolution * resolution
    progress = Progress(total_points, desc="Parameter Scan for gh = $gh", dt=1)
    
    # Parallel computation
    Threads.@threads for i in 1:resolution
        for j in 1:resolution
            p_ij = ps[i, j]
            sys = ContinuousDynamicalSystem(melibeNew, u0, p_ij)
            # Use fewer steps if runtime is excessive
            lyaps = lyapunovspectrum(sys, 500000)
            # Store the exponents
            lyap_vals[1, i, j] = length(lyaps) >= 1 ? lyaps[1] : 0
            lyap_vals[2, i, j] = length(lyaps) >= 2 ? lyaps[2] : 0
            lyap_vals[3, i, j] = length(lyaps) >= 3 ? lyaps[3] : 0
            lyap_vals[4, i, j] = length(lyaps) >= 4 ? lyaps[4] : 0
            next!(progress)
        end
    end
    
    return lyap_vals
end

# Function to create RGB image from Lyapunov exponents
function create_lyapunov_image(lyap_vals)
    min1, max1 = extrema(lyap_vals[1, :, :])
    min2, max2 = extrema(lyap_vals[2, :, :])
    
    # Calculate sum of positive Lyapunov exponents
    top3 = lyap_vals[1, :, :] .+ lyap_vals[2, :, :] .+ lyap_vals[3, :, :]
    tp = [top > 0 ? top : 0.0 for top in top3]
    min4, max4 = extrema(tp)
    
    # Create RGB image
    img = rotl90([RGBf(
                shape(lyap_vals[1, i, j], min1, max1, 0.0, 3e-5),
                shape(lyap_vals[2, i, j], min2, max2, 0.0, 0.0000025),
                shape(tp[i,j], min4, max4, 0, 0.00001)
            ) for j in 1:resolution, i in 1:resolution])
            
    return img, min1, max1
end

# Dictionary to store all computed Lyapunov exponents
all_lyapunov_data = Dict{Float64, Array{Float64, 3}}()

# We already have the data for gh = 0.0005, so store it
all_lyapunov_data[0.0005] = lyap_vals

# Compute for other gh values
for gh in gh_values
    # Skip 0.0005 as we already have it
    if gh ≈ 0.0005
        println("Skipping gh = 0.0005 as data already exists")
        continue
    end
    
    println("Computing Lyapunov exponents for gh = $gh")
    lyap_vals_gh = compute_lyapunov_for_gh(gh, resolution)
    all_lyapunov_data[gh] = lyap_vals_gh
end

# Create a comprehensive figure with all parameter scans
function create_multi_gh_figure(all_lyapunov_data, gh_values, Ca_shifts, x_shifts)
    n_gh = length(gh_values)
    
    # Determine grid layout based on number of gh values
    n_cols = min(4, n_gh)  # Maximum 4 columns
    n_rows = ceil(Int, n_gh / n_cols)
    
    # Create figure
    fig = Figure(size=(300*n_cols, 250*n_rows))
    
    # Create subfigures
    for (idx, gh) in enumerate(gh_values)
        row = ceil(Int, idx / n_cols)
        col = ((idx - 1) % n_cols) + 1
        
        # Get lyapunov data
        lyap_data = all_lyapunov_data[gh]
        
        # Create image
        img, min_lyap, max_lyap = create_lyapunov_image(lyap_data)
        
        # Create axis
        ax = Axis(fig[row, col], aspect=1,
                 title="gh = $gh",
                 xlabel="Ca shift",
                 ylabel="x shift")
        
        # Plot image
        image!(ax, Ca_shifts, x_shifts, rotr90(img))
        
        # Add annotation for max Lyapunov value
        max_val = maximum(lyap_data[1,:,:])
        chaos_percent = 100 * count(x -> x > 0, lyap_data[1,:,:]) / length(lyap_data[1,:,:])
        
        text!(ax, Ca_shifts[1], x_shifts[end]-0.1, 
              text="λ_max = $(round(max_val, digits=7))\n$(round(chaos_percent, digits=1))% chaotic",
              font=:bold, align=(:left, :top), 
              color=:white, textsize=12)
    end
    
    # Add a global title
    Label(fig[0, :], "Lyapunov Exponent Maps for Different gh Values", 
          fontsize=20, font=:bold, padding=(0, 0, 20, 0))
    
    # Add a colorbar explanation
    colorgrid = fig[n_rows+1, 1:n_cols]
    Label(colorgrid[1, 1], "Color Channels: Red = λ₁ (largest exponent), Green = λ₂, Blue = Sum of positive exponents", 
          fontsize=14)
    
    return fig
end

# Create comprehensive figure
multi_gh_fig = create_multi_gh_figure(all_lyapunov_data, gh_values, Ca_shifts, x_shifts)

# Save the figure
save("lyapunov_multiple_gh.png", multi_gh_fig)

# Save all computed data
@save "lyapunov_all_gh.jld2" all_lyapunov_data gh_values Ca_shifts x_shifts

# Display the figure
display(multi_gh_fig)

# Print summary statistics
println("\nSummary Statistics:")
for gh in gh_values
    lyap_data = all_lyapunov_data[gh]
    max_val = maximum(lyap_data[1,:,:])
    chaos_percent = 100 * count(x -> x > 0, lyap_data[1,:,:]) / length(lyap_data[1,:,:])
    println("gh = $gh: Max Lyapunov = $max_val, Chaotic regions: $(round(chaos_percent, digits=2))%")
end