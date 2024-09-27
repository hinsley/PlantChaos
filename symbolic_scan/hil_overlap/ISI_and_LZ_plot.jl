using Pkg
Pkg.activate("./symbolic_scan")
Pkg.instantiate()

# Imports.
using Colors
using Images
using JLD2
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf
using Roots
using StaticArrays
using Statistics

@time begin
    # Create the plot.
    plt = plot(
        title="",
        # Margins for plot with frame
        # left_margin=40mm,
        # bottom_margin=32mm,
        # guidefontsize=48,
        # tickfontsize=32,
        # xticks=[-60, -40, -20, 0, 20, 40, 60, 80, 100],
        # xlabel="Δ[Ca]",
        # ylabel="ΔVx",
        # Margins for plot without frame
        left_margin=-4mm,
        bottom_margin=-2mm,
        right_margin=-4mm,
        top_margin=-2mm,
        axis=false,
        grid=false,
        showaxis=false,
        ticks=nothing,
        size=(3840, 2160),
        # LZ complexity zoom
        xlim=(-60, 100),
        ylim=(-4, 1),
        aspect_ratio=18
    )

    # Load pre-rendered images.
    isi_variances_image = load("./symbolic_scan/hil_overlap/isi_variances.png")
    lz_complexities_image = load("./symbolic_scan/hil_overlap/lz_complexities.png")

    # # Plot ISI variances.
    plot!(
        plt,
        range(-130, 100, length=size(isi_variances_image, 1)), # ΔCa range
        range(-12, 15, length=size(isi_variances_image, 2)), # ΔVx range
        isi_variances_image,
        seriestype=:image
    )

    # # Plot LZ complexities.
    plot!(
        plt,
        range(-60, 100, length=size(lz_complexities_image, 1)), # ΔCa range
        range(-4, 1, length=size(lz_complexities_image, 2)), # ΔVx range
        lz_complexities_image,
        seriestype=:image
    )

    display(plt)
    # Save the figure as a high-resolution PNG file
    savefig(plt, "./symbolic_scan/hil_overlap/ISI_and_LZ_plot.png")
end