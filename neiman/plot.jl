# ========================= plot.jl =========================
# Load individual Lyapunov maps from disk and generate a
# composite figure summarising all gh values.
# -----------------------------------------------------------

using Pkg
Pkg.activate("../lyapunov_old/")
Pkg.instantiate()

using GLMakie, ColorTypes, JLD2

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
function shape(v, mn, mx, bot, top)
    v = clamp(v, bot, top)
    mx == mn && return 0.0
    return (v - bot) / (top - bot)
end

function create_lyapunov_image(lyap_vals::Array{Float64,3})
    res = size(lyap_vals, 2)
    min1, max1 = extrema(lyap_vals[1,:,:])
    min2, max2 = extrema(lyap_vals[2,:,:])

    pos_sum = max.(0, lyap_vals[1,:,:] .+ lyap_vals[2,:,:] .+ lyap_vals[3,:,:])
    min3, max3 = extrema(pos_sum)

    img = rotl90([RGBf(
        shape(lyap_vals[1,i,j], min1, max1, 0.0, 3e-5),
        shape(lyap_vals[2,i,j], min2, max2, 0.0, 2.5e-6),
        shape(pos_sum[i,j],      min3, max3, 0.0, 1e-5),
    ) for j in 1:res, i in 1:res])

    return img
end

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
const gh_values = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.02]

after_first = Ref(false)
Ca_shifts = Float64[]
x_shifts  = Float64[]
all_data  = Dict{Float64,Array{Float64,3}}()

for gh in gh_values
    fname = "lyapunov_scan_$(replace(string(gh), '.'=>'_')).jld2"
    @info "Loading $fname"
    @load fname lyap_vals Ca_shifts_temp x_shifts_temp
    all_data[gh] = lyap_vals
    if !after_first[]
        Ca_shifts .= Ca_shifts_temp
        x_shifts  .= x_shifts_temp
        after_first[] = true
    end
end

# ------------------------------------------------------------------
# Plotting routine
# ------------------------------------------------------------------
function plot_all(all_data, gh_values, Ca_shifts, x_shifts)
    n = length(gh_values)
    ncols = min(4,n)
    nrows = ceil(Int, n/ncols)

    fig = Figure(size=(300*ncols, 250*nrows))

    for (idx, gh) in enumerate(gh_values)
        r = ceil(Int, idx/ncols)
        c = ((idx-1) % ncols) + 1

        lyap_vals = all_data[gh]
        img = create_lyapunov_image(lyap_vals)

        ax = Axis(fig[r,c]; aspect=1,
                  title="gh = $gh",
                  xlabel="Ca shift",
                  ylabel="x shift")
        image!(ax, Ca_shifts, x_shifts, rotr90(img))

        maxλ = maximum(lyap_vals[1,:,:])
        pcchaos = 100 * count(>(0), lyap_vals[1,:,:]) / length(lyap_vals[1,:,:])

        text!(ax, Ca_shifts[1], x_shifts[end]-0.1,
              text="λ₁ᵐᵃˣ=$(round(maxλ, sigdigits=3))\n$(round(pcchaos, digits=1))% chaotic",
              align=(:left,:top), color=:white, font=:bold)
    end

    Label(fig[0,:], "Lyapunov Exponent Maps", fontsize=20, font=:bold, padding=(0,0,20,0))
    return fig
end

fig = plot_all(all_data, gh_values, Ca_shifts, x_shifts)

save("lyapunov_multiple_gh.png", fig)
display(fig)
