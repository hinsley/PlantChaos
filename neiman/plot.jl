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


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
const gh_values = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.02]
#const gh_values = [0.0, 0.0002, 0.0005, 0.001]

after_first = Ref(false)

all_data  = Dict{Float64,Array{Float64,3}}()

for gh in gh_values
    fname = "./neiman/lyapunov_scan_$(replace(string(gh), '.'=>'_')).jld2"
    @info "Loading $fname"
    @load fname lyap_vals gh Ca_shifts x_shifts
    all_data[gh] = lyap_vals
    global Ca_shifts
    global x_shifts
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

        ax = Axis(fig[r,c]; aspect=1,
                  title="gh = $gh",
                  xlabel="Ca shift",
                  ylabel="x shift")
        
        pos_sum = lyap_vals[1,:,:] .+ lyap_vals[3,:,:]
        heatmap!(ax, Ca_shifts, x_shifts, pos_sum', colormap = Reverse(:RdYlBu), colorrange = (-.0002, .0002))
        maxλ = maximum(lyap_vals[1,:,:])
        pcchaos = 100 * count(>(0.00001), lyap_vals[1,:,:]) / length(lyap_vals[1,:,:])

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

lyap_vals = all_data[.000]
res = size(lyap_vals, 2)
    min1, max1 = extrema(lyap_vals[1,:,:])
    min2, max2 = extrema(lyap_vals[2,:,:])

    pos_sum = lyap_vals[1,:,:] .+ lyap_vals[3,:,:]
    min3, max3 = extrema(pos_sum)

    img = rotl90([RGBf(
        0.,#shape(lyap_vals[1,i,j], min1, max1, 0.0, 3e-5),
        0.,#shape(lyap_vals[2,i,j], min2, max2, 0.0, 2.5e-6),
        shape(pos_sum[i,j],      min3, max3, -1e7, 6e-5),
    ) for j in 1:res, i in 1:res])

heatmap(clamp.(pos_sum, -.0002, .0002)',  colormap = Reverse(:RdYlBu))
heatmap(clamp.(lyap_vals[1,:,:], -.0002, .0002)',  colormap = Reverse(:RdYlBu))
heatmap(clamp.(lyap_vals[3,:,:], -.0001, .000)',  colormap = Reverse(:RdYlBu))


function lyapunov_dimension(λs::AbstractVector{<:Real})
    s = zero(eltype(λs))
    for j in 1:length(λs)
        s += λs[j]
        if s < 0
            # j-th exponent causes the cumulative sum to go negative
            return (j-1) + (s - λs[j]) / abs(λs[j])
        end
    end
    # all partial sums are non-negative ⇒ full dimension
    return float(length(λs))
end


function dimension_map(lyap_vals::Array{Float64,3})
    nx, ny = size(lyap_vals, 2), size(lyap_vals, 3)
    D = Array{Float64}(undef, nx, ny)
    for i in 1:nx, j in 1:ny
        D[i, j] = lyapunov_dimension(lyap_vals[:, i, j])
    end
    return D
end

# Example usage:
dim_map = dimension_map(all_data[.01])
heatmap(Ca_shifts, x_shifts, dim_map')  # to visualize
dim3map = [x ==  0 ? x : 1 for x in dim_map]
heatmap(Ca_shifts, x_shifts, dim3map')