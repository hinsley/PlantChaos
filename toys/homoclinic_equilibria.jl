using StaticArrays
using Roots
using Plots
using DelimitedFiles

include("../model/Plant.jl")

homoclinic_values = Float32[
    -9.3005    -13.969
    -9.031     -14.709
    -8.7312    -15.561
    -8.3981    -16.543
    -8.0282    -17.677
    -7.617     -18.994
    -7.1581    -20.533
    -6.6404    -22.36
    -6.041     -24.593
    -5.298     -27.528
    -3.6129    -34.647
    -2.7274    -38.047
    -2.4139    -38.702
    -2.2369    -38.459
    -2.1444    -37.439
    -2.1244    -35.477
    -2.1863    -32.037
    -2.381     -25.371
    -3.2112     4.6834
    -3.1138     69.325
    -2.5365     85.784
    -1.9086     95.3
    -1.1907     97.456
    -0.3424     91.51
     0.68404     77.826
     1.9088      59.236
     3.2808     39.919
     4.712      22.47
     6.1293     7.6196
     7.4828     -4.6939
     8.7455     -14.786
     9.9067     -23.012
     10.969     -29.714
     11.941     -35.181
     12.835     -39.649
     13.661     -43.302
     14.43      -46.281
     15.152     -48.695
     15.835     -50.629
     16.484     -52.142
     17.107     -53.284
     17.71      -54.088
     18.296     -54.577
     18.871     -54.765
     19.438     -54.655
     20.003     -54.242
     20.571     -53.509
    21.146     -52.427
    21.735     -50.947
    22.344     -48.998
    22.986     -46.462
    23.672     -43.165
    24.425     -38.806
    25.276     -32.862
    26.288     -24.244
    27.604     -10.185
    29.718     20.564
    37.569     289.15
]

IKCa(p, V) = p[2]*Plant.hinf(V)*Plant.minf(V)^3.0f0*(p[8]-V) + p[3]*Plant.ninf(V)^4.0f0*(p[9]-V) + p[6]*Plant.xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)

xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end

function Ca_null_Ca(p, v)
    return p[13]*Plant.xinf(p, v)*(p[12]-v+p[17])
end

function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

function Ca_x_eq(p)
    v_eqilibria =  find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))
    return [(v_eq, Ca_null_Ca(p, v_eq), Plant.xinf(p, v_eq)) for v_eq in v_eqilibria]
end

V_range = range(-70, 20, length=1000)

a = Animation()

j = 1
for (Δx, ΔCa) in eachrow(homoclinic_values)
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
        Δx,                       # Δx
        ΔCa                       # ΔCa
    ]

    plt = plot([Ca_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range], label="Ca-nullcline");
    plot!(plt, [x_null_Ca(p, V) for V in V_range], [Plant.xinf(p, V) for V in V_range], label="x-nullcline");
    
    homoclinic_orbit = readdlm("toys/homoclinics/$j.csv", ',', skipstart=1, Float32)
    plot!(plt, homoclinic_orbit[:,1], homoclinic_orbit[:,2], label="homoclinic");

    eqilibria = Ca_x_eq(p)
    scatter!(plt, getfield.(eqilibria, 2), getfield.(eqilibria, 3), label="equilibria");

    scatter!(plt, [homoclinic_orbit[1,1]], [homoclinic_orbit[1,2]], label="start");
    
    xlims!(0, 3)
    ylims!(0, 1)
    title!("(ΔCa: $ΔCa, Δx: $Δx)")
    
    # display(plt)
    
    frame(a, plt)
    j = j + 1
end

gif(a, fps=2)


