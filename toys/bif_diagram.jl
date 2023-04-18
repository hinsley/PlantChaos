using DelimitedFiles
using Plots

function plot_bif_diagram!(plt)
    hopf = readdlm("./explorer/hopf.csv", ',', Float64)
    homoclinic = readdlm("./explorer/homoclinic.csv", ',', Float64)
    snic = readdlm("./explorer/snic.csv", ',', Float64)
    plot!(plt, hopf[1,:], hopf[2,:], label="hopf")
    plot!(plt, homoclinic[1,:], homoclinic[2,:], label="homoclinc")
    plot!(plt, snic[1,:], snic[2,:], label="snic")

    scatter!(plt, [hopf[1,2]], [hopf[2,2]], color=:blue, marker=:star4, label="BT", markersize=16)
    scatter!(plt, [hopf[1,1063]], [hopf[2,1063]], color=:red, marker=:star6, label="GH", markersize=16)
    scatter!(plt, [hopf[1,8011]], [hopf[2,8011]], color=:green, marker=:star8, label="GH", markersize=16)
end