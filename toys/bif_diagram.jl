using DelimitedFiles
using Plots

function plot_bif_diagram!(plt)
    hopf = readdlm("../explorer/hopf.csv", ',', Float64)
    homoclinic = readdlm("../explorer/homoclinic.csv", ',', Float64)
    snic = readdlm("../explorer/snic.csv", ',', Float64)
    plot!(plt, hopf, label="hopf")
    plot!(plt, homoclinic, label="homoclinc")
    plot!(plt, snic, label="snic")

    scatter!(bifax, hopf[1,2], hopf[2,2], color=:blue, marker=:star4, label="BT", markersize=16)
    scatter!(bifax, hopf[1,1063], hopf[2,1063], color=:red, marker=:star6, label="GH", markersize=16)
    scatter!(bifax, hopf[1,8011], hopf[2,8011], color=:green, marker=:star8, label="GH", markersize=16)
end