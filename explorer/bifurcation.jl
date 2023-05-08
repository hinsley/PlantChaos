using DelimitedFiles
using FileIO

img = rotr90(load("./explorer/ISI_variance.png"))
min_x, max_x = -130, 100
min_y, max_y = -12, 15
image!(bifax, [min_x, max_x], [min_y, max_y], img, interpolate=false)

hopf = readdlm("./explorer/hopf.csv", ',', Float64)
lines!(bifax, hopf, label="hopf")
lines!(bifax, readdlm("./explorer/homoclinic.csv", ',', Float64), label="homoclinic")
lines!(bifax, readdlm("./explorer/snic.csv", ',', Float64), label="snic")
snpo = readdlm("./explorer/snpo.csv", ',', Float64)
lines!(bifax, snpo, label="snpo")

scatter!(bifax, hopf[1,2], hopf[2,2], color=:blue, marker=:star4, label="BT", markersize=16)
scatter!(bifax, hopf[1,1063], hopf[2,1063], color=:red, marker=:star6, label="GH", markersize=16)
scatter!(bifax, hopf[1,8011], hopf[2,8011], color=:green, marker=:star8, label="GH", markersize=16)
scatter!(bifax, snpo[1,827], snpo[2,827], color=:orange, marker=:star5, label="CPC", markersize=16)

bifaxpoint = @lift Point2f( $p[17],$p[16])
scatter!(bifax, bifaxpoint)

axislegend(bifax, position=:lb)

limits!(bifax, -150, 400, -20, 50)

bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    delCa, delx = pars
    bifax.title = "Bifurcation Diagram (ΔCa: $delCa, Δx: $delx)"
    p.val = (p[][1:end-2]..., delx, delCa)
    auto_dt_reset!(dynsys[].integ)
    p[] = p[]
    build_map!(map_prob, mapics[])
    #reset_limits!(mapax)
end