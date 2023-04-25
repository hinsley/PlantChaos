using DelimitedFiles

hopf = readdlm("./explorer/hopf.csv", ',', Float64)
lines!(bifax, hopf, label="hopf")
lines!(bifax, readdlm("./explorer/homoclinic.csv", ',', Float64), label="homoclinc")
lines!(bifax, readdlm("./explorer/snic.csv", ',', Float64), label="snic")
lines!(bifax, readdlm("./explorer/snpo.csv", ',', Float64), label="snpo")

scatter!(bifax, hopf[1,2], hopf[2,2], color=:blue, marker=:star4, label="BT", markersize=16)
scatter!(bifax, hopf[1,1063], hopf[2,1063], color=:red, marker=:star6, label="GH", markersize=16)
scatter!(bifax, hopf[1,8011], hopf[2,8011], color=:green, marker=:star8, label="GH", markersize=16)

bifaxpoint = @lift Point2f( $p[17],$p[16])
scatter!(bifax, bifaxpoint)

axislegend(bifax, position=:lb)

limits!(bifax, -150, 400, -20, 50)

bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    delCa, delx = pars
    bifax.title = "Bifurcation Diagram (ΔCa: $delCa, Δx: $delx)"
    p[] = (p[][1:end-2]..., delx, delCa)
    auto_dt_reset!(dynsys[].integ)
end