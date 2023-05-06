using DelimitedFiles
using FileIO

isiimg = rotr90(load("./explorer/ISI_variance.png"))
min_x, max_x = -150, 369.675
min_y, max_y = -20, 29.9375
image!(bifax, [min_x, max_x], [min_y, max_y], isiimg, interpolate=false)

beimg = rotr90(load("./explorer/block_entropy.png"))
min_x, max_x = -50, 100
min_y, max_y = -5, 1

image!(bifax, [min_x, max_x], [min_y, max_y], beimg, interpolate=false)

hopf = readdlm("./explorer/hopf.csv", ',', Float64)
lines!(bifax, hopf, label="hopf")
lines!(bifax, readdlm("./explorer/homoclinic.csv", ',', Float64), label="homoclinic")
lines!(bifax, readdlm("./explorer/snic.csv", ',', Float64), label="snic")
snpo = readdlm("./explorer/snpo.csv", ',', Float64)
lines!(bifax, snpo, label="snpo")

scatter!(bifax, hopf[1,2], hopf[2,2], color=:blue, marker=:star4, label="BT", markersize=16)
scatter!(bifax, hopf[1,1063], hopf[2,1063], color=:red, marker=:star6, label="BP", markersize=16)
scatter!(bifax, hopf[1,8011], hopf[2,8011], color=:green, marker=:star8, label="BP", markersize=16)
# scatter!(bifax, snpo[1,827], snpo[2,827], color=:orange, marker=:star5, label="CPC", markersize=16)

bifaxpoint = @lift Point2f( $p[17],$p[16])
scatter!(bifax, bifaxpoint)

axislegend(bifax, position=:lb)

limits!(bifax, -50, 100, -5, 1)

bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    delCa, delx = pars
    p.val = (p[][1:end-2]..., delx, delCa)
    auto_dt_reset!(dynsys[].integ)
    p[] = p[]
    build_map!(map_prob, mapics[])
    #reset_limits!(mapax)
end

Label(bifctrlax[1,1], "ΔCa: ")
bifctrlax[1,2] = delCa_tb = Textbox(fig, validator = Float32, placeholder="$(ΔCa[])", width=150)

on(ΔCa) do delCa
    delCa_tb.displayed_string = string(delCa)
end

Label(bifctrlax[1,3], "Δx: ")
bifctrlax[1,4] = delx_tb = Textbox(fig, validator = Float32, placeholder="$(Δx[])", width=150)

on(Δx) do delx
    delx_tb.displayed_string = string(delx)
end

bifupdatebutton = Button(bifctrlax[1,5], label = "update", buttoncolor = RGBf(.2,.2,.2))

on(bifupdatebutton.clicks) do clicks
    delCa = parse(Float64, delCa_tb.displayed_string[])
    delx = parse(Float64, delx_tb.displayed_string[])
    bifpoint[]=Point2f(delCa, delx)
end