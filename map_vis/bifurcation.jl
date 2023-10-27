
isiimg = rotr90(load("./explorer/ISI_variance.png"))
min_x, max_x = -130, 100
min_y, max_y = -12, 15
image!(bifax, [min_x, max_x], [min_y, max_y], isiimg, interpolate=false)

beimg = rotr90(load("./explorer/block_entropy.png"))
min_x, max_x = -50, 100
min_y, max_y = -5, 1

image!(bifax, [min_x, max_x], [min_y, max_y], beimg, interpolate=false)

hopf = readdlm("./explorer/hopf.csv", ',', Float64)
lines!(bifax, hopf, label="hopf", linewidth = 3)
lines!(bifax, readdlm("./explorer/homoclinic.csv", ',', Float64), label="homoclinic", linewidth = 3)
lines!(bifax, readdlm("./explorer/snic.csv", ',', Float64), label="snic", linewidth = 3)
snpo = readdlm("./explorer/snpo.csv", ',', Float64)
lines!(bifax, snpo, label="snpo", linewidth = 3)

scatter!(bifax, hopf[1,2], hopf[2,2], color=:yellow, marker='■', label="BT", markersize=32)
scatter!(bifax, hopf[1,1063], hopf[2,1063], color=:purple, marker='■', label="BP", markersize=32)
# scatter!(bifax, hopf[1,8011], hopf[2,8011], color=:green, marker=:star8, label="BP", markersize=16)
# scatter!(bifax, snpo[1,827], snpo[2,827], color=:orange, marker=:star5, label="CPC", markersize=16)

bifaxpoint = @lift Point2f( $p[17],$p[16])
scatter!(bifax, bifaxpoint)

axislegend(bifax, position=:rb)

limits!(bifax, -50, 100, -5, 1)

bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    # do not trigger when reset limit
    if !ispressed(mapax, Keyboard.left_control)
        delCa, delx = pars
        p.val = (p[][1:end-2]..., delx, delCa)
        p[] = p[]
        reset_limits!(mapax)
        reset_limits!(trajax)
    end
end

