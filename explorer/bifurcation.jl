
image!(bifax,
    range(-50, length = 5, stop = -15),
    range(-2.6, length = 5, stop = 0),
    rotr90(load("./explorer/bifurcation.png")))

Makie.deactivate_interaction!(bifax, :rectanglezoom)
bifpoint = select_point(bifax.scene, marker = :circle)

on(bifpoint) do pars
    delCa, delx = pars
    u0.val = Vector(last(u[]))
    _p = p[]
    p[] = (p[][1:end-2]..., delx, delCa)
end