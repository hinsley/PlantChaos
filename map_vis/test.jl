cais = findall(isnan, cass[])
xis = findall(isnan, xss[])
_cas = vcat([cass[][1]],[cass[][i+1] for i in cais[1:end-1]])
_xs = vcat([xss[][1]],[xss[][i+1] for i in xis[1:end-1]])

radius = mapslider.sliders[4].value

a = generate_ics_circle!(ics_probs, p[], eq[], radius[], map_resolution)
[e[5] for e in a] |> lines