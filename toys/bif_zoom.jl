import Pkg; Pkg.add("CairoMakie")
using CairoMakie
using DelimitedFiles
import Pkg; Pkg.add("FileIO")
using FileIO
f = Figure(resolution = (1500, 1000), backgroundcolor = :transparent)
ax = Axis(f[1, 1], backgroundcolor = :transparent)
# lines!(readdlm("./explorer/snic.csv", ',', Float64), color=:black, linewidth=3)
lines!(readdlm("./explorer/homoclinic.csv", ',', Float64), label="homoclinic", color=:black, linewidth=3,linestyle=:dashdot)
hopf = readdlm("./explorer/hopf.csv", ',', Float64)
lines!(hopf[:,1:1063], label="subcritical", color=:black, linewidth=3, linestyle=:dot)
lines!(hopf[:,1063:end], label="supercritical", color=:black, linewidth=3)
lines!( readdlm("./explorer/snpo.csv", ',', Float64), label="snpo", color=:black, linewidth = 3, linestyle=:dash)
limits!(-20, 100, -3.75, -2)
axislegend(ax, position=:lb, bgcolor=:transparent, labelcolor=:black, patchsize=(40.0f0, 20.0f0))
save("bif.svg", f)