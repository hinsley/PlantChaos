module Plotting export plot_slow_system

using Plots

function plot_slow_system(sol)
    plot(sol, idxs=(5, 1), xlabel="\$Ca\$", ylabel="\$x\$")
end

end