using Pkg
Pkg.activate("./ssf_scan")
using GLMakie, OrdinaryDiffEq, Roots, StaticArrays, LinearAlgebra, ForwardDiff
resolution = 10

thsp = range(0.0, 2pi, length = resolution)
csp = range(-42, -30, length = resolution)
xs = -1.125

# find sf equilibrium

# find sf equ