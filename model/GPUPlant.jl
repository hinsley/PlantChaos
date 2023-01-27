module GPUPlant

export melibeNew, default_params, default_state

using StaticArrays

default_params = @SVector Float32[
    1.0e0,    # Cₘ
    4.0e0,    # gI
    0.3e0,    # gK
    0.03e0,   # gₕ
    0.003e0,  # gL
    0.01e0,   # gT
    0.03e0,   # gKCa
    30.0e0,   # EI
    -75.0e0,  # EK
    70.0e0,   # Eₕ
    -40.0e0,  # EL
    140.0e0,  # ECa
    0.0085e0, # Kc
    100.0e0,  # τₓ
    0.0003e0, # ρ
    0.0e0,    # Δx
    0.0e0     # ΔCa
]

function melibeNew(u::AbstractVector{T}, p::AbstractVector{T}, t::T) where T<:AbstractFloat
    Vs(V::T)::T = (127.0 * V + 8265.0) / 105.0

    am(V::T)::T = 0.1 * (50.0 - Vs(V)) / (exp((50.0 - Vs(V)) / 10.0) - 1.0)
    bm(V::T)::T = 4.0 * exp((25.0 - Vs(V))/18.0)
    minf(V::T)::T = am(V) / (am(V) + bm(V))
    # Fast inward sodium and calcium current
    II(p::AbstractVector{T}, h::T, V::T)::T = p[2] * h * minf(V)^3 * (V - p[8])

    ah(V::T)::T = 0.07 * exp((25.0 - Vs(V)) / 20.0)
    bh(V::T)::T = 1.0 / (1.0 + exp((55.0 - Vs(V)) / 10.0))
    hinf(V::T)::T = ah(V) / (ah(V) + bh(V))
    th(V::T)::T = 12.5 / (ah(V) + bh(V))
    Ih(p::AbstractVector{T}, y::T, V::T)::T = p[4] * y * (V - p[10]) / (1.0 + exp((63.0 - V) / 7.8))^3
    dh(h::T, V::T)::T = (hinf(V) - h) / th(V)

    an(V::T)::T = 0.01 * (55.0 - Vs(V)) / (exp((55.0 - Vs(V)) / 10.0) - 1.0)
    bn(V::T)::T = 0.125 * exp((45.0 - Vs(V)) / 80.0)
    ninf(V::T)::T = an(V) / (an(V) + bn(V))
    tn(V::T)::T = 12.5 / (an(V) + bn(V))
    IK(p::AbstractVector{T}, n::T, V::T)::T = p[3] * n^4 * (V - p[9])
    dn(n::T, V::T)::T = (ninf(V) - n) / tn(V)

    xinf(p::AbstractVector{T}, V::T)::T = 1.0 / (1.0 + exp(0.15 * (p[16] - V - 50.0)))
    IT(p::AbstractVector{T}, x::T, V::T)::T = p[6] * x * (V - p[8])
    dx(p::AbstractVector{T}, x::T, V::T)::T = (xinf(p, V) - x) / p[14]

    dy(y::T, V::T)::T = (1.0 / (1.0 + exp(10.0 * (V - 50.0))) - y) / (14.2 + 20.8 / (1.0 + exp((V + 68.0) / 2.2)))

    Ileak(p::AbstractVector{T}, V::T)::T = p[5] * (V - p[11])

    IKCa(p::AbstractVector{T}, Ca::T, V::T)::T = p[7] * Ca * (V - p[9]) / (0.5 + Ca)
    dCa(p::AbstractVector{T}, Ca::T, x::T, V::T)::T = p[15] * (p[13] * x * (p[12] - V + p[17]) - Ca)

    function dV(p::AbstractVector{T}, x::T, y::T, n::T, h::T, Ca::T, V::T, Isyn::T)::T
        # TODO: Add a function for Isyn per (12) in the appendix of the paper.
        return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) + Ih(p, y, V) + Ileak(p, V) + Isyn) / p[1]
    end

    x, y, n, h, Ca, V, Isyn = u

    du1 = dx(p, x, V)
    du2 = dy(y, V)
    du3 = dn(n, V)
    du4 = dh(h, V)
    du5 = dCa(p, Ca, x, V)
    du6 = dV(p, x, y, n, h, Ca, V, Isyn)
    du7 = 0.0e0
    return @SVector T[du1, du2, du3, du4, du5, du6, du7]
end

default_state = @SVector Float32[
    0.8e0;     # x
    5.472e-46; # y
    0.137e0;   # n
    0.389e0;   # h
    0.8e0;     # Ca
    -62.0e0;   # V
    0.0e0      # Isyn
]

end # module