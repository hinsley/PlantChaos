module Plant

export melibeNew, melibeNew!, melibeNewReverse!, default_params, default_state,
       Vs, ah, bh, hinf, am, bm, minf, an, bn, ninf, xinf

using StaticArrays

default_params = @SVector Float32[
    1.0e0,    # Cₘ
    4.0e0,    # gI
    0.3e0,    # gK
    0.0e0,    # gₕ
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

 Vs(V) = (127.0f0 * V + 8265.0f0) / 105.0f0

 am(V) = 0.1f0 * (50.0f0 - Vs(V)) / (exp((50.0f0 - Vs(V)) / 10.0f0) - 1.0f0)
 bm(V) = 4.0f0 * exp((25.0f0 - Vs(V))/18.0f0)
 minf(V) = am(V) / (am(V) + bm(V))
# Fast inward sodium and calcium current
 II(p, h, V) = @inbounds p[2] * h * minf(V)^3.0f0 * (V - @inbounds p[8])

 ah(V) = 0.07f0 * exp((25.0f0 - Vs(V)) / 20.0f0)
 bh(V) = 1.0f0 / (1.0f0 + exp((55.0f0 - Vs(V)) / 10.0f0))
 hinf(V) = ah(V) / (ah(V) + bh(V))
 th(V) = 12.5f0 / (ah(V) + bh(V))
 Ih(p, y, V) = @inbounds p[4] * y * (V - @inbounds p[10]) / (1.0f0 + exp(-(63.0f0 + V) / 7.8f0))^3.0f0
 dh(h, V) = (hinf(V) - h) / th(V)

 an(V) = 0.01f0 * (55.0f0 - Vs(V)) / (exp((55.0f0 - Vs(V)) / 10.0f0) - 1.0f0)
 bn(V) = 0.125f0 * exp((45.0f0 - Vs(V)) / 80.0f0)
 ninf(V) = an(V) / (an(V) + bn(V))
 tn(V) = 12.5f0 / (an(V) + bn(V))
 IK(p, n, V) = @inbounds p[3] * n^4.0f0 * (V - p[9])
 dn(n, V) = (ninf(V) - n) / tn(V)

 xinf(p, V) = 1.0f0 / (1.0f0 + exp(0.15f0 * (@inbounds p[16] - V - 50.0f0)))
 IT(p, x, V) = @inbounds p[6] * x * (V - @inbounds p[8])
 dx(p, x, V) = (xinf(p, V) - x) / @inbounds p[14]
 xinfinv(p, xinf) = @inbounds p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.

 dy(y, V) = (1.0 / (1.0 + exp(10.0 * (V + 50.0))) - y) / (14.2 + 20.8 / (1.0 + exp((V + 68.0) / 2.2)))

 Ileak(p, V) = @inbounds p[5] * (V - @inbounds p[11])

 IKCa(p, Ca, V) = @inbounds p[7] * Ca * (V - @inbounds p[9]) / (0.5f0 + Ca)
 dCa(p, Ca, x, V) = @inbounds p[15] * (@inbounds p[13] * x * (@inbounds p[12] - V + @inbounds p[17]) - Ca)

 function dV(p, x, n, h, Ca, V)
    # TODO: Add a function for Isyn per (12) in the appendix of the paper.
    return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) + Ileak(p, V)) #/ p[1]
end
 function dV(p, x, y, n, h, Ca, V, Isyn)
    # TODO: Add a function for Isyn per (12) in the appendix of the paper.
    return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) + Ih(p, y, V) + Ileak(p, V) + Isyn) / p[1]
end

function melibeNew(u::AbstractArray{T}, p, t) where T
    @fastmath @inbounds begin
        x,n,h,Ca,V = u

        Vs::T = (127f0*x + 8265f0) / 105f0
        am::T = 0.1f0 * (50f0 - Vs) / (exp((50f0 - Vs) / 10f0) - 1f0)
        bm::T = 4f0 * exp((25f0 - Vs)/18f0)
        minf::T = am / (am + bm)
        II::T =  p[2] * h * minf^3.0f0 * (V -  p[8])

        ah::T = 0.07f0 * exp((25f0 - Vs) / 20f0)
        bh::T = 1f0 / (1f0 + exp((55f0 - Vs) / 10f0))
        hinf::T = ah / (ah + bh)
        th::T = 12.5f0 / (ah + bh)
        Ih::T =  p[4] * h * (V -  p[10]) / (1f0 + exp(-(63f0 + V) / 7.8f0))^3f0
        dh::T = (hinf - h) / th

        an::T = 0.01f0 * (55f0 - Vs) / (exp((55f0 - Vs) / 10f0) - 1f0)
        bn::T = 0.125f0 * exp((45f0 - Vs) / 80f0)
        ninf::T = an / (an + bn)
        tn::T = 12.5f0 / (an + bn)
        IK::T =  p[3] * n^4.0f0 * (V -  p[9])
        dn::T = (ninf - n) / tn

        xinf::T = 1f0 / (1f0 + exp(0.15f0 * ( p[16] - V - 50f0)))
        IT::T =  p[6] * x * (V -  p[8])
        dx::T = (xinf - x) /  p[14]

        Ileak::T =  p[5] * (V -  p[11])

        IKCa::T = p[7] * Ca * (V -  p[9]) / (0.5f0 + Ca)
        dCa::T = p[15] * ( p[13] * x * ( p[12] - V +  p[17]) - Ca)

        dV::T = -(II + IK + IT + IKCa + Ileak) #/p[1]
    end
    return SVector{5}(dx, dn, dh, dCa, dV)
end

function melibeNew!(du, u, p, t)
    @inbounds du[1] = @inline dx(p, u[1], u[5])
    @inbounds du[2] = @inline dn(u[2], u[5])
    @inbounds du[3] = @inline dh(u[3], u[5])
    @inbounds du[4] = @inline dCa(p, u[4], u[1], u[5])
    @inbounds du[5] = @inline dV(p, u[1], u[2], u[3], u[4], u[5])
end

function melibeNewReverse!(du, u, p, t)
    @inbounds du[1] = @inline -dx(p, u[1], u[5])
    @inbounds du[2] = @inline -dn(u[3], u[5])
    @inbounds du[3] = @inline -dh(u[4], u[5])
    @inbounds du[4] = @inline -dCa(p, u[4], u[1], u[5])
    @inbounds du[5] = @inline -dV(p, u[1], u[2], u[3], u[4], u[5])
end

default_state = @SVector Float32[
    0.8e0;     # x
    0.137e0;   # n
    0.389e0;   # h
    0.8e0;     # Ca
    -62.0e0;   # V
]

end # module