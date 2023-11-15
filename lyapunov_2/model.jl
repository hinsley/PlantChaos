function flyap(u, p)
    x, n , h, Ca, V, x1, n1, h1, Ca1, V1 = u

    Cm = 1.0f0; gI = 4.0f0; gK = 0.3f0; gl = 0.003f0; gx = 0.01f0; 
    gKCa = 0.03f0; Ex = 30.0f0; EK = -75.0f0
    Eh = 70.0f0; El = -40.0f0; ECa = 140.0f0; Kc = 0.0085f0
    tau_x = 100.0f0; rho = 0.0003f0; x_shift = p[1]; Ca_shift = p[2]

    Vs = (127.0f0 * V + 8265.0f0) / 105.0f0
    am = 0.1f0 * (50.0f0 - Vs) / (exp((50.0f0 - Vs) / 10.0f0) - 1.0f0)
    bm = 4.0f0 * exp((25.0f0 - Vs)/18.0f0)
    minf = am / (am + bm)
    INa = gI * h * minf^3.0 * (V - EI)

    ah = 0.07f0 * exp((25.0f0 - Vs) / 20.0f0)
    bh = 1.0f0 / (exp((45.0f0 - Vs) / 10.0f0) + 1.0f0)
    hinf = ah / (ah + bh)
    th = 12.5f0 / (ah + bh)
    dh = (hinf - h) / th
    Ih = gL * h * (V - Eh)

    an = 0.01f0 * (55.0f0 - Vs) / (exp((55.0f0 - Vs) / 10.0f0) - 1.0f0)
    bn = 0.125f0 * exp((45.0f0 - Vs) / 80.0f0)
    ninf = an / (an + bn)
    tn = 12.5f0 / (an + bn)
    dn = (ninf - n) / tn
    IK = gK * n^4.0 * (V - EK)

    xinf = 1.0f0 / (1.0f0 + exp(0.15f0 * (x_shift - V - 50.0f0)))
    dx = (xinf - x) / tau_x
    Ix = gx * x * (V - Ex)

    dCa = rho * (Kc * x * (ECa - V + Ca_shift) - Ca)
    ICa = gKCa * Ca * (V - EK) / (0.5f0 + Ca)

    IL = gl * (V - El)
    
    dV = -(INa + IK + Ix + ICa + Ih + IL) / Cm

    return [dx, dn, dh, dCa, dV, dx1, dn1, dh1, dCa1, dV1]
end