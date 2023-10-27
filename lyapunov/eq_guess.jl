using Roots
hinf = Plant.hinf
minf = Plant.minf
ninf = Plant.ninf
xinf = Plant.xinf

xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.
IKCa(p, V) = p[2]*hinf(V)*minf(V)^3.0f0*(p[8]-V) + p[3]*ninf(V)^4.0f0*(p[9]-V) + p[6]*xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end

function Ca_null_Ca(p, v)
    return p[13]*xinf(p, v)*(p[12]-v+p[17])
end

function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

function Ca_x_eq(p; which_root=Nothing)
    # Finds the equilibrium in the slow subsystem.
    # which_root: which root to return (if there are multiple)
    # Returns: v_eq, Ca_eq, x_eq
    v_eqs = find_zeros(v -> Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    if which_root == Nothing
        v_eq = length(v_eqs) > 1 ? v_eqs[2] : v_eqs[1]
    else
        v_eq = v_eqs[which_root]
    end
    Ca_eq = Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

function eq(p; which_root=Nothing)
    # Returns the full state vector of the equilibrium.
    # If using y and h currents, make sure to use finf functions for those.
    v_eq, Ca_eq, x_eq = Ca_x_eq(p; which_root=which_root)
    return [x_eq, Plant.ninf(v_eq), Plant.default_state[4], Ca_eq, v_eq]#, Plant.default_state[7]]
end

eq(ps[1].p)

