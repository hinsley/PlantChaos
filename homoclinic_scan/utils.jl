# define ODE
function f(u,p,t)
    p = p.p
    Plant.melibeNew(u,p,t)
end
mutable struct Params
    p::Vector{Float64}
    count::Int
    v_eq::Float64
end
function makespace(space)
    sz = size(space)
    data = Array{Params}(undef, sz...)
    u0s = Array{SVector{6,Float64}}(undef, size(space)...)
    Threads.@threads for i in 1:sz[1]
        for j in 1:sz[2]
            p = vcat(Plant.default_params[1:15], space[i,j]...)
            v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
            local u0
            local v_eq
            if length(v_eqs) < 4
                u0 = fill(NaN, 6)
                v_eq = NaN
            else
                v_eq = v_eqs[3]
                Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
                x_eq = Plant.xinf(p, v_eq)
                eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
                jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
                vals,vecs = eigen(jac)
                _,ix = findmax(real.(vals))
                eps = .001
                vec = real.(vecs)[:,ix]
                u0 = SVector{6}(eq - eps * vec * sign(vec[1]))
            end
            u0s[i,j] = u0
            data[i,j] = Params(p, 0, v_eq)
        end
    end
    return data, u0s
end




# define condition for catching spikes
function dV(p, x, y, n, h, Ca, V)
    p = p.p
   dV = -(
       Plant.II(p, h, V) + 
       Plant.IK(p, n, V) + 
       Plant.IT(p, x, V) + 
       Plant.IKCa(p, Ca, V) + 
       Plant.Ih(p, y, V) + 
       Plant.Ileak(p, V)
   ) / p[1]
end
function spike_condition(u, t, integrator)
   -f(u, integrator.p, 0.0)[6]
end
function spike_affect!(integrator)
   v = integrator.u[6]
   if v < integrator.p.v_eq
        terminate!(integrator)
   else
       integrator.p.count  += 1
   end
end

function output_func(sol,i)
    #return (sol, false)
    return (sol.prob.p.count, false)
end
