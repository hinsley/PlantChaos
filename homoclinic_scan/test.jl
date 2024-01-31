# test f
let
    ΔCa = -38.6268
    Δx = -2.2865
    ΔCa = caspace[2]
    Δx = xspace[3]
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
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

    prob2 = remake(prob, p = Params(p, 0,v_eq), u0 = u0)
    sol = solve(prob2, RK4(), callback = spike_cb)
    println(sol.prob.p.count)

    f = Figure()
    ax1 = Axis(f[1,1], xlabel = "t", ylabel = "V")
    ax2 = Axis(f[2,1], xlabel = "Ca", ylabel = "x")

    lines!(ax1, sol.t, sol[6,:], color = collect(1:length(sol.t)))
    lines!(ax2, sol[5,:], sol[1,:], color = collect(1:length(sol.t)))
    f
end

lines!(ax, [(-38.628,-2.29),(-38.628,-2.27),(-38.616,-2.27),(-38.616,-2.29)], color = :black)

function sharpen_scan(arr::Array{T}) where T
    nrows, ncols = size(arr)
    d = Array{T}undef(nrows*3,ncols*3)
    #set outside edges unchanged
    for i = 1:3
        for j = 1:nrows
            for k = 0:2
                d[j+k,i] = arr[j,1]
                d[j+k,end-i+1] = arr[j,end] 
            end
        end
        for j = 1:cols
            for k = 0:2
                d[i,j+k] = arr[1,j]
                d[end-i+1,j+k] = arr[end,j] 
            end
        end
    end
    # sharpen middle squares
    for i=2:nrows-1
        for j=2:ncols-1
            #center
            center = arr[i,j]
            d[3i+2,3i+2] = center
            #define vars
            U = arr[i,j-1]
            B = arr[i,j+1]
            L = arr[i-1,j]
            R = arr[i+1,j]
            UL = arr[i-1,j-1]
            UR = arr[i-1,j+1]
            BL = arr[i+1,j-1]
            BR = arr[i+1,j+1]
            u = @views d[1]


            if arr[i-1,j] == arr[i,j-1]
                d[3i+1,3j+1] = arr[i-1,j]
            else
                d[3i+1,3j+1] = arr[i,j]
            end
            #top right
            if arr[i-1,j] == arr[i,j+1]
                d[3i+1,3j+3] = arr[i-1,j]
            else
                d[3i+1,3j+3] = arr[i,j]
            end
            #bottom left
            if arr[i+1,j] == arr[i,j-1]
                d[3i+3,3j+1] = arr[i+1,j]
            else
                d[3i+3,3j+1] = arr[i,j]
            end
            #bottom right
            if arr[i+1,j] == arr[i,j+1]
                d[3i+3,3j+3] = arr[i+1,j]
            else
                d[3i+3,3j+3] = arr[i,j]
            end
            #left
        end
    end

