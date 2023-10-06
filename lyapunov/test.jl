function test_lyap_plot(sys, T, Ttr)
    d0 = 1e-6
    _d1 = [randn(), 0., randn(4)..., 0.]
    _d2 = _d1/norm(_d1)*d0
    psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])
    λs = Float64[]
    for i in 1:T
        d = λdist(psys)/d0
        λrescale!(psys, d)
        step!(psys)
        if i>Ttr
            push!(λs, log(d))
        end
    end
    # build average of normalized cumsums
    num_avg = 100
    max_period = 10000
    dt = 1.0
    max_i = ceil(Int, max_period/dt)
    λtot = zeros(length(λs)-max_i)
    idxs = floor.(Int, LinRange(1, max_i-1, num_avg))
    for i in idxs
        #return length(collect(1:(length(λs)-max_i)))
        λtot = λtot .+ cumsum(λs[i:end-max_i+i-1])./collect(1:(length(λs)-max_i))
    end
    λtot = λtot ./ length(idxs)
    λs = λtot

    F = Figure()
    traj = trajectory(sys, T, Δt = 1.0)
    println(length(traj[1][:,6]))
    ax = Axis(F[1,1], xlabel = "Time", ylabel = "v")
    #lines!(ax, traj[1][:,6]./maximum(traj[1][:,1]).*maximum(λs), color = :blue)
    ais2 = vcat(zeros(Ttr), λs)
    #lines!(ax, ais2, color = :red)
    lines!(ax, λs, color = :green)
    λs = vcat(zeros(length(λs)-max_i), cumsum(λs[end-max_i:end-1])./collect(1:max_i))
    lines!(ax, λs, color = :blue)
    F
end
test_lyap_plot(sys, 100000, 10000)