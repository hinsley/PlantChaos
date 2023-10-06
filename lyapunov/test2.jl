t = collect(.1:.1:1000-.1)
using GLMakie

function accum(a)
    len = 1000
    arr = Float64[]
    t = LinRange(-pi,0.0,len)
    f = t # cos.(t)./2 .+ .5
    tot = sum(f) # 2*len
    for i in len:length(a)
        push!(arr, sum(f.*a[i-len+1:i])./tot)
    end
    return arr
end

function randaccum(a)
    len = 1000
    arr = Float64[]
    for i in len:length(a)
        f = rand(len)
        tot = sum(f)
        push!(arr, sum(f.*a[i-len+1:i])./tot)
    end
    return arr
end

begin
    g(x) = cos(x)
    f = Figure(resolution = (800, 600))
    ax = Axis(f[1, 1])
    #lines!(ax, t, cos.(t))
    cum = cumsum(g.(t))./t
    acc = accum(cum)
    racc = randaccum(cum)
    lines!(ax, t, [0/(t[2]-t[1]) for i in t], color = :black, linewidth = 2)
    #lines!(ax, t, g.(t)./(t[2]-t[1]))
    #lines!(ax, t, cum)
    #lines!(ax, t[1000:end], acc, color = :green )
    #lines!(ax, t[1000:end], racc, color = :red )
    for ix in 1:1:50

        lines!(ax, t[ix:end], cumsum(g.(t[ix:end]))./t[1:end-ix+1], color = :grey, linewidth = .2 , opacity = .4)
    end
    f
end

# 1. compute averages for several several starting points
# 2. average the averages
# 3. compute the moving average of the averages of the averages with random kernel
# 4. compute the moving average of that.

function estimate_limit(loga,dt,max_freq; starting_points = 20, window1 = 2000, window2 = 500)
    #estimate the number of indexes to skip for each starting points
    skip = ceil(Int, max_freq*dt)
    t = dt:dt:length(loga)*dt
    av = zeros(length(loga))
    for i in 1:starting_points
        ac = 0
        for j in 1+skip*(i-1):length(loga)
            ac += loga[j]
            av[j] += ac
        end
    end
    av = av./starting_points./t
    #compute the last window with random kernel
    f = ones(window1)
    tot = sum(f)


    MAKE ACTUALLY MOVING AVG> FIGURE OUT THE MOST EFFICINET WAY
    f.*av[end-window1+1:end]./tot
    av2 = [sum(av2[1+i:window2+i]) for i in 1:length(av2)-window2]
    return av3
end

estimate_limit(cos.(t),.1,100.0) |> lines
-