function progress_for_one_step!(solver, u)
    for i = 1:10
        step!(solver[])
    end
    solver[] = solver[]
    push!(u[],solver[].integ.u)
    u[] = u[]
end

#set length of stored trajectory with slider
maxpoints = 2500
u::Observable{CircularBuffer{SVector{7, Float32}}} = Observable(CircularBuffer{SVector{7,Float32}}(maxpoints))

# create initial trajectory
for i = 1:maxpoints[]
    progress_for_one_step!(dynsys, u)
end
traj = @lift map(x -> Point3f(x[[5,1,6]]...), $u)

lines!(trajax, traj, colormap = :devon, color = @lift 1:$maxpoints)

##Interactivity

isrunning = Observable(true)
delay = Observable(0) #@lift 1/10^$(speedslider.slider.value)

function run_traj()
    i=1
    @async while isrunning[]
        isopen(fig.scene) || break # ensures computations stop if closed window
        progress_for_one_step!(dynsys, u)
        sleep(delay[]) # or `yield()` instead
        if i%10 == 0
            autolimits!(trajax)
        end
        i+=1
    end
end

run_traj()

on(pausebutton.clicks) do x
    isrunning[] = !isrunning[]
    schedule(runtask)
end

on(clearbutton.clicks) do clicks
    ul = last(u[])
    empty!(u[])
    push!(u[], ul)
    u[] = u[]
end

on(resetbutton.clicks) do clicks
    empty!(u[])
    push!(u[], u0[])
    u[] = u[]
end

@time progress_for_one_step!(dynsys, u)
