function progress_for_one_step!(solver, traj)
    s = solver[]
    step!(s, 5f0)
    if !(check_error(s.integ) in [ReturnCode.Default, ReturnCode.Success])
        throw(ErrorException("integration gone and broke itself again. Return code is $errcode"))
    end
    push!(traj[], Point3f(s.integ.u[[5,1,6]]))
    traj[] = traj[]
end

#set length of stored trajectory with slider
maxpoints = 2500
traj = Observable(CircularBuffer{Point{3,Float32}}(maxpoints))

# create initial trajectory
for i = 1:maxpoints[]
    progress_for_one_step!(dynsys, traj)
end

lines!(trajax, traj, colormap = :devon, color = @lift 1:$maxpoints)
limits!(trajax, 0, 2, 0, 1, -70, 40)

# voltage trace
trace = @lift [e[3] for e in $traj]
lines!(traceax, trace)

##Interactivity

isrunning = Observable(true)
delay = @lift 1/10^$(speedslider.sliders[1].value)

function run_traj()
    @async while isrunning[]
        isopen(fig.scene) || break # ensures computations stop if closed window
        progress_for_one_step!(dynsys, traj)
        sleep(delay[]) # or `yield()` instead
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
