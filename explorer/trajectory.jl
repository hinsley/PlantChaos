function progress_for_one_step!(solver, traj)
    s = solver[]
    step!(s.integ, 10.0, true)
    push!(traj[], Point3f(s.integ.u[[5,1,6]]))
    traj[] = traj[]
end

#set length of stored trajectory with slider
maxpoints = 1000
traj = Observable(CircularBuffer{Point{3,Float32}}(maxpoints))

# create initial trajectory
for i = 1:maxpoints
    progress_for_one_step!(dynsys, traj)
end

lines!(trajax, traj, colormap = :devon, color = @lift 1:$maxpoints)
limits!(trajax, 0, 2, 0, 1, -70, 40)

# voltage trace
trace = @lift [e[3] for e in $traj]
lines!(traceax, trace)

##Interactivity

isrunning = Observable(true)
delay = @lift 1/10^($(speedslider.sliders[1].value)+1)

function run_traj()
    @async while isrunning[]
        isopen(fig.scene) || break # ensures computations stop if closed window
        progress_for_one_step!(dynsys, traj)
        sleep(delay[]) # or `yield()` instead
    end
end

run_traj()

on(pausebutton.clicks) do clicks
    isrunning[] = !isrunning[]
end
on(pausebutton.clicks) do x
    isrunning[] && run_traj()
end

# clear and start integration
on(clearbutton.clicks) do clicks
    empty!(traj[])
    traj[] = traj[]
end

# restart integration from initial conditions to explore transients
on(resetbutton.clicks) do clicks
    empty!(traj[])
    reinit!(dynsys[])
    traj[] = traj[]
end
