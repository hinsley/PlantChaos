
begin
    s = GLMakie.Screen()
    i = Observable[1]
    a = 1000 + 1
    x = c_cass[][1:a:end] |> Observable
    y = c_xss[][1:a:end] |> Observable
    z = c_vss[][1:a:end] |> Observable
    x2 = x[]
    y2 = y[]
    z2 = z[]
    _fig = Figure();
    ax = Axis3(_fig[1,1])
    lines!(ax,x2,y2,z2)
    lines!(ax,x,y,z)
    display(s,_fig)

    framerate = 15
    is = 1:1000

    record(_fig, "time_animation.mp4", is;
            framerate = framerate) do i
        x[] = c_cass[][i:a:end]
        y[] = c_xss[][i:a:end]
        z[] = c_vss[][i:a:end]
        reset_limits!(ax)
    end
end
