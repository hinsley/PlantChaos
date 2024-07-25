using JLD2

ca_cut = 150
x_cut = 130
larr = load("lyapunov_data.jld2")["lyaparray"][1:end-x_cut,ca_cut:end]
l2arr = load("lyapunov_data.jld2")["lyap2array"][1:end-x_cut,ca_cut:end]
scale(x, min, max) = x < min ? 0.0 : x > max ? 1.0 : (x - min) / (max - min) 

lcarr = [Makie.RGB(
    let a = larr'[i,j]
        abs2(scale(a, 0, 0.00021))
    end, let a = l2arr'[i,j]
        sqrt(scale(-a, 0.0000, 0.0015))
    end, let a = l2arr'[i,j]
        sqrt(sqrt(scale(-a, .00, 0.00019)))
    end
) for i in 1:resolution-ca_cut, j in 1:resolution-x_cut] .+ Makie.RGB(.4,.4,.4)


start_p = [Plant.default_params...]
start_p[17] = -60.0 # Cashift
start_p[16] = -2 # xshift
end_p = [Plant.default_params...]
end_p[17] = 0.0 # Cashift
end_p[16] = .5 # xshift

resolution = 700 # How many points to sample.
Ca_shifts = LinRange(start_p[17] + ca_cut/resolution*(end_p[17]-start_p[17]), end_p[17], resolution-ca_cut)
x_shifts = LinRange(start_p[16], end_p[16]-x_cut/resolution*(end_p[16]-start_p[16]), resolution-x_cut)

#hand drawn curves
upper = [
    (-1.0649282932281494, -1.8542914390563965),
    (-8.015800476074219, -1.6503283977508545),
    (-13.98834228515625, -1.4550861120224),
    (-19.41448402404785, -1.269153118133545),
    (-24.416770935058594, -1.0934346914291382),
    (-28.866989135742188, -0.9138347506523132),
    (-32.86981964111328, -0.7407776713371277),
    (-36.63554000854492, -0.561443030834198),
    (-40.109222412109375, -0.3759620189666748),
    (-43.3690185546875, -0.19186238944530487),
    (-46.7093620300293, 0.03453662991523743),
]
lower = [
    (-0.02165699191391468, -1.961098551750183),
    (-4.6258769035339355, -1.852035641670227),
    (-10.666454315185547, -1.6712626218795776),
    (-16.223255157470703, -1.5073782205581665),
    (-21.097087860107422, -1.3465032577514648),
    (-25.43739891052246, -1.1946871280670166),
    (-29.362749099731445, -1.0425105094909668),
    (-32.95021057128906, -0.8935339450836182),
    (-36.31550598144531, -0.7223095893859863),
    (-39.25956726074219, -0.5924959778785706),
    (-41.65858459472656, -0.5417609810829163)
]

p1 = vcat(p[][1:15], (-1.0866918563842773, -15.22952651977539)...)
p2 = vcat(p[][1:15], (-1.565005898475647, -17.90139389038086)...)

begin
    try close(sc4) 
    catch
        nothing
    end
    global sc4 = GLMakie.Screen(;resize_to = (500, 500))
    set_theme!(Theme(
        Axis = (
            xticklabelsize = 14,
            yticklabelsize = 14,
            xlabelsize = 18,
            ylabelsize = 18,
            yticks = WilkinsonTicks(3),
            xticks = WilkinsonTicks(3),
        )
    ))
    fig = Figure()
    # bifurcation diagram
    bax = Axis(fig[1,1], xlabel = "ΔCa", ylabel = "Δx")
    image!(bax, Ca_shifts, x_shifts, lcarr)
    lines!(bax, upper, color = :black, linewidth = 2, linestyle = :dot)
    lines!(bax, lower, color = :black, linewidth = 2)
    scatter!(bax, upper, color = :black, markersize = 25, marker = '⋆')
    scatter!(bax, lower, color = :black, markersize = 35, marker = '▿')

    sc = 1.2
    ax1 = Axis(fig[1,1],
    width=Relative(0.2*sc),
    height=Relative(0.2*sc),
    halign=0.55,
    valign=0.75,
    aspect = DataAspect(),
    xlabel = L"V_{n}",
    ylabel = L"V_{n+1}")
    hidedecorations!(ax1, label = false)

    ax2 = Axis(fig[1,1],
    width=Relative(0.2*sc),
    height=Relative(0.2*sc),
    halign=0.35,
    valign=0.1,
    aspect = DataAspect(),
    xlabel = L"V_{n}",
    ylabel = L"V_{n+1}")
    hidedecorations!(ax2, label = false)

    ax3 = Axis(fig[1,1],
    width=Relative(0.2*sc),
    height=Relative(0.2*sc),
    halign=0.95,
    valign=0.55)
    hidedecorations!(ax3, label = false)
    x = range(-1, 1.5, length = 500)


    img = fill(Makie.RGBA(1,1,1,.5), 10,10)
    image!(ax3,x,x, img)

    f(x) = -8x^4 + x + 4/3*x^3 + 4x^2
    y = f.(x)
    g = (1 + sqrt(65))/16
    h = (1 - sqrt(65))/16
    y2 = @. x + f(g) - g
    y3 = @. x + f(h) - h
    lines!(ax3, x, y, color = :black)
    lines!(ax3, x, y2, color = :red, linestyle = :dash)
    lines!(ax3, x, x, color = :green, linestyle = :dash)
    lines!(ax3, x, y3, color = :blue, linestyle = :dash)
    scatter!(ax3, (g,f(g)), color = :black, markersize = 10)
    scatter!(ax3, (g,f(g)), color = :red, markersize = 8)
    scatter!(ax3, (h,f(h)), color = :black, markersize = 10)
    scatter!(ax3, (h,f(h)), color = :blue, markersize = 8)
    scatter!(ax3, (0,0), color = :black, markersize = 10)
    scatter!(ax3, (0,0), color = :green, markersize = 8)
    text!(ax3, -.8,-.3, text = L"α", color = :blue, fontsize = 14)
    text!(ax3, .1,-.3, text = L"β", color = :green, fontsize = 14)
    text!(ax3, .2,1.1, text = L"γ", color = :red, fontsize = 14)


    text!(ax3, -.9,1, text = "C", color = :black)

    limits!(ax3, -1, 1.5, -1, 1.5)

    resize!(fig, 500, 500)
    display(sc4, fig)
end

save("spike_adding_envelope.png", fig)