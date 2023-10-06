using BandedMatrices
function moving_avg(r, warr)
    cols = length(r)
    rows = cols+1

    for w in warr
        rows -= w
        r = BandedMatrix(Ones(rows, cols), (0,w-1)) * r ./ w
        cols = rows
    end
    return r
end

function total

r = sin.(cumsum(rand(4000))./2000*20*pi) #.+ rand(1000)/10
warr = [2100]#fill(25, 20)
a = moving_avg(r, warr)
r2 = vcat(zeros(sum(warr)-1), a)

using GLMakie
begin 
    fig = Figure()
    ax = Axis(fig[1,1])
    lines!(ax, r, color = :blue)
    lines!(ax, r2, color = :red)
    fig
end

moving_avg(r, warr)