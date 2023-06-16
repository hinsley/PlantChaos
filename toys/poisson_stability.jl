module PoissonStability export near_eq, recurrence_count, max_linger_time, min_distance, total_linger_time

include("../tools/solve.jl")
include("../tools/equilibria.jl")

using LinearAlgebra

# Recurrence #
# Residence/linger time

default_delta = 1e-2

function near_eq(x, eq; delta=default_delta)
    # Determine whether a point is in a neighborhood of an equilibrium.
    return sqrt((x[1]-eq[1])^2 + (x[5]-eq[5])^2) < delta # Distance in slow subsystem.
    #return norm(x - eq) < delta # Distance in full system.
end

function recurrence_count(sol, eq; delta=default_delta)
    # Count the number of times the solution visits the neighborhood of an equilibrium.
    count = 0
    was_near_eq = false
    for i in 1:length(sol)
        is_near_eq = near_eq(sol.u[i], eq; delta=delta)
        if is_near_eq && !was_near_eq
            count += 1
        end
        was_near_eq = is_near_eq
    end
    return count
end

function max_linger_time(sol, eq; delta=default_delta)
    # Find the maximum time the solution spends in the neighborhood of an equilibrium.
    longest_linger_time = 0
    linger_interval_start = nothing
    for i in 1:length(sol)
        if near_eq(sol.u[i], eq; delta=delta) && linger_interval_start === nothing
            linger_interval_start = sol.t[i]
        elseif linger_interval_start !== nothing
            longest_linger_time = max(longest_linger_time, sol.t[i] - linger_interval_start)
            linger_interval_start = nothing
        end
    end
    return longest_linger_time
end

function min_distance(sol, eq)
    # Find the minimum distance the solution achieves from an equilibrium.
    min_distance = Inf
    for i in 1:length(sol)
        min_distance = min(min_distance, sqrt((sol.u[i][1]-eq[1])^2 + (sol.u[i][5]-eq[5])^2)) # Distance in slow subsystem.
        #min_distance = min(min_distance, norm(sol.u[i] - eq)) # Distance in full system.
    end
    return min_distance
end

function total_linger_time(sol, eq; delta=default_delta)
    # Find the total time the solution spends in the neighborhood of an equilibrium.
    total_linger_time = 0
    linger_interval_start = nothing
    for i in 1:length(sol)
        if near_eq(sol.u[i], eq; delta=delta) && linger_interval_start === nothing
            linger_interval_start = sol.t[i]
        elseif linger_interval_start !== nothing
            total_linger_time += sol.t[i] - linger_interval_start
            linger_interval_start = nothing
        end
    end
    return total_linger_time
end

end