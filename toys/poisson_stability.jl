module PoissonStability export near_eq, recurrence_count, max_linger_time

include("../tools/solve.jl")
include("../tools/equilibria.jl")

using LinearAlgebra

# Recurrence #
# Residence/linger time

default_delta = 1e-2

function near_eq(x, eq; delta=default_delta)
    # Determine whether a point is in a neighborhood of an equilibrium.
    return norm(x - eq) < delta
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

end
