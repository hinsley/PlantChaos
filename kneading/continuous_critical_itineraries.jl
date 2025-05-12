# Compute the critical itineraries for the SiN model assuming Γ_SD^- and Γ_SD^+
# have the same image under the return map to the reinsertion loop and the
# return map is continuous (we are to the right of the homSF curve).

using Pkg
Pkg.activate("./kneading")
Pkg.instantiate()

using GLMakie, OrdinaryDiffEq, StaticArrays, Roots, NonlinearSolve
using Interpolations, ForwardDiff, LinearAlgebra, DynamicalSystems
using ProgressMeter, JLD2

include("../model/Plant.jl")
using .Plant
include("../tools/equilibria.jl")
include("../tools/symbolics.jl")
include("MultimodalMaps/kneading/power_series.jl")
include("MultimodalMaps/kneading/smallest_root.jl")
include("../map_vis/return_map_utils.jl")

# Define the parameter values to sweep over.
sweep_resolution = 1200
Δxs = range(-1.0, -1.0, length=sweep_resolution)
ΔCas = range(-36.0, -22.0, length=sweep_resolution)

Δx = Δxs[1]
ΔCa = ΔCas[1]
# TODO: Remove.
# ΔCa = -34.44703919933278
# ΔCa = ΔCas[204]

base_params = Plant.default_params[1:15]
p_svector = SVector{17, Float64}([base_params..., Δx, ΔCa])

p = Observable(p_svector)
tspan = (0.0, 1e5) # Default upper bound on time span.

function f(p, Ca0, x0, x_eq_SF)
  # Compute the return map at an initial calcium value Ca0.
  function _condition(u, t, integrator)
    if t < 1e3 || u[1] > x_eq_SF
      return 1.0
    end
    dCa = Plant.melibeNew(u, p, t)[5]
    return dCa
  end
  function _affect!(integrator)
    terminate!(integrator)
  end
  _u0 = Equilibria.dune(p, x0, Ca0)
  _cb = ContinuousCallback(_condition, _affect!)
  _prob = ODEProblem(Plant.melibeNew!, _u0, tspan, p)
  _sol = solve(
    _prob,
    Tsit5(),
    callback=_cb,
    abstol=1e-8,
    reltol=1e-8,
    save_everystep=false
  )
  return _sol.u[end][6]
end

function initialize_T_Ca0(p, plot_fig = false)
  # Compute the equilibria of the slow subsystem.
  V_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))

  # Compute the location of the saddle-focus equilibrium (SF).
  V_eq_SF = V_eqs[2]
  Ca_eq_SF = Equilibria.Ca_null_Ca(p, V_eq_SF)
  x_eq_SF = Plant.xinf(p, V_eq_SF)
  n_eq_SF = Plant.ninf(V_eq_SF)
  h_eq_SF = Plant.hinf(V_eq_SF)
  SF_eq = @SVector [x_eq_SF, 0.0, n_eq_SF, h_eq_SF, Ca_eq_SF, V_eq_SF]

  # Compute the location of the upper saddle equilibrium SD.
  V_eq_SD = V_eqs[3]
  Ca_eq_SD = Equilibria.Ca_null_Ca(p, V_eq_SD)
  x_eq_SD = Plant.xinf(p, V_eq_SD)
  n_eq_SD = Plant.ninf(V_eq_SD)
  h_eq_SD = Plant.hinf(V_eq_SD)
  SD_eq = @SVector [x_eq_SD, 0.0, n_eq_SD, h_eq_SD, Ca_eq_SD, V_eq_SD]

  # Obtain an initial condition for Γ_SD^-.
  jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), SD_eq)
  vals,vecs = eigen(jac)
  _,i = findmax(real.(vals))
  eps = .001
  Γ_SD_minus0 = SVector{6}(SD_eq .- eps .* real.(vecs)[:,i])

  # Condition for the callback: du[5] (Ca derivative) crossing zero from negative to positive.
  function condition(u, t, integrator)
    if t < 1e3 || u[1] > x_eq_SF
      return 1.0
    end
    dCa = Plant.melibeNew(u, integrator.p, integrator.t)[5]
    return dCa
  end

  function affect!(integrator)
    terminate!(integrator)
  end

  cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)

  # Set up and solve the ODE problem.
  prob = ODEProblem(Plant.melibeNew, Γ_SD_minus0, tspan, p)
  sol = solve(prob, Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

  # Store only the endpoint at the calcium minimum.
  Γ_SD_minus_Ca_min = sol.u[end][5]

  # Obtain the voltage value at the calcium minimum on the calcium nullcline.
  function Ca_x_eq(p)
      V_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
      Ca_eq = Ca_null_Ca(p, V_eq)
      x_eq = Plant.xinf(p, V_eq)
      return V_eq, Ca_eq, x_eq
  end
  function Ca_null_Ca(p, V)
      return p[13]*Plant.xinf(p, V)*(p[12]-V+p[17])
  end
  Γ_SD_minus_Ca_min_V = sol.u[end][6] # Initial guess.
  Γ_SD_minus_Ca_min_V = find_zero(V -> Ca_null_Ca(p, V) - Γ_SD_minus_Ca_min, Γ_SD_minus_Ca_min_V)

  # Generate a range of initial conditions along the Ca nullcline between
  # SF and Γ_SD_minus_Ca_min.
  map_resolution = 500
  Vs = range(V_eq_SF, Γ_SD_minus_Ca_min_V, length=map_resolution)
  x_offset = 1f-4 # Offset from xinf to avoid numerical issues.
  u0s = [
    SVector{6, Float64}([
      Plant.xinf(p, V)-x_offset,
      SF_eq[2:4]...,
      Ca_null_Ca(p, V),
      V])
    for V in Vs
  ]
  Ca0s = [u0[5] for u0 in u0s]

  # Calculate the return map.
  template_prob = ODEProblem(Plant.melibeNew, u0s[1], tspan, p)
  function prob_func(prob, i, repeat)
      remake(prob, u0 = u0s[i])
  end
  ensemble_prob = EnsembleProblem(template_prob, prob_func = prob_func)
  ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = length(u0s), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

  return_Ca_mins = [s.u[end][5] for s in ensemble_sol]

  if plot_fig
    # Plot the return map as a scatter plot of Ca0 vs return Ca min.
    fig = Figure(size=(800, 1200)) # Adjusted figure size for two plots.
    ax_return_map = Axis(fig[1, 1], title="Return Map", xlabel="Initial Ca (Ca₀)", ylabel="Return Ca at Minimum", aspect=DataAspect())
    scatter!(ax_return_map, Ca0s, return_Ca_mins, markersize = 4)

    # Add identity line
    all_ca_values = vcat(Ca0s, return_Ca_mins)
    min_ca, max_ca = extrema(all_ca_values)
    lines!(ax_return_map, [min_ca, max_ca], [min_ca, max_ca], color=:gray, linestyle=:dash)

    # Plot all trajectories from the ensemble solve.
    ax_trajectories = Axis(fig[2, 1], title="Trajectories (Ca vs x)", xlabel="Ca", ylabel="x")
    for s in ensemble_sol
        # Extract Ca (index 5) and x (index 1) for each point in the trajectory.
        x_vals = [pt[1] for pt in s.u]
        ca_vals = [pt[5] for pt in s.u]
        lines!(ax_trajectories, ca_vals, x_vals) # Makie will cycle colors.
    end

    # Add the trajectory for Γ_SD_minus0 (solution in sol) in red.
    if @isdefined(sol) && !isempty(sol.u)
        x_vals_gamma_sd_minus = [pt[1] for pt in sol.u]
        ca_vals_gamma_sd_minus = [pt[5] for pt in sol.u]
        lines!(ax_trajectories, ca_vals_gamma_sd_minus, x_vals_gamma_sd_minus, color = :red, linewidth = 2, linestyle = :dot, label = L"$\Gamma_{\text{SD}}^-$ trajectory")
    else
        println("Warning: `sol` for Γ_SD_minus0 is not defined or empty, cannot plot its trajectory.")
    end
  end

  # Obtain the first guess at the calcium value for the critical point
  # associated with the 1-spike preimage of T.

  # Find the first local maximum in the return map.
  function find_first_local_maximum(x, y)
      for i in 2:(length(y)-1)
          if y[i] > y[i-1] && y[i] > y[i+1]
              return i
          end
      end
      return nothing  # Return nothing if no local maximum is found.
  end

  # Get the index of the first local maximum in return_Ca_mins.
  first_max_index = find_first_local_maximum(Ca0s, return_Ca_mins)

  if plot_fig
    if first_max_index !== nothing
        println("T found at index: ", first_max_index)
        println("Corresponding Ca₀ value: ", Ca0s[first_max_index])
        println("Return Ca value: ", return_Ca_mins[first_max_index])
        
        # Mark the first local maximum on the return map plot.
        scatter!(ax_return_map, [Ca0s[first_max_index]], [return_Ca_mins[first_max_index]], 
                color = :red, markersize = 8, marker = :star5)
        
        # Mark the initial condition in the trajectories plot.
        scatter!(ax_trajectories, [Ca0s[first_max_index]], [u0s[first_max_index][1]], 
                color = :red, markersize = 8, marker = :star5)
        
        # Plot the trajectory associated with the maximum in extra large sized lines.
        max_trajectory = ensemble_sol[first_max_index]
        x_vals_max = [pt[1] for pt in max_trajectory.u]
        ca_vals_max = [pt[5] for pt in max_trajectory.u]
        lines!(ax_trajectories, ca_vals_max, x_vals_max, 
              color = :red, linewidth = 4, linestyle = :solid, 
              label = "1-spike preimage of T trajectory")
        axislegend(ax_trajectories, position = (:right, :bottom))
    else
        println("No local maximum found in the return map.")
    end

    display(fig)
  end

  # Compute the itinerary of the critical point associated with the 1-spike
  # preimage of T.

  # Find T_Ca0 using golden section search.
  T_Ca0_guess = Ca0s[first_max_index]
  u0 = u0s[first_max_index]
  search_radius = 3e-3
  a = T_Ca0_guess - search_radius
  b = T_Ca0_guess + search_radius
  golden_ratio = (sqrt(5) - 1) / 2
  c = b - golden_ratio * (b - a)
  d = a + golden_ratio * (b - a)
  tol = 1e-10

  # Golden section search algorithm to find maximum of f.
  while abs(b - a) > tol
      fc = f(p, c, u0[1], x_eq_SF)
      fd = f(p, d, u0[1], x_eq_SF)
      if fc > fd
          b = d
      else
          a = c
      end
      c = b - golden_ratio * (b - a)
      d = a + golden_ratio * (b - a)
  end

  T_Ca0 = (a + b) / 2

  if plot_fig
    # Run the trajectory associated with the refined critical point.
    __prob = remake(prob, u0 = SVector{6}([
      u0[1:4]...,
      T_Ca0,
      u0[6]
    ]))
    __sol = solve(__prob, Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8, save_everystep=true)

    # Plot the critical point on the return map.
    # scatter!(ax_return_map, [T_Ca0], [return_Ca_mins[first_max_index]], 
    #          color = :blue, markersize = 8, marker = :star5)
    scatter!(ax_return_map, [__sol.u[1][5]], [__sol.u[end][5]], 
            color = :blue, markersize = 8, marker = :star5)

    # Plot the trajectory in the trajectories plot.
    x_vals_critical = [pt[1] for pt in __sol.u]
    ca_vals_critical = [pt[5] for pt in __sol.u]
    lines!(ax_trajectories, ca_vals_critical, x_vals_critical, 
            color = :blue, linewidth = 4, linestyle = :solid, 
            label = "Critical trajectory")

    display(fig)
  end

  T0 = SVector{6}([u0[1:4]..., T_Ca0, u0[6]])
  return T0
end

T0 = initialize_T_Ca0(p[], true)
T_Ca0 = T0[5]

# Define Event Symbols for SSCS computation.
@enum EventSymbol begin
    Void # Nothing detected yet.
    I # Vdot maximum.
    Vplus # Spike (V max above V_sd).
    Vminus # Slow subsystem oscillation (V max below V_sd).
end

# Constants for SSCS and htop computation (from inline_scan.jl).
const TRANSIENT_TIME = 1e3 # Time to wait before beginning to detect events.
const MAX_SEQ_LENGTH = 100 # Maximum length of signed spike counts before terminating trajectory integration.
const MAX_SPIKE_COUNT = 35 # Maximum number of spikes in a single burst before considering the trajectory tonic-spiking and terminating.
const SSCS_ODE_TSPAN = (0.0, 1e8) # Timespan for SSCS ODE solves.
const KNEADING_DETERMINANT_PRUNE_WINDOW = 30 # Mitigates rootfinding problems.
const MAX_PERIODICITY_PRUNE = 4 # Mitigates rootfinding problems.

# Iterate over the parameter values in the specified sweep range.
T_SSCSs = Vector{Int}[] # For debugging.
Gamma_SD_minus_SSCSs = Vector{Int}[] # For debugging.
kneading_determinants = Vector{Int}[] # For debugging.
lz_complexity_values = Float64[]
htop_values = Float64[]
i = 1
while i <= sweep_resolution
  println("Sweeping... [$(i)/$(sweep_resolution)]")

  # Update the parameter vector.
  p = Observable(SVector{17, Float64}([base_params..., Δxs[i], ΔCas[i]]))

  # Placeholder value for the orientation determining max voltage threshold.
  # V_return_threshold = -Inf
  V_return_threshold = -46.0 # Seems (from inspection) to work in this sweep.

  # Compute the equilibria of the slow subsystem.
  V_eqs = find_zeros(v -> Equilibria.Ca_difference(p[], v), Plant.xinfinv(p[], 0.99e0), Plant.xinfinv(p[], 0.01e0))

  # Compute the location of the saddle-focus equilibrium (SF).
  V_eq_SF = V_eqs[2]
  Ca_eq_SF = Equilibria.Ca_null_Ca(p[], V_eq_SF)
  x_eq_SF = Plant.xinf(p[], V_eq_SF)
  n_eq_SF = Plant.ninf(V_eq_SF)
  h_eq_SF = Plant.hinf(V_eq_SF)
  SF_eq = @SVector [x_eq_SF, 0.0, n_eq_SF, h_eq_SF, Ca_eq_SF, V_eq_SF]

  # Compute the location of the upper saddle equilibrium SD.
  V_eq_SD = V_eqs[3]
  Ca_eq_SD = Equilibria.Ca_null_Ca(p[], V_eq_SD)
  x_eq_SD = Plant.xinf(p[], V_eq_SD)
  n_eq_SD = Plant.ninf(V_eq_SD)
  h_eq_SD = Plant.hinf(V_eq_SD)
  SD_eq = @SVector [x_eq_SD, 0.0, n_eq_SD, h_eq_SD, Ca_eq_SD, V_eq_SD]

  # Compute the initial condition for Γ_SD^-.
  jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p[],0), SD_eq)
  vals,vecs = eigen(jac)
  _,idx_eigen = findmax(real.(vals)) # Renamed i to idx_eigen to avoid conflict.
  eps = .001
  Γ_SD_minus0 = SVector{6}(SD_eq .- eps .* real.(vecs)[:,idx_eigen])

  # Find T_Ca0 using golden section search.
  T_Ca0_guess = T_Ca0#Ca0s[first_max_index] # Why was I using this?
  search_radius = 3e-3
  a = T_Ca0_guess - search_radius
  b = T_Ca0_guess + search_radius
  golden_ratio = (sqrt(5) - 1) / 2
  c_gs = b - golden_ratio * (b - a) # Renamed c to c_gs to avoid conflict.
  d_gs = a + golden_ratio * (b - a) # Renamed d to d_gs to avoid conflict.
  tol = 1e-10

  # Golden section search algorithm to find maximum of f.
  while abs(b - a) > tol
    fc = f(p[], c_gs, T0[1], x_eq_SF)
    fd = f(p[], d_gs, T0[1], x_eq_SF)
    if fc > fd
        b = d_gs
    else
        a = c_gs
    end
    c_gs = b - golden_ratio * (b - a)
    d_gs = a + golden_ratio * (b - a)
  end

  T_Ca0 = (a + b) / 2

  # Construct the initial conditions for T.
  T0 = SVector{6}([T0[1:4]..., T_Ca0, T0[6]])

  # Current V_sd for SSCS state machines.
  V_sd_current = V_eq_SD

  # Initialize state machines for SSCS computation.
  state_machine_T = Dict{Symbol, Any}(
      :scs => Int[],
      :count => 0,
      :last_symbol => Void,
      :last2_symbol => Void,
      :V_sd => V_sd_current
  )
  state_machine_Gamma_SD_minus0 = Dict{Symbol, Any}(
      :scs => Int[],
      :count => 0,
      :last_symbol => Void,
      :last2_symbol => Void,
      :V_sd => V_sd_current
  )

  # Define the condition function for SSCS callback.
  function condition_sscs(out, u, t, integrator)
    if t < TRANSIENT_TIME
      # Condition should be non-zero if no event, zero if event.
      # To prevent triggering before TRANSIENT_TIME, set to a non-zero value.
      out[1] = 1.0 
      out[2] = 1.0
      return
    end
    
    current_p_val = integrator.p
    # Vdot calculation from single_trajectory_fig.jl
    # The `u` in the callback is the SVector [x, 0.0, n, h, Ca, V].
    # Plant.dV expects arguments p, x, y, n, h, Ca, V.
    # u[2] is y, which is 0.0 in this model's state vector for melibeNew.
    Vdot_val = Plant.dV(current_p_val, u[1], u[2], u[3], u[4], u[5], u[6])
    out[1] = -Vdot_val # Condition from single_trajectory_fig.jl for Vdot == 0.

    # Vddot calculation using Plant.numerical_derivative as in single_trajectory_fig.jl.
    Vddot_val = Plant.numerical_derivative(
        (params_nd, h_nd, hdot_nd, n_nd, ndot_nd, x_nd, xdot_nd, Ca_nd, Cadot_nd, V_nd, Vdot_selector) -> Vdot_selector,
        u, # current state vector [x,0,n,h,Ca,V]
        current_p_val, # parameters
        1e-4 # dt for numerical differentiation
    )
    out[2] = -Vddot_val # Condition from single_trajectory_fig.jl for Vddot == 0.
  end

  # Factory for the affect! function for SSCS callback.
  function make_affect_sscs!(state_machine)
    function affect_sscs!(integrator, idx)
      if idx == 1 # V extremum (Vdot == 0), this is where V+ or V- is determined for Algorithm 1.
        current_V = integrator.u[6]
        current_algorithmic_event = (current_V > state_machine[:V_sd]) ? Vplus : Vminus

        if current_algorithmic_event == Vminus
          # If the first spike count is not 1 for T, fall back to the less
          # accurate method, which does not use a burst return voltage
          # threshold.
          if V_return_threshold == -Inf && state_machine[:count] == 1
            V_return_threshold = current_V
          end
          if current_V > V_return_threshold || state_machine[:last2_symbol] != Vplus
            push!(state_machine[:scs], state_machine[:count])
          else
            push!(state_machine[:scs], -state_machine[:count])
          end
          state_machine[:count] = 0 # Spikes <- 0.
        else
          state_machine[:count] += 1 # Spikes <- Spikes + 1.
        end

        state_machine[:last2_symbol] = state_machine[:last_symbol]
        state_machine[:last_symbol] = current_algorithmic_event
        
        if length(state_machine[:scs]) >= MAX_SEQ_LENGTH
          terminate!(integrator)
        end
        if current_algorithmic_event == Vplus && state_machine[:count] > MAX_SPIKE_COUNT
            terminate!(integrator)
        end
      elseif idx == 2
        state_machine[:last2_symbol] = state_machine[:last_symbol]
        state_machine[:last_symbol] = I
      end
    end
    return affect_sscs!
  end

  affect_for_T! = make_affect_sscs!(state_machine_T)
  affect_for_Gamma! = make_affect_sscs!(state_machine_Gamma_SD_minus0)

  # Compute SSCS for T.
  # Note: Because we're not using I, V-, V+ events, but instead a threshold
  # for V on the return to the dune, the SSCS for T has to be computed before
  # the SSCS for any other trajectory so that the signs of SSCS can be
  # properly determined.
  # println("Computing SSCS for T...")
  cb_T_sscs = VectorContinuousCallback(condition_sscs, affect_for_T!, nothing, 2, save_positions=(false,false))
  prob_T_sscs = ODEProblem(Plant.melibeNew, T0, SSCS_ODE_TSPAN, p[], callback=cb_T_sscs)
  sol_T_sscs = solve(prob_T_sscs, Tsit5(), abstol=3e-6, reltol=3e-6, save_everystep=false)
  T_scs = state_machine_T[:scs]
  println("SSCS for T: ", T_scs)
  if abs(T_scs[1]) != 1 # If T's SSCS doesn't start with +/-1, T_Ca0 is wrong.
    println("Reinitializing T_Ca0...")
    i -= 1
    T0 = initialize_T_Ca0(p[])
    T_Ca0 = T0[5]
    continue
  end
  push!(T_SSCSs, T_scs)
  
  # Compute SSCS for Γ_SD_minus0.
  # println("Computing SSCS for Γ_SD_minus0...")
  cb_Gamma_sscs = VectorContinuousCallback(condition_sscs, affect_for_Gamma!, nothing, 2, save_positions=(false,false))
  prob_Gamma_sscs = ODEProblem(Plant.melibeNew, Γ_SD_minus0, SSCS_ODE_TSPAN, p[], callback=cb_Gamma_sscs)
  sol_Gamma_sscs = solve(prob_Gamma_sscs, Tsit5(), abstol=1e-8, reltol=1e-8, save_everystep=false)
  Gamma_SD_minus0_scs = state_machine_Gamma_SD_minus0[:scs]
  println("SSCS for Γ_SD_minus0: ", Gamma_SD_minus0_scs)
  push!(Gamma_SD_minus_SSCSs, Gamma_SD_minus0_scs)

  # Compute the LZ complexity of the SSCS for Γ_SD^-.
  @time LZ_complexity = normalized_LZ76_complexity(Gamma_SD_minus0_scs)
  println("LZ complexity of Γ_SD_minus: ", LZ_complexity)

  # Compute the kneading sequences of each critical trajectory.
  T_kneading_sequence = itinerary_to_kneading_sequence(
    SSCS_to_itinerary(T_scs[2:end])
  )
  # println("Kneading sequence of T: ", T_kneading_sequence)
  Gamma_SD_minus_kneading_sequence = itinerary_to_kneading_sequence(
    SSCS_to_itinerary(Gamma_SD_minus0_scs)
  )
  # println("Kneading sequence of Γ_SD_minus: ", Gamma_SD_minus_kneading_sequence)

  # Compute the kneading determinant using the matrix determinant lemma trick
  # for saddled Swiss-roll attractors.
  # ℓ = Gamma_SD_minus_kneading_sequence[1] # Top lap index in the core.
  # Patch. Sometimes the first symbol is wrong.
  ℓ = maximum([Gamma_SD_minus_kneading_sequence..., T_kneading_sequence...])
  
  # Get the shorter length among the two kneading sequences.
  K = min(
    length(Gamma_SD_minus_kneading_sequence),
    length(T_kneading_sequence)
  )

  is_periodic(xs) = findfirst( # Used to mitigate rootfinding problems.
    x -> all(xs .== xs[mod1.(1:length(xs), x)]),
    1:min(MAX_PERIODICITY_PRUNE, length(xs))
  )
  htop = 0.0
  T_max_lap = maximum(T_kneading_sequence)
  @time if T_max_lap > ℓ
    println("ℓ incorrectly computed: Is $(ℓ) and should be at least $(T_max_lap). Setting htop = 0.0.")
    push!(kneading_determinants, zeros(Integer, K))
    htop = 0.0
  elseif !isnothing(is_periodic(T_kneading_sequence[
    end-KNEADING_DETERMINANT_PRUNE_WINDOW+1:end
  ])) && !isnothing(is_periodic(Gamma_SD_minus_kneading_sequence[
    end-KNEADING_DETERMINANT_PRUNE_WINDOW+1:end
  ]))
    println("Both sequences have periodic tails. Setting htop = 0.0.")
    push!(kneading_determinants, zeros(Integer, K))
    htop = 0.0
  elseif ℓ == 2
    # Compute topological entropy using the formula for a unimodal map.

    kneading_matrix = zeros(Integer, K)
    kneading_matrix[1] = 1
    sign1 = -1
    for k in 2:K
      lap1 = Gamma_SD_minus_kneading_sequence[k-1]
      if lap1 == 2
        kneading_matrix[k] = 2 * sign1
        sign1 = -sign1
      end
    end

    push!(kneading_determinants, det)

    # Compute the smallest root of the kneading determinant.
    r = 0.0
    try
      r = smallest_root(kneading_matrix)
    catch e
      if isa(e, BoundsError)
        println("Smallest root is likely magnitude 1, setting to 1.0.")
        r = 1.0  # Default value if computation fails.
      else
        rethrow(e)  # Re-throw other types of errors.
      end
    end
    # println("Smallest root of the kneading determinant: ", r)
    htop = max(0.0, -log(r))
  elseif ℓ > 2
    # Perform the multimodal kneading determinant computation, using the matrix
    # determinant lemma trick for saddled Swiss-roll attractors.

    # Allocate a 3D matrix for shorthand kneading matrix computation.
    kneading_matrix = zeros(Integer, 2, ℓ-1, K)

    # Prepopulate the constant terms with known values.
    kneading_matrix[1, 1, 1] = 1
    kneading_matrix[2, 1, 1] = -1
    kneading_matrix[2, 2, 1] = 1

    # Compute the rest of the kneading matrix.
    sign1 = -1
    sign2 = 1
    for k in 2:K
      lap1 = Gamma_SD_minus_kneading_sequence[k-1]
      if lap1 > 1
        kneading_matrix[
          1,
          lap1 - 1,
          k
        ] = 2 * sign1
        if iseven(lap1)
          sign1 = -sign1
        end
      end
      lap2 = T_kneading_sequence[k]
      if lap2 > 1
        kneading_matrix[
          2,
          lap2 - 1,
          k
        ] = 2 * sign2
        if iseven(lap2)
          sign2 = -sign2
        end
      end
    end

    # println("Kneading matrix: ", kneading_matrix)

    # Helper function for computing the coefficients of the kneading determinant.
    function coeff(j, k)
      if isodd(j)
        if isodd(k)
          return 0
        else
          return (1-j)/2
        end
      else
        if isodd(k)
          return (k-1)/2
        else
          return (k-j)/2
        end
      end
    end

    # Compute the kneading determinant.
    det = Integer[]
    for j in 2:ℓ
      for k in j+1:ℓ
        factor1 = kneading_matrix[1, j-1, :]
        factor2 = kneading_matrix[2, k-1, :]
        factor3 = kneading_matrix[1, k-1, :]
        factor4 = kneading_matrix[2, j-1, :]
        result = scale(
          coeff(j, k),
          add(
            multiply(factor1, factor2),
            scale(-1, multiply(factor3, factor4))
          )
        )
        det = add(det, result)
      end
    end

    # Truncate the kneading determinant and convert to Int64.
    det = convert(Vector{Int64}, det[1:K])
    push!(kneading_determinants, det)
    # println("Kneading determinant: ", det)

    # Compute the smallest root of the kneading determinant.
    r = 0.0
    try
      r = smallest_root(det)
    catch e
      if isa(e, BoundsError)
        println("Smallest root is likely magnitude 1, setting to 1.0.")
        r = 1.0  # Default value if computation fails.
      else
        rethrow(e)  # Re-throw other types of errors.
      end
    end
    # println("Smallest root of the kneading determinant: ", r)

    # Compute the topological entropy of the system.
    htop = max(0.0, -log(r))
  end
  println("Topological entropy: ", htop)

  push!(lz_complexity_values, LZ_complexity)
  push!(htop_values, htop)
  i += 1
end

# Compute the LLE of some trajectory for all points in the parameter sweep.
# Pre-allocate lle_values outside the loop.
lle_values = Vector{Float64}(undef, sweep_resolution)
T = 1e6
Ttr = 1e5
d0 = 1e-6
Δt = 1e-1
progress_bar = Progress(sweep_resolution, 1, "Computing LLE: ", 50) # Progress bar can be outside.
Threads.@threads for i in 1:sweep_resolution
  local_p_svector = SVector{17, Float64}([base_params..., Δxs[i], ΔCas[i]])
  
  # Just reuse the old T0, it doesn't matter too much what the initial condition
  # is, so long as there aren't multiple attracting sets.
  system = CoupledODEs(Plant.melibeNew, T0, local_p_svector)
  LLE_val = lyapunov(
    system, T;
    Ttr = Ttr,
    d0 = d0,
    Δt = Δt
  )
  lle_values[i] = LLE_val
  next!(progress_bar)
end

# Save the computed values to a JLD2 file for later use.
@save "./kneading/lz_htop_lle_data.jld2" lz_complexity_values htop_values lle_values ΔCas
println("Data saved to lz_htop_lle_data.jld2.")

# Save the SSCSs to a JLD2 file for later use.
@save "./kneading/sscs_data.jld2" T_SSCSs Gamma_SD_minus_SSCSs
println("SSCSs saved to sscs_data.jld2.")

# Function to load the saved data if needed.
function load_saved_data(filename="./kneading/lz_htop_lle_data.jld2")
    data = load(filename)
    return data["lz_complexity_values"], data["htop_values"], data["lle_values"], data["ΔCas"]
end

# Function to load the saved SSCS data if needed.
function load_saved_sscs_data(filename="./kneading/sscs_data.jld2")
    data = load(filename)
    return data["T_SSCSs"], data["Gamma_SD_minus_SSCSs"]
end

# Load the saved values.
# lz_complexity_values, htop_values, lle_values, ΔCas = load_saved_data()
# T_SSCSs, Gamma_SD_minus_SSCSs = load_saved_sscs_data()

# Plot the results.
fig = Figure(size=(1200, 1200))
ax_htop = Axis(fig[1, 1], title=L"\text{Topological Entropy}", 
    titlesize=24,
    ylabel=L"h_{top}", 
    ylabelsize=24)
ax_lz_complexity = Axis(fig[2, 1], title=L"\text{Lempel-Ziv Complexity (\Gamma_{\text{SD}}^-)}", 
    titlesize=24,
    ylabel=L"\text{LZ76}", 
    ylabelsize=24)
ax_lle = Axis(fig[3, 1], title=L"\text{Leading Lyapunov Exponent}", 
    titlesize=24,
    xlabel=L"$\Delta$Ca", ylabel=L"\text{LLE}", 
    xlabelsize=24, ylabelsize=24)

# Link the horizontal axes together.
linkxaxes!(ax_htop, ax_lz_complexity, ax_lle)

lines!(ax_htop, ΔCas, htop_values)
lines!(ax_lz_complexity, ΔCas, lz_complexity_values)
lines!(ax_lle, ΔCas, lle_values)
display(fig)

# Save the figure to a file.
save("lz_htop_lle.png", fig)