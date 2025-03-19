# # Half-Center Oscillator
# Symbolic encoding and network coordinate diagram of trajectories of a
# half-center oscillator.

using Pkg
Pkg.activate("./network_symbolics")
Pkg.instantiate()

using GLMakie, OrdinaryDiffEq, StaticArrays

include("../model/Plant.jl")
using .Plant

tspan = (0.0, 1e6)

function two_neurons!(du, u, p, t)
    # Split parameters into two sets
    p1 = p[1:17]  # Parameters for neuron 1
    p2 = p[18:34] # Parameters for neuron 2
    
    # Coupling parameters
    g12 = p[35]   # Weight from neuron 1 to neuron 2
    g21 = p[36]   # Weight from neuron 2 to neuron 1
    
    # Logistic synapse parameters (separate for each direction)
    # 1->2 synapse parameters
    alpha12 = p[37]
    beta12 = p[38]
    S_0_12 = p[39]
    k12 = p[40]
    Theta_syn_12 = p[41]
    E_rev_12 = p[42]
    
    # 2->1 synapse parameters
    alpha21 = p[43]
    beta21 = p[44]
    S_0_21 = p[45]
    k21 = p[46]
    Theta_syn_21 = p[47]
    E_rev_21 = p[48]
    
    # Extract individual neuron states
    # Neuron 1: u[1:7], Neuron 2: u[8:14]
    V1 = u[6]     # Voltage of neuron 1
    V2 = u[13]    # Voltage of neuron 2
    S1 = u[7]     # Synaptic variable for neuron 1
    S2 = u[14]    # Synaptic variable for neuron 2
    
    # Calculate activation function values
    f_inf_V1 = 1.0 / (1.0 + exp(-k12 * (V1 - Theta_syn_12)))
    f_inf_V2 = 1.0 / (1.0 + exp(-k21 * (V2 - Theta_syn_21)))
    
    # Calculate synaptic dynamics
    dS1 = alpha12 * S1 * (1.0 - S1) * f_inf_V1 - beta12 * (S1 - S_0_12)
    dS2 = alpha21 * S2 * (1.0 - S2) * f_inf_V2 - beta21 * (S2 - S_0_21)
    
    # Calculate synaptic currents
    Isyn1 = g21 * S2 * (V1 - E_rev_21)  # Current from neuron 2 to 1
    Isyn2 = g12 * S1 * (V2 - E_rev_12)  # Current from neuron 1 to 2
    
    # Create temporary state vectors with updated Isyn values
    u1 = @view u[1:7]
    u2 = @view u[8:14]
    
    # Set synaptic currents (will be used by melibeNewIsyn!)
    u1_modified = SVector{7}(u1[1], u1[2], u1[3], u1[4], u1[5], u1[6], Isyn1)
    u2_modified = SVector{7}(u2[1], u2[2], u2[3], u2[4], u2[5], u2[6], Isyn2)
    
    # Update derivatives for each neuron using Plant.melibeNewIsyn!
    du1 = @view du[1:7]
    du2 = @view du[8:14]
    Plant.melibeNewIsyn!(du1, u1_modified, p1, t)
    Plant.melibeNewIsyn!(du2, u2_modified, p2, t)
    
    # Replace the last element of du1 and du2 with dS1 and dS2
    du[7] = dS1
    du[14] = dS2
end

# Function to run a half-center oscillator simulation
function run_hco_simulation(tspan=(0.0, 1000.0))
    # Initial conditions for both neurons (slightly different to break symmetry)
    u0_1 = [
      Plant.default_state_Isyn[1],
      0.0,
      Plant.default_state_Isyn[2:end-1]...,
      0.001 # Initial S1 value
    ]
    u0_2_Ca = 0.8
    u0_2 = [
      u0_1[1:4]...,
      u0_2_Ca,
      u0_1[5],
      0.001 # Initial S2 value
    ]
    
    # Combine into one state vector
    u0 = vcat(u0_1, u0_2)
    
    # Parameters for both neurons
    p1_ΔCa, p1_Δx = -38.285, -0.9
    p2_ΔCa, p2_Δx = -38.285, -0.9
    p1 = vcat(Plant.default_params[1:15], [p1_Δx, p1_ΔCa])
    p2 = vcat(Plant.default_params[1:15], [p2_Δx, p2_ΔCa])
    
    # Coupling parameters - mutual inhibition for half-center oscillator
    g12 = 0.3  # Inhibition from neuron 1 to 2
    g21 = 0.3  # Inhibition from neuron 2 to 1
    
    # Logistic synapse parameters
    # 1->2 synapse parameters
    alpha12 = 3e-2      # Rate parameter for activation
    beta12 = 1e-3       # Rate parameter for decay
    S_0_12 = 1e-3        # Baseline synaptic activity
    k12 = 1.0           # Steepness parameter for activation function
    Theta_syn_12 = 20.0  # Threshold for activation
    E_rev_12 = -80.0     # Reversal potential (inhibitory)
    
    # 2->1 synapse parameters (same defaults as 1->2)
    alpha21 = alpha12
    beta21 = beta12
    S_0_21 = S_0_12
    k21 = k12
    Theta_syn_21 = Theta_syn_12
    E_rev_21 = E_rev_12
    
    # Combined parameter vector
    p = vcat(p1, p2, [g12, g21, 
                      alpha12, beta12, S_0_12, k12, Theta_syn_12, E_rev_12,
                      alpha21, beta21, S_0_21, k21, Theta_syn_21, E_rev_21])
    
    # Create the ODE problem
    prob = ODEProblem(two_neurons!, u0, tspan, p)
    
    # Solve the system
    sol = solve(prob, Tsit5(), abstol=3e-6, reltol=3e-6)
    
    return sol
end

# Plot the results of a half-center oscillator
function plot_hco_results(sol)
  fig = Figure(resolution=(800, 800))
  
  # Create 4 axes in vertical arrangement
  ax1 = Axis(fig[1, 1], ylabel="V₁ (mV)", title="Half-Center Oscillator")
  ax2 = Axis(fig[2, 1], ylabel="Isyn₁")
  ax3 = Axis(fig[3, 1], ylabel="V₂ (mV)")
  ax4 = Axis(fig[4, 1], ylabel="Isyn₂", xlabel="Time")
  
  # Hide x-axis labels for all but the bottom plot
  hidexdecorations!(ax1, grid=false)
  hidexdecorations!(ax2, grid=false)
  hidexdecorations!(ax3, grid=false)
  
  # Plot the time series
  lines!(ax1, sol.t, sol[6, :], color=:blue)   # V1
  lines!(ax2, sol.t, sol[7, :], color=:blue)   # Isyn1
  lines!(ax3, sol.t, sol[13, :], color=:red)   # V2
  lines!(ax4, sol.t, sol[14, :], color=:red)   # Isyn2
  
  # Link the x-axes so they zoom together
  linkxaxes!(ax1, ax2, ax3, ax4)
  
  return fig
end

# Function to encode voltage traces as SSCSs after simulation
function encode_sscs(sol, V_sd=-50.0, filter_by=:minima, filter_threshold=50.0)
  # Extract voltage and synaptic current time series for both neurons
  t_values = sol.t
  V1_values = sol[6, :]
  V2_values = sol[13, :]
  Isyn1_values = sol[7, :]
  Isyn2_values = sol[14, :]
  
  # Initialize SSCS data structures for both neurons
  symbols1 = Int[]
  symbols2 = Int[]
  Vplus_times1 = Float64[]
  Vminus_times1 = Float64[]
  Vplus_times2 = Float64[]
  Vminus_times2 = Float64[]
  
  # Find extrema in synaptic currents for filtering (if needed)
  Isyn1_extrema_times = Float64[]
  Isyn2_extrema_times = Float64[]
  
  # Find local extrema in Isyn1
  for i in 2:(length(t_values)-1)
    is_extremum = false
    if filter_by == :minima
      is_extremum = Isyn1_values[i] < Isyn1_values[i-1] && Isyn1_values[i] < Isyn1_values[i+1]
    elseif filter_by == :maxima
      is_extremum = Isyn1_values[i] > Isyn1_values[i-1] && Isyn1_values[i] > Isyn1_values[i+1]
    end
    
    if is_extremum
      push!(Isyn1_extrema_times, t_values[i])
    end
  end
  
  # Find local extrema in Isyn2
  for i in 2:(length(t_values)-1)
    is_extremum = false
    if filter_by == :minima
      is_extremum = Isyn2_values[i] < Isyn2_values[i-1] && Isyn2_values[i] < Isyn2_values[i+1]
    elseif filter_by == :maxima
      is_extremum = Isyn2_values[i] > Isyn2_values[i-1] && Isyn2_values[i] > Isyn2_values[i+1]
    end
    
    if is_extremum
      push!(Isyn2_extrema_times, t_values[i])
    end
  end
  
  # Detect V maxima and encode as SSCS for Neuron 1
  count1 = 0
  last_symbol1 = :void
  last2_symbol1 = :void
  
  for i in 2:(length(t_values)-1)
    # Check if this is a local maximum in V1
    if V1_values[i] > V1_values[i-1] && V1_values[i] > V1_values[i+1]
      is_close_to_extremum = false
      # Check if this maximum is close to a synaptic current extremum
      for ext_time in Isyn1_extrema_times
        if abs(t_values[i] - ext_time) < filter_threshold
          is_close_to_extremum = true
          break
        end
      end
      
      if !is_close_to_extremum
        if V1_values[i] > V_sd
          # Vplus event (spike)
          count1 += 1
          last2_symbol1 = last_symbol1
          last_symbol1 = :Vplus
          push!(Vplus_times1, t_values[i])
        else
          # Vminus event (slow oscillation)
          if count1 > 0  # Only record if there were spikes to count
            if last2_symbol1 == :Vplus
              push!(symbols1, -count1)  # Negative count
            else
              push!(symbols1, count1)   # Positive count
            end
            count1 = 0
            last2_symbol1 = last_symbol1
            last_symbol1 = :Vminus
            push!(Vminus_times1, t_values[i])
          end
        end
      end
    end
  end
  
  # Detect V maxima and encode as SSCS for Neuron 2
  count2 = 0
  last_symbol2 = :void
  last2_symbol2 = :void
  
  for i in 2:(length(t_values)-1)
    # Check if this is a local maximum in V2
    if V2_values[i] > V2_values[i-1] && V2_values[i] > V2_values[i+1]
      is_close_to_extremum = false
      # Check if this maximum is close to a synaptic current extremum
      for ext_time in Isyn2_extrema_times
        if abs(t_values[i] - ext_time) < filter_threshold
          is_close_to_extremum = true
          break
        end
      end
      
      if !is_close_to_extremum
        if V2_values[i] > V_sd
          # Vplus event (spike)
          count2 += 1
          last2_symbol2 = last_symbol2
          last_symbol2 = :Vplus
          push!(Vplus_times2, t_values[i])
        else
          # Vminus event (slow oscillation)
          if count2 > 0  # Only record if there were spikes to count
            if last2_symbol2 == :Vplus
              push!(symbols2, -count2)  # Negative count
            else
              push!(symbols2, count2)   # Positive count
            end
            count2 = 0
            last2_symbol2 = last_symbol2
            last_symbol2 = :Vminus
            push!(Vminus_times2, t_values[i])
          end
        end
      end
    end
  end
  
  return (
    symbols1=symbols1, Vplus_times1=Vplus_times1, Vminus_times1=Vminus_times1,
    symbols2=symbols2, Vplus_times2=Vplus_times2, Vminus_times2=Vminus_times2
  )
end

# Function to convert SSCS to branch coordinate
function sscs_to_branch_coordinate(sscs::Vector{Int})
  coordinate_interval = (0, 1)
  orientation = 1
  new_orientation = orientation
  for i in 1:length(sscs)
    if sscs[i] <= 0
      individual_coordinate_interval = (1.0-2.0^sscs[i], 1.0-3.0*2.0^(sscs[i]-2))
      new_orientation *= -1
    else
      individual_coordinate_interval = (1.0-3.0*2.0^(-sscs[i]-2), 1.0-2.0^(-sscs[i]-1))
    end
    a, b = individual_coordinate_interval
    if orientation == 1
      coordinate_interval = (
        (1-a)*coordinate_interval[1] + a*coordinate_interval[2],
        (1-b)*coordinate_interval[1] + b*coordinate_interval[2]
      )
    else
      coordinate_interval = (
        a*coordinate_interval[1] + (1-a)*coordinate_interval[2],
        b*coordinate_interval[1] + (1-b)*coordinate_interval[2]
      )
    end
    orientation = new_orientation
  end
  return coordinate_interval
end

# Function to plot SSCS events on voltage traces
function plot_hco_results_with_sscs(sol, sscs_data, V_sd=-50.0)
  fig = Figure(resolution=(1000, 800))
  
  # Create 4 axes in vertical arrangement
  ax1 = Axis(fig[1, 1], ylabel="V₁ (mV)", title="Half-Center Oscillator with SSCS")
  ax2 = Axis(fig[2, 1], ylabel="Isyn₁")
  ax3 = Axis(fig[3, 1], ylabel="V₂ (mV)")
  ax4 = Axis(fig[4, 1], ylabel="Isyn₂", xlabel="Time")
  
  # Hide x-axis labels for all but the bottom plot
  hidexdecorations!(ax1, grid=false)
  hidexdecorations!(ax2, grid=false)
  hidexdecorations!(ax3, grid=false)
  
  # Plot the time series
  lines!(ax1, sol.t, sol[6, :], color=:blue)   # V1
  lines!(ax2, sol.t, sol[7, :], color=:blue)   # Isyn1
  lines!(ax3, sol.t, sol[13, :], color=:red)   # V2
  lines!(ax4, sol.t, sol[14, :], color=:red)   # Isyn2
  
  # Add horizontal line for V_sd threshold
  hlines!(ax1, V_sd, color=:black, linestyle=:dash)
  hlines!(ax3, V_sd, color=:black, linestyle=:dash)
  
  # Plot SSCS events for neuron 1
  scatter!(ax1, sscs_data.Vplus_times1, fill(V_sd+2, length(sscs_data.Vplus_times1)), 
           color=:green, markersize=8, marker=:utriangle)
  scatter!(ax1, sscs_data.Vminus_times1, fill(V_sd-2, length(sscs_data.Vminus_times1)), 
           color=:purple, markersize=8, marker=:dtriangle)
  
  # Plot SSCS events for neuron 2
  scatter!(ax3, sscs_data.Vplus_times2, fill(V_sd+2, length(sscs_data.Vplus_times2)), 
           color=:green, markersize=8, marker=:utriangle)
  scatter!(ax3, sscs_data.Vminus_times2, fill(V_sd-2, length(sscs_data.Vminus_times2)), 
           color=:purple, markersize=8, marker=:dtriangle)
  
  # Add SSCS labels for neuron 1
  for (i, t) in enumerate(sscs_data.Vminus_times1)
    if i <= length(sscs_data.symbols1)
      text!(ax1, string(sscs_data.symbols1[i]),
            position=(t, V_sd-10),
            align=(:center, :center), color=:purple, fontsize=14)
    end
  end
  
  # Add SSCS labels for neuron 2
  for (i, t) in enumerate(sscs_data.Vminus_times2)
      if i <= length(sscs_data.symbols2)
          text!(ax3, string(sscs_data.symbols2[i]),
              position=(t, V_sd-10),
              align=(:center, :center), color=:purple, fontsize=14)
      end
  end
  
  # Link the x-axes so they zoom together
  linkxaxes!(ax1, ax2, ax3, ax4)
  
  return fig
end

# Run the simulation
sol = run_hco_simulation(tspan)

# Encode SSCSs from the voltage traces
sscs_data = encode_sscs(sol, 20.0, :maxima, 1.0)

# Calculate branch coordinates
branch1 = sscs_to_branch_coordinate(sscs_data.symbols1)
branch2 = sscs_to_branch_coordinate(sscs_data.symbols2)

# Print SSCS information
println("Neuron 1 SSCS: ", sscs_data.symbols1)
println("Neuron 1 branch coordinate range: ", branch1)
println("Neuron 1 center branch coordinate: ", sum(branch1)/2)
println("\nNeuron 2 SSCS: ", sscs_data.symbols2)
println("Neuron 2 branch coordinate range: ", branch2)
println("Neuron 2 center branch coordinate: ", sum(branch2)/2)

# Plot results with SSCS markers
fig = plot_hco_results_with_sscs(sol, sscs_data)
save("hco_results_with_sscs.png", fig)
display(fig)

# Also display the original plot
fig_original = plot_hco_results(sol)
display(fig_original)