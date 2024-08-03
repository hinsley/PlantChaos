@enum BranchSymbol begin
  SymbolA
  SymbolB
  SymbolC
  SymbolD
  SymbolE
  SymbolF
end

function voltage_trace_to_itinerary(voltage_trace::Vector{Float64}, times::Vector{Float64})::Vector{BranchSymbol}
  itinerary = BranchSymbol[]
  spike_threshold = 0.0 # This is a V value.
  exit_on_E = false # Whether transition from gateau roule back to dune is on E or F branch; necessary for state tracking.

  # Bounce detection. Used for distinguishing E/F symbols.
  interspike_state = 1 # 1: Pre-hyperpolarization, 2: Post-hyperpolarization, 3: Prebounce, 4: Bounce detection watchdog.
  Vdot_hyperpolarization_threshold_min = -1.0
  Vdot_prebounce_threshold_max = 0.05
  Vdot_bounce_watchdog_threshold_min = 0.005
  min_Vdot_observed = Inf
  bounce_Vdot_increase_threshold = 1e-6

  for i in 3:length(voltage_trace) # Start at 3 so we can detect turning points.
    if length(itinerary) == 0 || itinerary[end] == SymbolA || itinerary[end] == SymbolE || itinerary[end] == SymbolF # Top branch line.
      if voltage_trace[i-2] < voltage_trace[i-1] && voltage_trace[i-1] > voltage_trace[i] # V maximum.
        push!(itinerary, SymbolA)
      elseif voltage_trace[i] > spike_threshold # Starting to spike.
        push!(itinerary, SymbolB)
      end
    elseif itinerary[end] == SymbolB || itinerary[end] == SymbolD # Second branch line (gateau roule).
      # Calculate derivative.
      Vdot = (voltage_trace[i] - voltage_trace[i-1]) / (times[i] - times[i-1])
      if interspike_state == 1 # Pre-hyperpolarization.
        if Vdot < Vdot_hyperpolarization_threshold_min
          interspike_state = 2
        end
      elseif interspike_state == 2 # Post-hyperpolarization.
        if Vdot > Vdot_prebounce_threshold_max
          interspike_state = 3
        end
      else
        if voltage_trace[i] > spike_threshold # Additional spike detected.
          push!(itinerary, SymbolD)
          interspike_state = 1
          min_Vdot_observed = Inf
          exit_on_E = false
        elseif interspike_state == 3 # Prebounce.
          if Vdot < Vdot_bounce_watchdog_threshold_min
            interspike_state = 4
          end
        elseif interspike_state == 4 # Bounce detection watchdog.
          min_Vdot_observed = min(min_Vdot_observed, Vdot)
          if Vdot >= min_Vdot_observed + bounce_Vdot_increase_threshold # Bounce detected.
            exit_on_E = true
          elseif Vdot <= 0 # Completed return to dune, including any potential half-rotation.
            push!(itinerary, SymbolC)
            interspike_state = 1
            min_Vdot_observed = Inf
          end
        end
      end
    elseif itinerary[end] == SymbolC # Third branch line (horseshoe on dune).
      push!(itinerary, exit_on_E ? SymbolE : SymbolF)
      exit_on_E = false
    end
  end
  
  return itinerary
end
