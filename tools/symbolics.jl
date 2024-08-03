@enum BranchSymbol begin
  SymbolA
  SymbolB
  SymbolC
  SymbolD
  SymbolE
  SymbolF
end

function itinerary_to_kneading_coordinate(itinerary::Vector{BranchSymbol}, periodic::Bool = false)::Vector{Float64}
  positive_orientation = true
  window = [0.0, 1.0]

  truncated_itinerary = itinerary
  start_index = findfirst(s -> s == SymbolA || s == SymbolB, itinerary)
  if start_index !== nothing
    truncated_itinerary = itinerary[start_index:end]
  else
    throw(ErrorException("No valid starting symbol (A or B) found in itinerary"))
  end

  while true
    for symbol in truncated_itinerary
      traverse_left = true
      left_index = positive_orientation ? 1 : 2
      right_index = positive_orientation ? 2 : 1
      if symbol == SymbolA
        traverse_left = true
      elseif symbol == SymbolB || symbol == SymbolD || symbol == SymbolF
        traverse_left = false
      elseif symbol == SymbolC || symbol == SymbolE
        positive_orientation = !positive_orientation
      end
      window[traverse_left ? right_index : left_index] = (window[left_index] + window[right_index]) / 2
    end
    if !periodic
      break
    end
    if window[1] == window[2]
      break
    end
    truncated_itinerary = itinerary[]
  end

  return window
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
