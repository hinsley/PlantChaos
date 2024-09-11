@enum BranchSymbol begin
  SymbolA
  SymbolB
  SymbolC
  SymbolD
  SymbolE
  SymbolF
end

# Conditional block entropy for size n blocks
function conditional_block_entropy(sequence::Vector{Int}, n::Int)::Float64
    if n <= 0 || isempty(sequence)
        return 0.0
    end

    # Count occurrences of blocks and their extensions
    block_counts = Dict{Vector{Int}, Int}()
    extended_block_counts = Dict{Vector{Int}, Int}()

    for i in 1:(length(sequence) - n)
        block = sequence[i:(i+n-1)]
        extended_block = sequence[i:(i+n)]

        block_counts[block] = get(block_counts, block, 0) + 1
        extended_block_counts[extended_block] = get(extended_block_counts, extended_block, 0) + 1
    end

    # Calculate conditional entropy
    total_blocks = length(sequence) - n + 1
    entropy = 0.0

    for (block, count) in block_counts
        p_block = count / total_blocks

        for symbol in unique(sequence)
            extended_block = vcat(block, symbol)
            extended_count = get(extended_block_counts, extended_block, 0)

            if extended_count > 0
                p_extended = extended_count / count
                entropy -= p_block * p_extended * log2(p_extended)
            end
        end
    end

    return entropy
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

function itinerary_to_kneading_sequence(itinerary::Vector{BranchSymbol})::Vector{Int}
  # Assumes the itinerary starts with A or B.

  kneading_sequence = Int[]
  kneading_symbol_accumulator = 2

  for symbol in itinerary
    if symbol == SymbolA
      push!(kneading_sequence, 1)
    elseif symbol == SymbolB
      kneading_symbol_accumulator = 2
    # Symbol C is redundant.
    elseif symbol == SymbolD
      kneading_symbol_accumulator += 2
    elseif symbol == SymbolE
      kneading_symbol_accumulator += 1
      push!(kneading_sequence, kneading_symbol_accumulator)
    elseif symbol == SymbolF
      push!(kneading_sequence, kneading_symbol_accumulator)
    end
  end

  return kneading_sequence
end

# Normalized Lempel-Ziv complexity.
function normalized_LZ_complexity(sequence::Vector{Int})::Float64
    if isempty(sequence)
        return 0.0
    end

    # Initialize variables
    dictionary = Dict{Vector{Int}, Bool}()
    current_substring = Int[]
    complexity = 0

    for symbol in sequence
        push!(current_substring, symbol)
        
        if !haskey(dictionary, current_substring)
            complexity += 1
            dictionary[copy(current_substring)] = true
            current_substring = Int[]
        end
    end

    # Add the last substring if it's not empty
    if !isempty(current_substring) && !haskey(dictionary, current_substring)
        complexity += 1
    end

    # Normalize the complexity
    n = length(sequence)
    b = length(unique(sequence))
    if b == 1
        normalized_complexity = 0.0
    else
        normalized_complexity = complexity * log2(n) / (n * log2(b))
    end

    return complexity
    #return normalized_complexity
end

function topological_entropy(upper_saddle_kneading_sequence::Vector{Int}, flow_tangency_kneading_sequence::Vector{Int}, n_max::Int, ε::Float64=1e-2)::Float64
  # (A1) Parameters.
  l = upper_saddle_kneading_sequence[1] - 1 # Number of extrema in the map.
  # n_max is the maximum number of iterations (symbols to process per itinerary).
  # ε is the dynamic halt criterion.

  # We don't have i going from 1 to l, but instead just from 1 to 2, so we need to have multipliers for the number of extrema.
  χ_1 = BigInt(ceil(l/2)) # Number of preimages of the upper saddle equilibrium.
  χ_2 = BigInt(l - χ_1) # Number of preimages of the flow tangency.
  max_1 = true # Tracks the min-max state of the upper saddle. True for max, false for min.
  max_2 = false # Tracks the min-max state of the flow tangency. True for max, false for min.

  # (A2) Initialization.
  s1 = BigInt[1] # Number of interior simple zeroes of f^\nu(x) - c_1. (Transversal preimages of the upper saddle.) Zero-indexed in the algorithm pseudocode from the paper.
  s2 = BigInt[1] # Number of interior simple zeroes of f^\nu(x) - c_2. (Transversal preimages of the flow tangency.) Zero-indexed in the algorithm pseudocode from the paper.
  s = BigInt[l] # χ_1 * s1 + χ_2 * s2
  s_sum = BigInt(l)
  K1 = [(1, 1)] # Bad symbols with respect to the 1st critical line (associated with the upper saddle maximum, so these are maxima).
  K2 = [(2, 1)] # Bad symbols with respect to the 2nd critical line (associated with the flow tangency minimum, so these are minima).

  # (A3) First iteration.
  S1 = BigInt[]
  S2 = BigInt[]
  S = BigInt(0)
  
  function update_S()
    # Initialize accumulators.
    S1_acc = BigInt(0)
    S2_acc = BigInt(0)

    ν = length(s)
    # Update S1_acc.
    for (k, κ) in K1
      if k == 1
        S1_acc += s1[ν+1-κ] * χ_1
      else
        S1_acc += s2[ν+1-κ] * χ_2
      end
    end
    # Update S2_acc.
    for (k, κ) in K2
      if k == 1
        S2_acc += s1[ν+1-κ] * χ_1
      else
        S2_acc += s2[ν+1-κ] * χ_2
      end
    end

    # Multiply by 2 and conclude computation.
    push!(S1, 2*S1_acc)
    push!(S2, 2*S2_acc)
    
    # Update S.
    S = χ_1 * S1[end] + χ_2 * S2[end]
  end

  function update_s()
    # Update s_ν^i.
    push!(s1, 1 + s_sum - S1[end]) # Equation (14) from the paper.
    push!(s2, 1 + s_sum - S2[end]) # Equation (14) from the paper.

    # Update s_ν.
    push!(s, l * (1 + s_sum) - S[end])
    s_sum += s[end] # Update the sum over all s_ν ("all" with respect to ν).
  end

  update_S()
  update_s()

  # (A4) Computation loop.

  top_entropy_estimate = Inf # Initial estimate. Inf is chosen because algorithm converges from above.
  while (ν = length(s) - 1) < n_max
    # Update K_ν^i.
    function update_K()
      # Determine next min-max symbols.
      if iseven(upper_saddle_kneading_sequence[ν])
        max_1 = !max_1
      end
      if iseven(flow_tangency_kneading_sequence[ν])
        max_2 = !max_2
      end
      # Determine which are bad symbols.
      if max_1 # Upper saddle image is a maximum.
        push!(K1, (1, ν))
      else # Upper saddle image is a minimum.
        push!(K2, (1, ν))
      end
      if max_2 # Flow tangency image is a maximum.
        push!(K1, (2, ν))
      else # Flow tangency image is a minimum.
        push!(K2, (2, ν))
      end
    end

    update_K()
    update_S()
    update_s()

    # Update topological entropy estimate.
    top_entropy_estimate = log((s[end]+S[end])/l)/ν
    println(top_entropy_estimate)

    # Check for halt criterion.
    if abs(top_entropy_estimate/(1-ν)) ≤ ε
      return top_entropy_estimate # (A5.1) Output.
    end
  end

  # (A5.2) Output (failure).
  error("Failed to converge within maximum iterations $n_max.")
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
          if Vdot > min_Vdot_observed# Bounce detected.
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