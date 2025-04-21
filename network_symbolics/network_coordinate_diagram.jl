# # Network coordinate diagrams
# A network coordinate diagram is a sequence of points in [0, 1]^n, where n is
# the number of neurons in the network. The components of the points in the
# sequence are the symbolic coordinates of the neurons in the network after each
# SSCS symbol is observed; each SSCS symbol among all neurons' voltage traces
# corresponds to a single point in the network coordinate diagram.

# ## Inputs
# In order to construct a network coordinate diagram, the SSCSs for each neuron
# in the network must be supplied. However, because it is necessary to know what
# order the symbols from the SSCSs occur in among multiple neurons, extra data
# must be supplied specifying the order. The way this is done is by passing in
# an extra numeric array for each neuron of the same length as the corresponding
# SSCS supplied, where the numbers among all of these numeric arrays specify
# from earliest symbol observed to latest. Therefore, these can be either the
# times at which the symbols are observed, or an integer index, or any other
# such ordering.
# Alternatively, the ordering of the symbols can be supplied by a single array
# of indices having length the sum of lengths of all SSCSs, where each index
# specifies to which SSCS the corresponding symbol belongs.

module NetworkCoordinateDiagram

export sequence

function sscs_to_symbolic_coordinate(sscs::Vector{Int})::Tuple{Float64, Float64}
  coordinate_interval = (0.0, 1.0)
  orientation = 1
  new_orientation = orientation
  for i in 1:length(sscs)
    if sscs[i] <= 0
      individual_coordinate_interval = (
        1.0 - 2.0^sscs[i],
        1.0 - 3.0*2.0^(sscs[i]-2)
      )
      new_orientation *= -1
    else
      individual_coordinate_interval = (
        1.0 - 3.0*2.0^(-sscs[i]-2),
        1.0 - 2.0^(-sscs[i]-1)
      )
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

function sequence(
  sscs_list::Vector{Vector{Int}},
  order_list::Vector{Vector{Float64}} # Times or some analogues thereof.
)::Vector{Vector{Float64}}
  # Make sure the number of SSCSs and order lists match.
  @assert length(sscs_list) == length(order_list)
  # Make sure the lengths of the SSCSs and order lists match.
  for i in 1:length(sscs_list)
    @assert length(sscs_list[i]) == length(order_list[i])
  end
  
  latest_ordinal = -Inf
  largest_ordinal = maximum(maximum.(order_list))
  sscs_indices = Int[]
  # Copy ordinal lists from order_list so we can mutate them.
  _order_list = [order_list[i][:] for i in 1:length(order_list)]
  println("Foo")
  println(_order_list)
  while latest_ordinal < largest_ordinal
    # Get the ordinal sequence index of the smallest ordinal among first entries
    # in _order_list and pop it, skipping empty vectors.
    non_empty_indices = findall(x -> !isempty(x), _order_list)
    if isempty(non_empty_indices)
        break  # Exit the loop if all vectors are empty.
    end
    smallest_ordinal = minimum(first.(_order_list[non_empty_indices]))
    smallest_ordinal_index = findfirst(i -> !isempty(_order_list[i]) && 
                                       first(_order_list[i]) == smallest_ordinal, 
                                       1:length(_order_list))
    push!(sscs_indices, smallest_ordinal_index)
    popfirst!(_order_list[smallest_ordinal_index])
  end

  println(sscs_indices)
  return sequence(sscs_list, sscs_indices)
end

function sequence(
  sscs_list::Vector{Vector{Int}},
  sscs_indices::Vector{Int}
)::Vector{Vector{Float64}}
  # Make sure the indices make sense.
  @assert maximum(sscs_indices) <= length(sscs_list)
  # Make sure the length of sscs_indices matches the number of symbols.
  @assert length(sscs_indices) == sum(length.(sscs_list))

  coordinate_sequence = Vector{Float64}[]
  # Copy sscs_list so we can mutate it.
  _sscs_list = [sscs_list[i][:] for i in 1:length(sscs_list)]
  for i in 1:length(sscs_indices)
    coordinate = [
      sum(sscs_to_symbolic_coordinate(sscs))/2 for sscs in _sscs_list
    ]
    push!(coordinate_sequence, coordinate)
    sscs_index = sscs_indices[i]
    popfirst!(_sscs_list[sscs_index])
  end
  return coordinate_sequence
end

end # module