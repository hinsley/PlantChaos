# Symbolic encoding

Trajectories in the full system can be represented as symbolic sequences of events.
We use a $\mathbb{Z}$-valued symbolic encoding for events in trajectories:

- The symbol $0$ corresponds to a subthreshold oscillation, which is orientation-preserving.
- Non-zero symbols correspond to individual bursting spike trains, with the magnitude of the symbol corresponding to the number of spikes in the train.
- The sign of a symbol corresponds to whether it preserves orientation, returning on the bottom or top of the reinsertion loop.
  A negative symbol corresponds to a "fresh" burst, reversing orientation at the return to the dune and corresponding to the template branch $\rm E$, while a positive symbol corresponds to a "stale" burst, preserving orientation and corresponding to the template branch $\rm F$.

## Encoder callbacks

While integrating the equations, a callback is used to detect events and record relevant events from the voltage timeseries $V(t)$ in realtime.

### Events
$V_{\rm sd}$ is the voltage of the upper saddle equilibrium.

- $V^+: \dot{V} = 0, V > V_{\rm sd}.$
  Non-terminal spike.
- $V^-: \dot{V} = 0, V < V_{\rm sd}, \ddot{V} < 0.$
  Rebound after terminal spike. TODO: Is the third condition even necessary?
- $I: \ddot{V} = 0, \frac{d^3V}{dt^3} < 0.$
  Local maximum of velocity.

### Event sequence interpreter
As events are detected, they are interpreted in an encoding algorithm.
This decreases processing time and memory requirements by removing the need to retain entire event sequences in memory.

The event sequence can be processed by ignoring instances of the event $I$ and keeping track of the symbol 2 indices prior in the sequence.
Reading the event sequence left-to-right, we may come across either the symbol $V^+$ or $V^-$.
- If we encounter $V^+$, this is to be thought of as a spike, either associated with branch $C$ or $D$ on the template.
- If we encounter $V^-$, we look back to the symbol two indices back, which we refer to below as the *context symbol.*
  - If the context symbol is $I$, then we have traversed the $E$ branch on the template.
  - If the context symbol is $V^+$, then we have traversed the $F$ branch on the template.
  - If the context symbol is $V^-$, then we have traversed the $A$ branch on the template.

Instead of logging branch symbols in accordance with semiflows on the template, we instead produce a final encoding that consists of a sequence of integers effectively representing which monotone interval of the one-dimensional return map on the template the trajectory passes through.

- An encoding symbol of $0$ indicates the observation of a subthreshold oscillation, accruing no spikes.
- A positive encoding symbol indicates the observation of a spike-train burst (with the symbol's magnitude indicating the number of spikes) ending with a traversal of the $E$ branch, corresponding to an increasing interval in the 1D return map.
- A negative encoding symbol indicates the observation of a spike-train burst (with the symbol's magnitude indicating the number of spikes) ending with a traversal of the $F$ branch, corresponding to a decreasing interval in the 1D return map.

This encoding scheme matches that at the beginning of this document.

The following pseudo-Julia describes the encoding algorithm via a callback ran by the solver:

```julia
count: Int = 0
last: Symbol = None
last2: Symbol = None
kneadings: Vector{Int} = []
V_sd: Float64 = calculate_V_sd()

function callback(V, Vdot, Vddot, Vdddot)
  if V_max()
    if V > V_sd
      count += 1
      last2 = last
      last = Vplus
    else
      push!(kneadings, last2 == Vplus ? -count : count)
      count = 0
      last2 = last
      last = Vminus
    end
  elseif Vdot_max()
    last2 = last
    last = I
  end
end
```