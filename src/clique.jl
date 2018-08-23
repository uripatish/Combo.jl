const GraphMatrix = Union{Matrix, BitMatrix}

function clique_size(seq::Union{Vector{Bool}, BitArray}, A::GraphMatrix)
  set = find(seq)
  clique = true
  for i in set, j in set
    if (i != j) && !(A[i,j] > 0)
      clique = false
      break
    end
  end
  return clique*length(set)
end

function clique_loss(seq::Union{Vector{Bool}, BitArray}, A::GraphMatrix, kappa::Float)  
  set = find(seq)
  clique = true
  edges = Float(0)
  n = length(set)  
  for i in set, j in set
    if i != j
      aij = (A[i,j] > 0)
      edges += aij 
      clique = clique && aij
    end
  end

  return (Float(1) - Float(edges)/Float(max(n*(n-1+kappa), 1)), Float(-n*clique))
end

function clique_loss(seq::Vector{Int}, A::GraphMatrix, kappa::Float)
  n = size(A, 1)
  if length(seq) < n
    idx = falses(n)
    idx[seq] = true
  else
    idx = seq .> 1
  end
  return clique_loss(idx, A, kappa)
end

"""
### Clique finding using the cakewalk method
Minimize the soft-clique-loss function with cakewalk. Usage: \n
`opt_seq, opt_val = clique_cw(A, kappa)` \n
where `kappa` is a size booster parameter beteen 0.0 and 1.0. 
For details about the loss, kappa, and the reasoning behind them see our paper.\n\n

When `opt_seq[i]` is 2, it means vertex `i` is selected, and 1 means it is not. \n
`opt_val` is a tuple of two floats. The first is the value being optimized, i.e., 
the soft-clique-loss, and the second is the actual loss that we are interested in, 
the negative clique size. 
"""
clique_cw(A::GraphMatrix, args...; kwargs...) = cakewalk(clique_loss, 2, size(A, 1), A, args...; kwargs...)

inc_clique_size(ind, A::GraphMatrix) = maximum(map(v -> (w = copy(ind); w[v] = true; clique_size(w, A)), 1:size(A,1)))

"""
### Validate that a clique is locally maximal
First, find some clique, \n
`opt_seq, opt_val = clique_cw(A, kappa)` \n
Then, validate the clique is locally maximal, \n
`is_local_max_clique(opt_seq .== 2, A)`\n
The returned value is binary.
"""
is_local_max_clique(ind, A::GraphMatrix) = (cs = clique_size(ind,A); (cs > 0) && (cs >= inc_clique_size(ind, A)))
