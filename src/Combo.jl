"""
# COMBinatorial Optimization in Julia

See help on each of the exported functions.\n

## blackbox combinatorial optization

`cakewalk` \n

### clique finding in undirected graphs

`clique_cw` \n
`is_local_max_clique` \n

### k-medoids

`kmedoids_voronoi` \n
`kmedoids_pam` \n

"""
module Combo

using StatsBase
using Iterators

const Float = Float64

import Base: get
include("utils.jl")
include("Dist.jl")
include("cakewalk.jl")
export cakewalk
include("clique.jl")
export clique_cw, is_local_max_clique
include("kmedoids.jl")
export kmedoids_voronoi, kmedoids_pam

end
