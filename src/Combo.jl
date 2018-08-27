"""
# COMBinatorial Optimization in Julia

The functions listed below are exported, see help on each of these using the ? operator.\n

## blackbox combinatorial optization

`cakewalk` - main entry point for the package. \n

### clique finding in undirected graphs

`clique_cw` - uses cakewalk to find cliques. \n
`is_local_max_clique` - validate that a clique is a locally maximal. \n

### k-medoids

Two standard algorithms for the k-medoids problem are supplied. A third algorithm
that utilizes a greedy search, and whose starting point is optimized by cakewalk 
will be added sometime soon. \n

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
