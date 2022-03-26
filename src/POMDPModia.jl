module POMDPModia

using POMDPs
using POMDPSimulators
using POMDPModelTools
using BeliefUpdaters
using POMDPTesting
using UUIDs
using NamedTupleTools
using Random
using Tricks: static_hasmethod

# # TODO: Use these? github.com/JuliaPOMDP/QMDP.jl/blob/master/src/QMDP.jl
# using POMDPPolicies
# using DiscreteValueIteration

import POMDPs: Solver
import POMDPs: solve

export
    MODIA,
    MODIA_of_POMDPs,
    MODIA_of_MDPs,
    MODIA_of_Mixture

include("main.jl")

end # module
