using POMDPs
using BeliefUpdaters
using POMDPPolicies: action

abstract type MODIA end

mutable struct MODIA_of_POMDPs <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{Vector{POMDP}}
    beliefs
end

mutable struct MODIA_of_MDPs <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{Vector{MDP}}
end

mutable struct MODIA_of_Mixture <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{Vector{Union{POMDP,MDP}}}
    beliefs
end

function MODIA(DPs, DCs, SSF)
    beliefs = nothing
    dps = [[item] for item in DPs]
    markov_prcs = repeat.(dps, DCs)
    if all(isa.(DPs, POMDP))
        modia = MODIA_of_POMDPs(DPs, DCs, SSF, markov_prcs, beliefs)
    elseif all(isa.(DPs, MDP))
        modia = MODIA_of_MDPs(DPs, DCs, SSF, markov_prcs)
    else
        modia = MODIA_of_Mixture(DPs, DCs, SSF, markov_prcs, beliefs)
    end
    return modia
end

function MODIA(DPs, DCs, SSF, beliefs::Function)
    dps = [[item] for item in DPs]
    markov_prcs = repeat.(dps, DCs)
    if all(isa.(DPs, POMDP))
        modia = MODIA_of_POMDPs(DPs, DCs, SSF, markov_prcs, beliefs)
        initialize_beliefs!(modia, beliefs)

    elseif all(isa.(DPs, MDP))
        modia = MODIA_of_MDPs(DPs, DCs, SSF, markov_prcs)

    else
        modia = MODIA_of_Mixture(DPs, DCs, SSF, markov_prcs, beliefs)
        initialize_beliefs!(modia, beliefs)   # TODO

    end
    return modia
end

"""Collects a field of a MODIA as a 1D Vector. Defaults to the field `markov_prcs`."""
collect(modia::MODIA, field::Symbol=:markov_prcs) = vcat(getproperty(modia, field)...)


"""Returns a set of all possible actions among MODIA (PO)MDPs."""
POMDPs.actions(modia::MODIA) = Set(vcat(actions.(collect(modia))...))

"""Initialize belief for all POMDPs in a MODIA."""
function initialize_beliefs!(modia::MODIA_of_POMDPs, belief_func::Function)
    beliefs = [belief_func.(sublist) for sublist in modia.markov_prcs]
    modia.beliefs = beliefs
    return
end

"""Compute optimal policies for all POMDPs in a MODIA."""
POMDPs.solve(solver::Solver, modia::MODIA_of_POMDPs) = [solve.(Ref(solver), sublist) for sublist in modia.markov_prcs]

"""Returns the safest action among all possible actions in MODIA (PO)MDPs."""
function safest_action(policies, modia::MODIA)
    acts = POMDPs.action.(vcat(policies...), vcat(modia.beliefs...))
    return modia.SSF(acts)
end

BeliefUpdaters.update(bu, modia::MODIA_of_POMDPs, act, obs) = [update.(bu[idx], sublist, Ref(act), obs[idx]) for (idx, sublist) in enumerate(modia.beliefs)]

function update!(bu, modia::MODIA_of_POMDPs, act, obs)
    modia.beliefs = BeliefUpdaters.update(bu, modia::MODIA_of_POMDPs, act, obs)
    return
end

BeliefUpdaters.DiscreteUpdater(modia::MODIA_of_POMDPs) = [BeliefUpdaters.DiscreteUpdater.(sublist) for sublist in modia.markov_prcs]





function stepthrough(modia::MODIA)
end

function simulate(modia::MODIA)
end