using POMDPs
using BeliefUpdaters
using POMDPPolicies: action
using Random
using POMDPSimulators

abstract type MODIA end
struct MODIA_Policies; policies; end
struct MODIA_Updater; updater; end
struct MODIA_Beliefs; beliefs; end
struct MODIA_Distribution; dist; end

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

function MODIA(DPs, DCs, SSF, belief_func::Function)
    dps = [[item] for item in DPs]
    markov_prcs = repeat.(dps, DCs)
    if all(isa.(DPs, POMDP))
        modia = MODIA_of_POMDPs(DPs, DCs, SSF, markov_prcs, nothing)
        initialize_beliefs!(modia, belief_func)

    elseif all(isa.(DPs, MDP))
        error("Cannot use `belief_func` for MDPs. Call MODIA(DPs, DCs, SSF) instead.")

    else
        modia = MODIA_of_Mixture(DPs, DCs, SSF, markov_prcs, nothing)
        initialize_beliefs!(modia, belief_func)   # TODO

    end
    return modia
end

"""Initialize belief for all POMDPs in a MODIA."""
function initialize_beliefs!(modia::MODIA_of_POMDPs, belief_func::Function)
    beliefs = MODIA_Beliefs([belief_func.(sublist) for sublist in modia.markov_prcs])
    modia.beliefs = beliefs
    return
end

"""Collects `markov_prcs` in a MODIA as a 1D Vector."""
collect(modia::MODIA) = vcat(getproperty(modia, :markov_prcs)...)

"""Collects `beliefs` in a MODIA as a 1D Vector."""
function collect_beliefs(modia::MODIA) 
    field = :beliefs
    dt = getproperty(modia, field)
    f = fieldnames(typeof(dt))[1]
    return vcat(getproperty(dt,f)...)
end


"""Returns the safest action among all possible actions in MODIA (PO)MDPs."""
function safest_action(policies::MODIA_Policies, modia::MODIA)
    acts = POMDPs.action.(vcat(policies.policies...), vcat(modia.beliefs.beliefs...))
    return modia.SSF(acts)
end



### POMDPs.jl overloads ###

"""Returns a set of all possible actions among MODIA (PO)MDPs."""
POMDPs.actions(modia::MODIA) = Set(vcat(actions.(collect(modia))...))

"""Compute optimal policies for all POMDPs in a MODIA."""
POMDPs.solve(solver::Solver, modia::MODIA_of_POMDPs) = MODIA_Policies([solve.(Ref(solver), sublist) for sublist in modia.markov_prcs])

POMDPs.updater(policies::MODIA_Policies) = MODIA_Updater([POMDPs.updater(pol[1]) for pol in policies.policies])

POMDPs.initialstate(modia::MODIA_of_POMDPs) = MODIA_Distribution([initialstate.(sublist) for sublist in modia.markov_prcs])
POMDPs.initialobs(modia::MODIA_of_POMDPs, s0) = MODIA_Distribution([initialobs.(sublist, s0[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)])

POMDPs.rand(rng::AbstractRNG, d::MODIA_Distribution) = [rand.(Ref(rng),sublist) for sublist in d.dist]
POMDPs.rand(d::MODIA_Distribution) = [rand.(sublist) for sublist in d.dist]


### BeliefUpdaters.jl overloads ###

BeliefUpdaters.update(bu::MODIA_Updater, modia::MODIA_of_POMDPs, act, obs) = MODIA_Beliefs([update.(Ref(bu.updater[idx]), sublist, Ref(act), obs[idx]) for (idx, sublist) in enumerate(modia.beliefs.beliefs)])

function update!(bu::MODIA_Updater, modia::MODIA_of_POMDPs, act, obs)
    modia.beliefs = BeliefUpdaters.update(bu, modia::MODIA_of_POMDPs, act, obs)
    return
end

BeliefUpdaters.DiscreteUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.DiscreteUpdater(sublist[1]) for sublist in modia.markov_prcs])
BeliefUpdaters.NothingUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.NothingUpdater() for sublist in modia.markov_prcs])
BeliefUpdaters.PreviousObservationUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.PreviousObservationUpdater() for sublist in modia.markov_prcs])
BeliefUpdaters.KMarkovUpdater(modia::MODIA_of_POMDPs, k::Int) = MODIA_Updater([BeliefUpdaters.KMarkovUpdater(k) for sublist in modia.markov_prcs])

BeliefUpdaters.initialize_belief(modia::MODIA_Updater, initial_obs_vec::AbstractVector) = MODIA_Beliefs([BeliefUpdaters.PreviousObservations.(eachcol(transpose(hcat(item...)))) for item in eachrow(hcat(initial_obs_vec...))])

BeliefUpdaters.uniform_belief(modia::MODIA_of_POMDPs) = MODIA_Beliefs([uniform_belief.(sublist) for sublist in modia.markov_prcs])


### POMDPSimulators.jl overloads ###

POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]

POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies, initial_states) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_states) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs, initial_states) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx], initial_states[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]


POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]

POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies, initial_states) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_states) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs, initial_states) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx], initial_states[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]


POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]

POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies, initial_states) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_states) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs, initial_states) = [POMDPSimulators.simulate.(Ref(sim), sublist, policies.policies[idx], Ref(bu.updater[idx]), initial_beliefs.beliefs[idx], initial_states[idx]) for (idx,sublist) in enumerate(modia.markov_prcs)]


function POMDPSimulators.stepthrough(modia::MODIA_of_POMDPs, policies::MODIA_Policies, args...; kwargs...)
    spec_included=false
    if !isempty(args) && isa(last(args), Union{String, Tuple, Symbol})
        spec = last(args)
        spec_included = true
        # if spec isa statetype(pomdp) && length(args) == 3
        #     error("Ambiguity between `initial_state` and `spec` arguments in stepthrough. Please explicitly specify the initial state and spec.")
        # end
    else
        spec = tuple(:s, :a, :sp, :o, :r, :info, :t, :action_info, :b, :bp, :update_info)
    end
    sim = StepSimulator(spec; kwargs...)
    return simulate(sim, modia, policies, args[1:end-spec_included]...)
end


