using POMDPs
using BeliefUpdaters
using POMDPPolicies: action
using Random
using POMDPSimulators

abstract type MODIA end
struct MODIA_Policies; policies; end
struct MODIA_Updater; updater; end
struct MODIA_Beliefs; beliefs; end
struct MODIA_States; states; end
struct MODIA_Distribution; dist; end

mutable struct MODIA_of_POMDPs <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{POMDP}
    beliefs
end

mutable struct MODIA_of_MDPs <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{MDP}
    states
end

mutable struct MODIA_of_Mixture <: MODIA
    DPs
    DCs
    SSF
    markov_prcs::Vector{Union{POMDP,MDP}}
    states
    beliefs
end


function MODIA(DPs, DCs, SSF)
    dps = [[item] for item in DPs]
    markov_prcs = vcat(repeat.(dps, DCs)...)
    if all(isa.(DPs, POMDP))
        modia = MODIA_of_POMDPs(DPs, DCs, SSF, markov_prcs, nothing)
    elseif all(isa.(DPs, MDP))
        modia = MODIA_of_MDPs(DPs, DCs, SSF, markov_prcs, nothing)
    else
        modia = MODIA_of_Mixture(DPs, DCs, SSF, markov_prcs, nothing)
    end
    return modia
end

function MODIA(DPs, DCs, SSF, state_or_belief_init_func::Function)
    dps = [[item] for item in DPs]
    markov_prcs = vcat(repeat.(dps, DCs)...)
    if all(isa.(DPs, POMDP))
        modia = MODIA_of_POMDPs(DPs, DCs, SSF, markov_prcs, nothing)
        initialize_beliefs!(modia, state_or_belief_init_func)

    elseif all(isa.(DPs, MDP))
        modia = MODIA_of_MDPs(DPs, DCs, SSF, markov_prcs, nothing)
        initialize_states!(modia, state_or_belief_init_func)

    else
        modia = MODIA_of_Mixture(DPs, DCs, SSF, markov_prcs, nothing)
        initialize_states_and_beliefs!(modia, state_or_belief_init_func)
    end
    return modia
end

"""Initialize beliefs for all POMDPs in a MODIA."""
function initialize_beliefs!(modia::MODIA_of_POMDPs, belief_func::Function)
    beliefs = MODIA_Beliefs(belief_func.(modia.markov_prcs))
    modia.beliefs = beliefs
    return
end

"""Initialize states for all MDPs in a MODIA."""
function initialize_states!(modia::MODIA_of_MDPs, init_states_dist_func::Function)
    states = MODIA_States(rand.(init_states_dist_func.(modia.markov_prcs)))
    modia.states = states
    return
end

# TODO
"""Initialize states for all (PO)MDPs in a MODIA."""
function initialize_states_and_beliefs!(modia::MODIA_of_Mixture, init_states_dist_func::Function, belief_func::Function)
    # states = MODIA_States(rand.(init_states_dist_func.(modia.markov_prcs)))
    # modia.states = states
    return
end


"""Collects `markov_prcs` in a MODIA as a 1D Vector."""
collect(modia::MODIA) = vcat(getproperty(modia, :markov_prcs)...)

"""Collects `beliefs` in a MODIA as a 1D Vector."""
function collect_beliefs(modia::MODIA_of_POMDPs) 
    field = :beliefs
    dt = getproperty(modia, field)
    f = fieldnames(typeof(dt))[1]
    return vcat(getproperty(dt,f)...)
end

"""Collects `states` in a MODIA as a 1D Vector."""
function collect_states(modia::MODIA_of_MDPs) 
    field = :states
    dt = getproperty(modia, field)
    f = fieldnames(typeof(dt))[1]
    return getproperty(dt,f)
end

"""Returns the safest action among all possible actions in MODIA POMDPs."""
function safest_action(policies::MODIA_Policies, modia::MODIA_of_POMDPs)
    acts = POMDPs.action.(policies.policies, modia.beliefs.beliefs)
    return modia.SSF(acts)
end

"""Returns the safest action among all possible actions in MODIA MDPs."""
function safest_action(policies::MODIA_Policies, modia::MODIA_of_MDPs)
    acts = POMDPs.action.(policies.policies, modia.states.states)
    return modia.SSF(acts)
end

function get_DCs_inds_of_DP(modia::MODIA, DP_index::Int)
    prev_ids = sum(modia.DCs[1:DP_index-1])
    curr_idx = modia.DCs[DP_index]
    return [prev_ids, prev_ids+curr_idx]
end

function delete_DC!(modia::MODIA_of_POMDPs, DP_index::Int, DC_index::Int)
    if modia.DCs[DP_index] < 1
        error("Trying to delete more DCs than what exists.")
    end
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1 + DC_index)
    deleteat!(modia.beliefs.beliefs, inds1 + DC_index)
    modia.DCs[DP_index] -=  1
    return 
end

function delete_DC!(modia::MODIA_of_MDPs, DP_index::Int, DC_index::Int)
    if modia.DCs[DP_index] < 1
        error("Trying to delete more DCs than what exists.")
    end
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1 + DC_index)
    deleteat!(modia.states.states, inds1 + DC_index)
    modia.DCs[DP_index] -=  1
    return 
end

function delete_DCs!(modia::MODIA_of_POMDPs, DP_index::Int, DC_inds::AbstractArray{Int})
    if modia.DCs[DP_index] < length(DC_inds)
        error("Trying to delete more DCs than what exists.")
    end
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1 .+ DC_inds)
    deleteat!(modia.beliefs.beliefs, inds1 .+ DC_inds)
    modia.DCs[DP_index] -=  length(DC_inds)
    return 
end

function delete_DCs!(modia::MODIA_of_MDPs, DP_index::Int, DC_inds::AbstractArray{Int})
    if modia.DCs[DP_index] < length(DC_inds)
        error("Trying to delete more DCs than what exists.")
    end
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1 .+ DC_inds)
    deleteat!(modia.states.states, inds1 .+ DC_inds)
    modia.DCs[DP_index] -=  length(DC_inds)
    return 
end

function push_DCs!(modia::MODIA_of_POMDPs, DP_index::Int, DCs_to_add::Int)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    for _ in range(0, length=DCs_to_add)
        insert!(modia.beliefs.beliefs, inds1+1, nothing)
        insert!(modia.markov_prcs, inds1+1, modia.DPs[DP_index])
    end
    modia.DCs[DP_index] += DCs_to_add
    return
end

function push_DCs!(modia::MODIA_of_MDPs, DP_index::Int, DCs_to_add::Int)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    for _ in range(0, length=DCs_to_add)
        insert!(modia.states.states, inds1+1, nothing)
        insert!(modia.markov_prcs, inds1+1, modia.DPs[DP_index])
    end
    modia.DCs[DP_index] += DCs_to_add
    return
end

function push_DCs!(modia::MODIA_of_POMDPs, DP_index::Int, DCs_to_add::Int, belief_func::Function)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    for _ in range(0, length=DCs_to_add)
        insert!(modia.beliefs.beliefs, inds1+1, belief_func(modia.DPs[DP_index]))
        insert!(modia.markov_prcs, inds1+1, modia.DPs[DP_index])
    end
    modia.DCs[DP_index] += DCs_to_add
    return
end

function push_DCs!(modia::MODIA_of_MDPs, DP_index::Int, DCs_to_add::Int, init_states_dist_func::Function)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    for _ in range(0, length=DCs_to_add)
        insert!(modia.states.states, inds1+1, rand(init_states_dist_func(modia.DPs[DP_index])))
        insert!(modia.markov_prcs, inds1+1, modia.DPs[DP_index])
    end
    modia.DCs[DP_index] += DCs_to_add
    return
end

function delete_DP!(modia::MODIA_of_POMDPs, DP_index::Int)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1+1: inds2)
    deleteat!(modia.beliefs.beliefs, inds1+1: inds2)
    deleteat!(modia.DPs, DP_index)
    deleteat!(modia.DCs, DP_index)
    return 
end

function delete_DP!(modia::MODIA_of_MDPs, DP_index::Int)
    inds1, inds2 = get_DCs_inds_of_DP(modia, DP_index)
    deleteat!(modia.markov_prcs, inds1+1: inds2)
    deleteat!(modia.states.states, inds1+1: inds2)
    deleteat!(modia.DPs, DP_index)
    deleteat!(modia.DCs, DP_index)
    return 
end

function delete_DPs!(modia::MODIA_of_POMDPs, DP_inds::AbstractArray{Int})
    bounds = vcat((Base.collect(UnitRange(get_DCs_inds_of_DP(modia, DP_index) + [1,0]...)) for DP_index in DP_inds)...)
    sort!(bounds)
    deleteat!(modia.markov_prcs, bounds)
    deleteat!(modia.beliefs.beliefs, bounds)
    deleteat!(modia.DPs, DP_inds)
    deleteat!(modia.DCs, DP_inds)
    return 
end

function delete_DPs!(modia::MODIA_of_MDPs, DP_inds::AbstractArray{Int})
    bounds = vcat((Base.collect(UnitRange(get_DCs_inds_of_DP(modia, DP_index) + [1,0]...)) for DP_index in DP_inds)...)
    sort!(bounds)
    deleteat!(modia.markov_prcs, bounds)
    deleteat!(modia.states.states, bounds)
    deleteat!(modia.DPs, DP_inds)
    deleteat!(modia.DCs, DP_inds)
    return 
end

function push_DP!(modia::MODIA, DP_to_add)
    push!(modia.DPs, DP_to_add)
    push!(modia.DCs, 0)
    return
end

function push_DPs!(modia::MODIA, DPs_to_add::AbstractArray)
    append!(modia.DPs, DPs_to_add)
    for _ in range(0, length=length(DPs_to_add))
        push!(modia.DCs, 0)
    end
    return
end


### POMDPs.jl overloads ###

"""Returns a set of all possible actions among MODIA (PO)MDPs."""
POMDPs.actions(modia::MODIA) = Set(vcat(actions.(modia.markov_prcs)...))

"""Compute optimal policies for all POMDPs in a MODIA."""
POMDPs.solve(solver::Solver, modia::MODIA) = MODIA_Policies(solve.(Ref(solver), modia.markov_prcs))

POMDPs.updater(policies::MODIA_Policies) = MODIA_Updater(POMDPs.updater.(policies.policies))

POMDPs.initialstate(modia::MODIA) = MODIA_Distribution(initialstate.(modia.markov_prcs))
POMDPs.initialobs(modia::MODIA_of_POMDPs, s0) = MODIA_Distribution(initialobs.(modia.markov_prcs, s0))

POMDPs.rand(rng::AbstractRNG, d::MODIA_Distribution) = rand.(Ref(rng), d.dist)
POMDPs.rand(d::MODIA_Distribution) = rand.(d.dist)

POMDPs.isterminal(modia::MODIA, s) = any(POMDPs.isterminal.(modia.markov_prcs, s))


### BeliefUpdaters.jl overloads ###

BeliefUpdaters.uniform_belief(modia::MODIA_of_POMDPs) = MODIA_Beliefs(uniform_belief.(modia.markov_prcs))
BeliefUpdaters.update(bu::MODIA_Updater, modia::MODIA_of_POMDPs, act, obs) = MODIA_Beliefs(update.(bu.updater, modia.beliefs.beliefs, act, obs))

function update!(bu::MODIA_Updater, modia::MODIA_of_POMDPs, act, obs)
    modia.beliefs = BeliefUpdaters.update(bu, modia::MODIA_of_POMDPs, act, obs)
    return
end

BeliefUpdaters.DiscreteUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.DiscreteUpdater(pomdp) for pomdp in modia.markov_prcs])
BeliefUpdaters.NothingUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.NothingUpdater() for pomdp in modia.markov_prcs])
BeliefUpdaters.PreviousObservationUpdater(modia::MODIA_of_POMDPs) = MODIA_Updater([BeliefUpdaters.PreviousObservationUpdater() for pomdp in modia.markov_prcs])
BeliefUpdaters.KMarkovUpdater(modia::MODIA_of_POMDPs, k::Int) = MODIA_Updater([BeliefUpdaters.KMarkovUpdater(k) for pomdp in modia.markov_prcs])

# For KMarkovUpdater:
BeliefUpdaters.initialize_belief(modia::MODIA_Updater, initial_obs_vec::AbstractVector) = MODIA_Beliefs([BeliefUpdaters.PreviousObservations.(eachcol(transpose(hcat(item...)))) for item in eachrow(hcat(initial_obs_vec...))])


### POMDPSimulators.jl overloads ###

POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_MDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, modia.states)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, rand(initialstate(modia)))

POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)


POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA_of_MDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, modia.states)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA_of_POMDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, rand(initialstate(modia)))

POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA_of_POMDPs, policies::MODIA_Policies, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::HistoryRecorder, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)


POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA_of_MDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, modia.states)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater) = POMDPSimulators.simulate(sim, modia, policies, bu, rand(initialstate(modia)))

POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, POMDPs.updater(policies), modia.beliefs, initial_states)
POMDPSimulators.simulate(sim::StepSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater, initial_states::AbstractArray) = POMDPSimulators.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)


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

# TODO: Implement MMF
function POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_POMDPs, policies::MODIA_Policies, bu::MODIA_Updater, initial_beliefs::MODIA_Beliefs, initial_states::AbstractArray, MMF::Function=x->x)
    function take_step(pomdp::POMDP, up, safest_a, s, b, disc, rtotal)
        a = safest_a
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)
        
        rtotal += disc*r
        s = sp

        bp = update(up, b, a, o)
        b = bp

        disc *= discount(pomdp)
        return (s=s, b=b, disc=disc, rtotal=rtotal)
    end

    function safest_action(policies::MODIA_Policies, belief)
        acts = POMDPs.action.(policies.policies, belief)
        return modia.SSF(acts)
    end

    if isnothing(sim.eps)
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if isnothing(sim.max_steps)
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initial_states
    b = initialize_belief.(bu.updater, initial_beliefs.beliefs)
    disc = ones(length(modia.markov_prcs))
    rtotal = zeros(length(modia.markov_prcs))

    s_b_disc_rtotal = (s=s, b=b, disc=disc, rtotal=rtotal)
    
    step = 1
    is_terminal_flag = isterminal(modia, s)
    
    while any(disc .> eps) && !is_terminal_flag && step <= max_steps

        safest_a = safest_action(policies, b)
        s_b_disc_rtotal = take_step.(modia.markov_prcs, bu.updater, Ref(safest_a), s, b, disc, rtotal)

        s = getproperty.(s_b_disc_rtotal, :s)
        b = getproperty.(s_b_disc_rtotal, :b)
        disc = getproperty.(s_b_disc_rtotal, :disc)
        rtotal = getproperty.(s_b_disc_rtotal, :rtotal)
        s_b_disc_rtotal = (s=s, b=b, disc=disc, rtotal=rtotal)

        step += 1 
        is_terminal_flag = isterminal(modia, s)
    end

    return rtotal
end

# TODO: Implement HistoryRecorder

# TODO: Implement StepSimulator (is it already stepthrough?)



# TODO: Implement MMF
function POMDPSimulators.simulate(sim::RolloutSimulator, modia::MODIA_of_MDPs, policies::MODIA_Policies, initial_states::MODIA_States, MMF::Function=x->x)
    function take_step(mdp::MDP, safest_a, s, disc, rtotal)
        a = safest_a
        sp, r = @gen(:sp,:r)(mdp, s, a, sim.rng)

        rtotal += disc*r
        s = sp

        disc *= discount(mdp)
        return (s=s, disc=disc, rtotal=rtotal)
    end

    function safest_action(policies::MODIA_Policies, states)
        acts = POMDPs.action.(policies.policies, states)
        return modia.SSF(acts)
    end

    if isnothing(sim.eps)
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if isnothing(sim.max_steps)
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initial_states.states
    disc = ones(length(modia.markov_prcs))
    rtotal = zeros(length(modia.markov_prcs))

    s_disc_rtotal = (s=s, disc=disc, rtotal=rtotal)
    
    step = 1
    is_terminal_flag = isterminal(modia, s)
    
    while any(disc .> eps) && !is_terminal_flag && step <= max_steps

        safest_a = safest_action(policies, s)
        s_disc_rtotal = take_step.(modia.markov_prcs, Ref(safest_a), s, disc, rtotal)

        s = getproperty.(s_disc_rtotal, :s)
        disc = getproperty.(s_disc_rtotal, :disc)
        rtotal = getproperty.(s_disc_rtotal, :rtotal)
        s_disc_rtotal = (s=s, disc=disc, rtotal=rtotal)

        step += 1 
        is_terminal_flag = isterminal(modia, s)
    end

    return rtotal
end