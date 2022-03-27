using POMDPs
using POMDPModels: TigerPOMDP
using QMDP: QMDPSolver
using BeliefUpdaters
using POMDPPolicies: action
using Random
using Printf
using POMDPSimulators

tiger_problem1 = TigerPOMDP(-1.0, -100.0, 10.0, 0.90, 0.90)
tiger_problem2 = TigerPOMDP(-2.0, -50.0, 17.0, 0.80, 0.60)
tiger_problem3 = TigerPOMDP(-3.0, -75.0, 8.0, 0.85, 0.75)

DPs = (tiger_problem1, tiger_problem2, tiger_problem3)
DCs = [4, 1, 2]
SSF = Base.minimum


## Method 1 ##
modia = MODIA(DPs, DCs, SSF)
initialize_beliefs!(modia, uniform_belief)

## Method 2 ##
modia = MODIA(DPs, DCs, SSF, uniform_belief)


"""Inspect the `modia` object"""
collect(modia)
typeof(modia)   # returns `MODIA_of_POMDPs`
actions(modia)   # returns the set containing (0, 1, 2)


"""Define a solver, and solve for the safest action"""
solver = QMDPSolver()
policies = solve(solver, modia)
act = safest_action(policies, modia)


"""Update beliefs of DCs in the MODIA"""
bu = DiscreteUpdater(modia);    # can also use updater(policies) to retrieve an appropiate belief updater
random_obs = [[Bool(rand(0:1)) for _ in sublist] for sublist in modia.markov_prcs]  # create random observations for all DCs
new_belief = update(bu, modia, act, random_obs);   # calculates new belief, but does not update within modia

update!(bu, modia, act, random_obs);   # calculates new belief, and updates within modia
collect_beliefs(modia)

"""Simulate DCs in the MODIA (does not update belief of modia)"""
sim = RolloutSimulator(max_steps=5)
simulate(sim, modia, policies)
simulate(sim, modia, policies, bu)

"""Simulate DCs in the MODIA, with specified initial states"""
rng = MersenneTwister()
s0 = rand(rng, initialstate(modia))   # rng argument is optional
simulate(sim, modia, policies, bu, s0)

"""History recorder"""
hr = HistoryRecorder(max_steps=5)
history = simulate(hr, modia, policies)
for (i, DPs) in enumerate(history)
    for (j, DCs) in enumerate(DPs)
        for (b,a,o,r,t) in eachstep(DCs, "b,a,o,r,t")
            println("DP: $i, DC: $j -> timestep: $t, reward: $r")
        end
    end
end


"""Stepthrough"""
sthru = stepthrough(modia, policies, bu, "b,a,o,r,t", max_steps=5)
for (i, DPs) in enumerate(sthru)
    for (j, DCs) in enumerate(DPs)
        for (b,a,o,r,t) in DCs
            println("DP: $i, DC: $j -> timestep: $t, reward: $r")
        end
    end
end


"""KMarkovUpdater"""
k_past_obs = 4
bu = KMarkovUpdater(modia, k_past_obs)
rng = MersenneTwister()
s0 = rand(rng, initialstate(modia))
initial_observation = rand(rng, initialobs(modia, s0))
initial_obs_vec = fill(initial_observation, k_past_obs)
b0 = initialize_belief(bu, initial_obs_vec)

modia = MODIA(DPs, DCs, SSF)
modia.beliefs = b0

random_obs = [[Bool(rand(0:1)) for _ in sublist] for sublist in modia.markov_prcs]  # create random observations for all DCs
new_belief = update(bu, modia, act, random_obs);   # calculates new belief, but does not update within modia
