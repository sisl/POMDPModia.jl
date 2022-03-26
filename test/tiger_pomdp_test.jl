using POMDPs
using POMDPModels: TigerPOMDP
using QMDP: QMDPSolver
using BeliefUpdaters: DiscreteBelief, DiscreteUpdater, uniform_belief, update
using POMDPPolicies: action

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

# ## Method 3 ##
# modia = MODIA(DPs, DCs, SSF)
# modia.belief = object_here  # TODO

"""Inspect the `modia` object"""
collect(modia)
typeof(modia)   # returns `MODIA_of_POMDPs`
actions(modia)   # returns the set containing (0, 1, 2)


"""Define a solver, and solve for the safest action"""
solver = QMDPSolver()
policies = solve(solver, modia)
act = safest_action(policies, modia)


"""Update beliefs of DCs in the MODIA"""
bu = DiscreteUpdater(modia);
random_obs = [[Bool(rand(0:1)) for _ in sublist] for sublist in modia.markov_prcs]  # Create random observations for all DCs
new_belief = update(bu, modia, act, random_obs);   # calculates new belief, but does not update within modia

update!(bu, modia, act, random_obs); 
collect(modia, :beliefs)