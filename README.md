# POMDPModia

This package provides a core interface for creating [MODIA](https://www.ijcai.org/proceedings/2017/664) objects comprising [Markov decision processes (MDPs)](https://en.wikipedia.org/wiki/Markov_decision_process) and/or [partially observable Markov decision processes (POMDPs)](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process).  The package is written in the [Julia Programming Language](https://julialang.org/). 

! TODO: Describe MODIA architecture in more detail here.


The POMDPModia.jl package is highly compatible and dependent on the other packages in the [JuliaPOMDP](https://github.com/JuliaPOMDP) ecosystem. 

## Dependencies

The (PO)MDP definitions used in POMDPModia.jl are instantiate through the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) package. Through the POMDPs.jl interface, problems are defined and solved using a large variety of tools available:

|  **`Package`**   |  **`Build`** | **`Coverage`** |
|-------------------|----------------------|------------------|
| [POMDPs](https://github.com/JuliaPOMDP/POMDPs.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPs.jl/actions/workflows/CI.yml/) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPs.jl) |
| [QuickPOMDPs](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) | [![Build Status](https://github.com/JuliaPOMDP/QuickPOMDPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/QuickPOMDPs.jl/actions/workflows/CI.yml/) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/QuickPOMDPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/QuickPOMDPs.jl) |
| [POMDPModelTools](https://github.com/JuliaPOMDP/POMDPModelTools.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPModelTools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPModelTools.jl/actions/workflows/CI.yml/) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPModelTools.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPModelTools.jl) |
| [BeliefUpdaters](https://github.com/JuliaPOMDP/BeliefUpdaters.jl) | [![Build Status](https://github.com/JuliaPOMDP/BeliefUpdaters.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/BeliefUpdaters.jl) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/BeliefUpdaters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/BeliefUpdaters.jl?) |
| [POMDPPolicies](https://github.com/JuliaPOMDP/POMDPPolicies.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPPolicies.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPPolicies.jl) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPPolicies.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPPolicies.jl?) |
| [POMDPSimulators](https://github.com/JuliaPOMDP/POMDPSimulators.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPSimulators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPSimulators.jl) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPSimulators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPSimulators.jl?) |
| [POMDPModels](https://github.com/JuliaPOMDP/POMDPModels.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPModels.jl) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPModels.jl?) |
| [POMDPTesting](https://github.com/JuliaPOMDP/POMDPTesting.jl) | [![Build Status](https://github.com/JuliaPOMDP/POMDPTesting.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPTesting.jl) | [![Coverage Status](https://codecov.io/gh/JuliaPOMDP/POMDPTesting.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaPOMDP/POMDPTesting.jl?) |
<!-- | [ParticleFilters](https://github.com/JuliaPOMDP/ParticleFilters.jl) | [![Build Status](https://github.com/JuliaPOMDP/ParticleFilters.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/ParticleFilters.jl) | [![codecov.io](http://codecov.io/github/JuliaPOMDP/ParticleFilters.jl/coverage.svg?)](http://codecov.io/github/JuliaPOMDP/ParticleFilters.jl?) | -->

Also, see the [supported solvers](https://github.com/JuliaPOMDP/POMDPs.jl#supported-packages) for the summary of solvers available, and the [solvers documentation](https://juliapomdp.github.io/POMDPs.jl/stable/def_solver/) for examples and writing custom solvers.

## Quick Start

In this short example, a MODIA will be created for various amounts of three different [TigerPOMDP](https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/TigerPOMDPs.jl) characteristics.

```julia
using POMDPModels: TigerPOMDP
using QMDP: QMDPSolver

# Define the DPs, DCs, and SSF for a MODIA object
tiger_problem1 = TigerPOMDP(-1.0, -100.0, 10.0, 0.90, 0.90);
tiger_problem2 = TigerPOMDP(-2.0, -50.0, 17.0, 0.80, 0.60);
tiger_problem3 = TigerPOMDP(-3.0, -75.0, 8.0, 0.85, 0.75);
DPs = [tiger_problem1, tiger_problem2, tiger_problem3];
DCs = [4, 1, 2];
SSF = Base.minimum;

# Create a MODIA, where all POMDP beliefs are initialized uniformly
modia = MODIA(DPs, DCs, SSF)
initialize_beliefs!(modia, BeliefUpdaters.uniform_belief);

# Compute optimal alpha vectors (policies) offline, and a define belief updater
solver = QMDPSolver();
policies = POMDPs.solve(solver, modia); 
bu = BeliefUpdaters.DiscreteUpdater(modia);

# Simulate the MODIA object for 5 timesteps, retrieve discounted rewards
sim = POMDPSimulators.RolloutSimulator(max_steps=5)
initial_states = convert(Array{Bool},(rand(initialstate(modia))))
r_totals = POMDPs.simulate(sim, modia, policies, bu, modia.beliefs, initial_states)
```

## Citations

If you have found the [MODIA architecture](https://www.ijcai.org/proceedings/2017/664) to be useful, consider citing the paper:
```
@inproceedings{ijcai2017-664,
  author    = {Kyle Hollins Wray and Stefan J. Witwicki and Shlomo Zilberstein},
  title     = {Online Decision-Making for Scalable Autonomous Systems},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {4768-4774},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/664},
  url       = {https://doi.org/10.24963/ijcai.2017/664},
} 
```