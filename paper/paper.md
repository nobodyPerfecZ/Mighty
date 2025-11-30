---
title: 'Mighty: A Comprehensive Tool for studying Generalization, Meta-RL and AutoRL'
tags:
  - Python
  - reinforcement learning
  - contextual reinforcement learning
  - meta-learning
  - generalization
  - automated machine learning
authors:
  - name: Aditya Mohan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Theresa Eimer
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Carolin Benjamins
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Marius Lindauer
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3"
  - name: Andre Biedenkapp
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: Leibniz University Hannover, Germany
   index: 1
 - name: University of Freiburg, Germany
   index: 2
 - name: L3S Research Center, Germany
   index: 3
date: 30 November 2024
bibliography: paper.bib
---

## Summary

Robust generalization, rapid adaptation, and automated tuning are critical for deploying reinforcement learning (RL) in real-world settings. Yet research in these areas remains fragmented across non-standard codebases and custom orchestration scripts. We introduce *Mighty*, an open-source library that unifies contextual generalization, Meta-RL, and AutoRL within a single modular interface. Mighty cleanly separates the *Agent* from a configurable environment modeled as a *Contextual MDP*, decoupling *inner-loop* updates from *outer-loop* adaptations. This enables unified support for: (i) contextual generalization and curriculum learning (e.g., unsupervised environment design), (ii) bi-level meta-learning (e.g., MAML, black-box strategies), and (iii) automated hyperparameter and architecture search (e.g. Bayesian optimization, evolutionary strategies, population‐based training). We outline Mighty's design and validate its implementation on standard RL benchmarks. By offering a unified modular platform, Mighty simplifies experimentation and accelerates research on robust, adaptable RL.

## Statement of need

Reinforcement learning (RL) has emerged as a powerful decision-making paradigm in complex and dynamic environments. Despite impressive successes in domains such as games and robotics, RL algorithms frequently overfit their training conditions and struggle to generalize to new tasks. Addressing this challenge requires methods that not only learn efficiently on a single task but also adapt rapidly to novel settings and automatically tune their learning process.

Recent research has advanced in three complementary directions: (i) Generalization in RL, (ii) Meta-RL methods, and (iii) Automated RL (AutoRL). Although each has led to promising algorithms, researchers frequently resort to fragmented codebases and ad hoc scripting across environment design, RL training, and meta-optimization. This fragmentation increases engineering effort, impedes rapid iteration, and undermines reproducibility.

We introduce *Mighty*: a modular library designed to enable research at the intersection of generalization, Meta-RL, and AutoRL. Mighty enforces a clean and principled separation between inner- and outer-loop processes, making it easy to combine, for example, curricula, context adaptation, and automated tuning within a unified framework. Users can prototype new methods, compose existing ones, and run controlled comparisons - all without ad hoc orchestration code.

![Overview of Mighty's concept and modules.](Figures/mighty_concept.pdf)

## Key Features

Mighty is designed around three design principles: *flexibility, smooth integration with existing libraries, and environment parallelization*.

**User Interface:** Mighty prioritizes usability and flexibility. We use Hydra for structured configuration files that expose all relevant training details without overwhelming new users. This also plugs Mighty into Hydra's ecosystem for cluster execution and hyperparameter optimization. The algorithm components in Mighty are modular and can be replaced via configurations, allowing users to integrate new components without editing the training loop.

**Agent Framework:** Mighty includes three base RL algorithms -- DQN, SAC and PPO -- built from four modular components: exploration policy, replay buffer, update function, and model parameterization. Each component is easily extendable, allowing users to swap in new methods without rewriting the entire algorithm or touching the training loop.

**Meta-Learning Framework:** Mighty's support for meta-methods is unique in the RL landscape. It offers two key abstractions: *runners* and *meta-components*. Runners control training lifecycles, interacting with agents and environments while accessing artifacts like performance metrics and policy weights. Meta-components operate within a single run, with access to six hook points and full training context.

**Currently Implemented Methods:** Mighty comes with several built-in options including ε-greedy exploration, prioritized replay buffer, and DDQN update. The meta-components show online interactions with hyperparameters (cosine annealing), transitions (RND and NovelD) and contextual environments (PLR and SPaCE).

## Usage Example

```python
# Train a PPO agent on a contextual environment
python mighty/run_mighty.py 'algorithm=ppo' 'environment=carl/cartpole' \
    '+env_kwargs.num_contexts=10' \
    '+algorithm_kwargs.meta_methods=[mighty.mighty_meta.RND]'

# Run hyperparameter optimization with SMAC
python mighty/run_mighty.py --config-name=hypersweeper_smac_example_config -m
```

## Empirical Validation

We validate our implementations by comparing them with OpenRL benchmark results. Our aim is not to outperform existing baselines, but to demonstrate that Mighty achieves comparable performance at similar training budgets.

| Algorithm | Environment | Steps | Time (min) | Final Return | OpenRL Return |
|-----------|-------------|-------|------------|--------------|---------------|
| DQN | MountainCar | 5×10⁵ | 51.1 | -200.00 ± 0.00 | -189.92 ± 11.00 |
| DQN | CartPole | 5×10⁵ | 60.41 | 486.40 ± 30.77 | 499.92 ± 0.00 |
| PPO | MountainCar | 5×10⁵ | 3.03 | -200.00 ± 0.00 | -200.00 ± 0.00 |
| PPO | CartPole | 5×10⁵ | 3.67 | 479.80 ± 17.21 | 487.48 ± 6.79 |
| SAC | Walker2D | 10⁶ | 353.13 | 4478.67 ± 689.22 | 4471.15 ± 1896.34 |
| SAC | HalfCheetah | 10⁶ | 302.53 | 10588.34 ± 874.19 | 10958.60 ± 1335.62 |

The trends broadly align: PPO and DQN on CartPole closely track OpenRL, and SAC in Walker2D and HalfCheetah remains close to the mean performance reported by OpenRL. The results demonstrate that Mighty's implementations reproduce the results of established baselines, both in sample efficiency and runtime.

## Acknowledgements

We acknowledge contributions from the AutoML community and thank the developers of CARL, DACBench, and other integrated frameworks that make Mighty's unified interface possible.

## References
