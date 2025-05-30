# Mighty Examples

This is a collection of different ways to run Mighty. It's not a full documentation of all features or possibilities, but rather documenting different entry points into the code base.
Here's an overview of example content:

- [Mighty Examples](#mighty-examples)
  - [Running Mighty from the Command Line](#running-mighty-from-the-command-line)
    - [The Runner File](#the-runner-file)
    - [Setting the Environment](#setting-the-environment)
    - [Configuring Agents](#configuring-agents)
    - [Meta Components](#meta-components)
    - [Running Multiple Seeds](#running-multiple-seeds)
  - [Building Mighty Hydra Configs](#building-mighty-hydra-configs)
    - [Configuration Basics](#configuration-basics)
    - [Considerations for configuration stacking](#considerations-for-configuration-stacking)
  - [Hyperparameter Optimization Options](#hyperparameter-optimization-options)
    - [Hydra Sweepers](#hydra-sweepers)
    - [Hypersweeper](#hypersweeper)
    - [ES Runner](#es-runner)
  - [Adding Custom Mighty Components](#adding-custom-mighty-components)
    - [The Metrics Dictionary](#the-metrics-dictionary)
    - [Priority Flexibility: Meta-Components](#priority-flexibility-meta-components)
    - [Building a Custom Component](#building-a-custom-component)
  - [Logging \& Plotting](#logging--plotting)


## Running Mighty from the Command Line

Let's start with the basics, running a Mighty run from the command line! Mighty uses [Hydra](https://hydra.cc/) as a command line interface, so having basic familiarity with it will make your life easier.

### The Runner File

Your central script will look very similar to the 'run_mighty.py' file we provide:

```python
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from mighty.mighty_runners.factory import get_runner_class


@hydra.main("./configs", "base", version_base=None)
def run_mighty(cfg: DictConfig) -> None:
    # Make runner
    runner_cls = get_runner_class(cfg.runner)
    runner = runner_cls(cfg)

    # Execute run
    start = time.time()
    train_result, eval_result = runner.run()
    end = time.time()

    # Print stats
    print("Training finished!")
    print(
        f"Reached a reward of {np.round(eval_result['mean_eval_reward'], decimals=2)} in {train_result['step']} steps and {np.round(end - start, decimals=2)}s."
    )


if __name__ == "__main__":
    run_mighty()
```

### Setting the Environment

For these examples, we'll directly work with 'run_mighty.py' and our pre-defined configs. First, we want to specify an environment to train on, e.g. CartPole-v1:
```bash
python mighty/run_mighty.py 'env=CartPole-v1'
```
We can also be more specific, e.g. by adding our desired number of interaction steps and the number of parallel environments we want to run:
```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=10'
```
For some environments, including CartPole-1, these details are pre-configured in the Mighty configs, meaning we can use the environment keyword to set them all at once:
```bash
python mighty/run_mighty.py 'environment=gymnasium/cartpole'
```
### Configuring Agents
Overriding algorithms works very similarly, we can change from PPO to DQN by running:
```bash
python mighty/run_mighty.py 'environment=gymnasium/cartpole' 'algorithm=dqn'
```
Algorithms have pre-configured algorithm arguments like the learning rate or type of policy they use. These overrides work the same as the ones we have seem so far:
```bash
python mighty/run_mighty.py 'environment=gymnasium/cartpole' 'algorithm=dqn' 'algorithm_kwargs.learning_rate=0.1'
```
Or to use e.g. an ez-greedy exploration policy for DQN:
```bash
python mighty/run_mighty.py 'environment=gymnasium/cartpole' 'algorithm=dqn' '+algorithm_kwargs.policy_class=mighty.mighty_exploration.EZGreedy'
```
You can see that in this case, the value we pass to the script is a class name string which can take the value of any function you want, including custom ones as we'll see further down.

### Meta Components
The meta components are a bit more complex, since they are a list of class names and optional keyword arguments:
```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=10' '+algorithm_kwargs.meta_methods=[mighty.mighty_meta.RND]'
```
As this can become complex, we recommend configuring these in Hydra config files.

### Running Multiple Seeds
Hydra has a multirun functionality with which you can specify a grid of arguments that will automatically be run when appending '-m'. 
Its best use is probably for easily running multiple seeds at once like this:

```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=10' 'seed=0,1,2,3,4' 'output_dir=examples/multiple_runs' -m 
```

## Building Mighty Hydra Configs

Since Mighty pipelines can contain many different design decisions, we recommend using config files instead of long override sequences. 

### Configuration Basics

Our own Mighty base config file looks like this:
```yaml
defaults:
  - _self_
  - algorithm: dqn
  - environment: pufferlib_ocean/bandit
  - search_space: dqn_gym_classic
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
  - override hydra/help: mighty_help

runner: standard
debug: false
seed: 0
output_dir: runs
wandb_project: null
tensorboard_file: null
experiment_name: mighty_experiment

algorithm_kwargs: {}

# Training
eval_every_n_steps: 1e4  # After how many steps to evaluate.
n_episodes_eval: 10
checkpoint: null  # Path to load model checkpoint
save_model_every_n_steps: 5e5

hydra:
  run:
    dir: ${output_dir}/${experiment_name}_${seed}
  sweep:
    dir: ${output_dir}/${experiment_name}_${seed}
```
Note that much of the relevant information actually comes from the defaults list above. 
This config file, for example, does not specify any algorithm_kwargs, these are loaded from the algorithm config, in this case 'dqn'.

Some of these defaults are not strictly necessary. The hydra 'job_logging', 'hydra_logging' and 'help' keywords are only for usability and you won't need the search space except for working with hyperparameter optimization.
What's important for you to know are:
- the algorithm configuration
- the environment configuration
- the runner type
- the seed
- logging settings like experiment_name, output dir and optional tensorboard/wandb projects

This will be the basic information for your runs. Let's take a look at the algorithm config next:
```yaml
# @package _global_
algorithm: DQN
q_func: ???

algorithm_kwargs:
  # Hyperparameters
  n_units: 8
  epsilon: 0.2  # Controls epsilon-greedy action selection in policy.

  replay_buffer_class:
    _target_: mighty.mighty_replay.PrioritizedReplay #Using prioritized experience replay
  replay_buffer_kwargs:
    capacity: 1000000  # Maximum size of replay buffer.
    alpha: 0.6

  # Training
  learning_rate: 0.001
  batch_size: 64  # Batch size for training.
  gamma: 0.9  # The amount by which to discount future rewards.
#  begin_updating_weights: 1  # Begin updating policy weights after this many observed transitions.
  soft_update_weight: 1.  # If we set :math:`\tau=1` we do a hard update. If we pick a smaller value, we do a smooth update.
  td_update_class: mighty.mighty_update.QLearning #Simple Q-learning update instead of default DDQN
  q_kwargs:
    dueling: False
    feature_extractor_kwargs:
      architecture: mlp
      n_layers: 1
      hidden_sizes: [32]
    head_kwargs:
      hidden_sizes: [32]
```
We see there's a '@package _global_' marker here to indicate that these keywords will be accessible at the top level of the full configuration.
The main part here is taken up by the 'algorithm_kwargs' which contain all the information needed to instatiate and run the algorithm (hyperparameter, architecture, policy, buffer, etc.).

The environment config will usually be a lot shorter, like here for MiniGrid via Pufferlib:
```yaml
# @package _global_

num_steps: 2e5
env_name: MiniGrid-DoorKey-8x8-v0    # Overide with names z.B MiniGrid-LavaGapS5-v0, MiniGrid-DoorKey-8x8-v0, MiniGrid-ObstructedMaze-1Dl-v0, MiniGrid-KeyCorridorS3R2-v0, MiniGrid-UnlockPickup-v0
env: pufferlib.environments.minigrid.${env_name}
env_kwargs: {}
env_wrappers: []
num_envs: 64
```
Here you can also see that you can use the keys to make configs more efficient. Using this config enables us to override the env_name only without having to write out the full env path all the time.

### Considerations for configuration stacking
So subconfigs are useful, but are there limitations? If you follow our suggested config structure, you'll notice that we use a superconfig per runner (even though this is not strictly necessary) and that for each sub-category, there is at least some overlap in config keys. Some of this can be split up more efficiently if you need it, for example you could define a 'network' category and explicitly reference to it in your algorithm_kwargs. 
Here the interaction and resolution between configs can become very complicated, however! 
A good example are the meta-components. They are part of the algorithm_kwargs, so they could be defined within the algorithm config (inefficient if we want to reuse this component with other algorithms) or defining them in a separate category. 
What do we do if we want to combine multiple meta-components, however? In this case, re-using two pre-defined configs and simply merging them will not work, since hydra will not merge the configs but they override each other instead. 
This should not be a problem in most cases, but for more complex configs, you should take a look at the hydra documentation to be extra sure config resolution works like you think it does!

## Hyperparameter Optimization Options
Hyperparameter Optimization (HPO) is often essential for RL. Mighty comes with a few different options to take care of this step.

### Hydra Sweepers
The simplest way to do hyperparameter optimization is likely to use the sweepers installable via hydra. These include grid search (though that's not recommended!), optuna or Ax. Each has their own documentation of how to set up the search space and meta-parameters, so you should pick one and look them up individually. A nice feature of these runners is that they can parallelize runs on slurm and ray clusters as well as run locally.

We have prepared an example using Optuna, you can run it by:


### Hypersweeper
We do our own HPO with [Hypersweeper](https://github.com/automl/hypersweeper), a package that integrates HPO packages from research. Similarly to the hydra sweepers, you should look into the Hypersweeper examples, though the HPO configurations in Mighty are already set up to directly use Hypersweeper, so it will run out of the box. If you want to run it on clusters, you will need to add a cluster config containing partitions etc.

As an example, you can run:
```bash
uv pip install hydra-optuna-sweeper --upgrade
python mighty/run_mighty.py --config-path=../examples --config-name=optuna_example_config -m
```

### ES Runner
You can also do HPO with the ES runner within Mighty. This is usually not the most efficient way of doing things (there are also ES in Hypersweeper which will parallelize), but you won't have to deal with external packages. You can select any evosax algorithm to optimize the hyperparameters of your choice and simply run it like any other Mighty run in the command line like this:
```bash
uv pip install hypersweeper
python mighty/run_mighty.py --config-path=../examples --config-name=hypersweeper_smac_example_config -m
```

## Adding Custom Mighty Components
You can add your own components to Mighty without touching the core loop. Generally, this makes sense for the following components:
- replay or rollout buffers
- network structures
- updates or loss functions
- action selection policies
- in-the-loop meta-approaches (e.g. anything related to state novelty or task scheduling)
- outer-loop algorithms (e.g. meta-learning or black-box optimization)

You can also add full algorithms, but this will likely be more involved since there are quite a few dependencies in the main algorithm classes.
The other components are designed to be more contained. The API documentation should tell you how all of them interact with learning and what kind of functions they need to implement.

### The Metrics Dictionary
The most important part when adding components is the 'metrics' dictionary. This is Mighty's central information hub. 
Here you can find transitions, losses, predictions, batches and parameters - everything you need to build methods that actively work with the RL loop.
If you want examples of how it is used, you can check out our RND implementation:
```python
    def get_reward(self, metrics):
        """Adapt LR on step.

        :param metrics: Dict of current metrics
        :return:
        """
        if self.rnd_net is None:
            self.initialize_networks(metrics["transition"]["next_state"].shape[1:])

        rnd_error = self.rnd_net.get_error(metrics["transition"]["next_state"])
        metrics["transition"]["intrinsic_reward"] = (
            self.internal_reward_weight * rnd_error
        )
        metrics["transition"]["reward"] = (
            metrics["transition"]["reward"] + self.internal_reward_weight * rnd_error
        )
        return metrics
```
Here we read the next state from the metrics dictionary, predict state novelty from it and update the transition reward.
We also add a new intrinsic reward key to enable logging.
You can assume that most if not all relevant information is contained in the metrics dictionary at any given time. 
It is also transmitted to many different Mighty components like the exploration policy, the buffer, the update function or to any meta-components.

### Priority Flexibility: Meta-Components
Meta-components are classes with methods that can be called at different points in the learning loop. There are several different call positions and they are specified by the component itself:
```python
    def __init__(self) -> None:
        """Meta module init.

        :return:
        """
        self.pre_step_methods = []
        self.post_step_methods = []
        self.pre_update_methods = []
        self.post_update_methods = []
        self.pre_episode_methods = []
        self.post_episode_methods = []
```
Each of these calls will receive the metrics dictionary, resulting in a very flexible type. 
Right now Mighty contains a few meta-components doing very different things, e.g.:
- task scheduling/curriculum learning
- hyperparameter scheduling
- intrinsic rewards
Meta-components are also stackable, i.e. you can run multiple ones per training run. 
In principle, you can do almost anything in a meta-component, including training additional networks or calling the policy directly.
Before you default to using this class, however, we recommend double checking if your idea isn't better suited to a more specific class.

### Building a Custom Component

The 'examples/custom_policy.py' file contains an example of a custom exploration policy and 'examples/custom_exploration_scheduler.py' contains and example of a meta module for epsilon scheduling.
Compare their structure: the custom policy has a fixed set of methods inherited form the abstract class while the meta module is free to choose the interaction time.

If you want to run these custom modules, you can do so by adding them by their import path:
```bash
python mighty/run_mighty.py 'algorithm=dqn' '+algorithm_kwargs.policy_class=examples.custom_policy.QValueUCB' '+algorithm_kwargs.policy_kwargs={}'
```
For the meta-module, it works exactly the same way:
```bash
python mighty/run_mighty.py 'algorithm=dqn' '+algorithm_kwargs.meta_methods=[examples.custom_exploration_scheduler.EpsilonSchedule]'
```

## Logging & Plotting
We have an example notebook that shows you how to load and plot the default Mighty logs. Apart from these, you can also use Tensorboard or W&B for your plotting needs, though for these you should refer to their own documentations.
You can run this notebook to produce new runs for plotting, use randomly generated example data or load your own data into it.
For now these examples are focused on single-task learning instead of generalization or multi-task RL, but we plan on expanding them.
