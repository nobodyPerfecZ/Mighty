# Mighty Examples

This is a collection of different ways to run Mighty. It's not a full documentation of all features or possibilities, but rather documenting different entry points into the code base. If you're interested how a full project built with Mighty looks like, check out our [example repository](https://github.com/automl/mighty_dr_example) using the [Syllabus curriculum learning library](https://github.com/RyanNavillus/Syllabus) to implement domain randomization with Mighty.
Here's an overview of example content:

- [Running Mighty from the Command Line](#running-mighty-from-the-command-line)
    - [The Runner File](#the-runner-file)
    - [Setting the Environment](#setting-the-environment)
    - [Configuring Agents](#configuring-agents)
    - [Meta Components](#meta-components)
    - [Running Multiple Seeds](#running-multiple-seeds)
- [Adding Custom Mighty Components](#adding-custom-mighty-components)
    - [The Metrics Dictionary](#the-metrics-dictionary)
    - [Priority Flexibility: Meta-Components](#priority-flexibility-meta-components)
    - [Building a Custom Component](#building-a-custom-component)
- [Hyperparameter Optimization Options](#hyperparameter-optimization-options)
    - [The Mighty ES Runner](#the-mighty-es-runner)
    - [Hydra Sweepers](#hydra-sweepers)
    - [Hypersweeper](#hypersweeper)
- [Logging \& Plotting](#logging--plotting)

## Running Mighty from the Command Line

Let's start with the basics, running a Mighty run from the command line! Mighty uses [Hydra](https://hydra.cc/) as a command line interface, so having basic familiarity with it will make your life easier.
These examples show you how to configure Mighty from the command line, if you're interested in how to build configuration files, check out [our short guide](./Building_mighty_configs.md).

#### The Runner File
<details>
  <summary>This is your basic Mighty runscript.</summary>
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
</details>

#### Setting the Environment
<details>
  <summary>How to switch environments using the command line. </summary>
For these examples, we'll directly work with 'run_mighty.py' and our pre-defined configs. First, we want to specify an environment to train on, e.g. CartPole-v1:

```bash
python mighty/run_mighty.py 'env=CartPole-v1'
```
We can also be more specific, e.g. by adding our desired number of interaction steps and the number of parallel environments we want to run:

```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=16'
```
For some environments, including CartPole-1, these details are pre-configured in the Mighty configs, meaning we can use the environment keyword to set them all at once:

```bash
python mighty/run_mighty.py 'environment=gymnasium/cartpole'
```
</details>

#### Configuring Agents
<details>
  <summary>Changing algorithms and their settings. </summary>
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
python mighty/run_mighty.py 'environment=gymnasium/cartpole' 'algorithm=dqn' 'algorithm_kwargs.policy_class=mighty.mighty_exploration.EZGreedy' 'algorithm_kwargs.policy_kwargs=null'
```
You can see that in this case, the value we pass to the script is a class name string which can take the value of any function you want, including custom ones as we'll see further down.
</details>

#### Meta Components
<details>
  <summary>Adding meta components. </summary>
The meta components are a bit more complex, since they are a list of class names and optional keyword arguments:

```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=16' '+algorithm_kwargs.meta_methods=[mighty.mighty_meta.RND]'
```
As this can become complex, we recommend configuring these in Hydra config files.
</details>

#### Running Multiple Seeds
<details>
  <summary>Conventiently sweeper of variations like seeds. </summary>
Hydra has a multirun functionality with which you can specify a grid of arguments that will automatically be run when appending '-m'. 
Its best use is probably for easily running multiple seeds at once like this:

```bash
python mighty/run_mighty.py 'env=CartPole-v1' 'num_steps=50_000' 'num_envs=16' 'seed=0,1,2,3,4' 'output_dir=examples/multiple_runs' -m 
```
</details>

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

#### The Metrics Dictionary
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

#### Priority Flexibility: Meta-Components
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

#### Building a Custom Component

The 'examples/custom_policy.py' file contains an example of a custom exploration policy and 'examples/custom_exploration_scheduler.py' contains and example of a meta module for epsilon scheduling.
Compare their structure: the custom policy has a fixed set of methods inherited form the abstract class while the meta module is free to choose the interaction time.

If you want to run these custom modules, you can do so by adding them by their import path:
```bash
python mighty/run_mighty.py 'algorithm=dqn' 'algorithm_kwargs.policy_class=examples.custom_policy.QValueUCB' 'algorithm_kwargs.policy_kwargs=null'
```
For the meta-module, it works exactly the same way:
```bash
python mighty/run_mighty.py 'algorithm=dqn' '+algorithm_kwargs.meta_methods=[examples.custom_exploration_scheduler.EpsilonSchedule]'
```

## Hyperparameter Optimization Options
Hyperparameter Optimization (HPO) is often essential for RL. Mighty comes with a few different options to take care of this step.

#### The Mighty ES Runner
You can do HPO with the ES runner within Mighty directly. This is usually not the most efficient way of doing things since all configurations will run sequentially, but you won't have to deal with external packages. You can select any evosax algorithm to optimize the hyperparameters of your choice and simply run it like any other Mighty run in the command line like this:
```bash
python mighty/run_mighty.py --config-name=cmaes_hpo
```

#### Hydra Sweepers
The simplest way to do parallel HPO is likely to use the sweepers installable via hydra. These include grid search (though that's not recommended!), optuna or Ax. Each has their own documentation of how to set up the search space and meta-parameters, so you should pick one and look them up individually. A nice feature of these runners is that they can parallelize runs on slurm and ray clusters as well as run locally.

We have prepared an example using Optuna, you can run it by:
```bash
uv pip install hydra-optuna-sweeper --upgrade
python mighty/run_mighty.py --config-path=../examples --config-name=optuna_example_config -m
```

#### Hypersweeper
We do our own HPO with [Hypersweeper](https://github.com/automl/hypersweeper), a package that integrates HPO packages from research. Similarly to the hydra sweepers, you should look into the Hypersweeper examples, though the HPO configurations in Mighty are already set up to directly use Hypersweeper, so it will run out of the box. If you want to run it on clusters, you will need to add a cluster config containing partitions etc.

As an example, you can run:

```bash
uv pip install hypersweeper
python mighty/run_mighty.py --config-path=../examples --config-name=hypersweeper_smac_example_config -m
```

## Logging & Plotting
We have an example notebook that shows you how to load and plot the default Mighty logs. Apart from these, you can also use Tensorboard or W&B for your plotting needs, though for these you should refer to their own documentations.
You can run this notebook to produce new runs for plotting, use randomly generated example data or load your own data into it.
For now these examples are focused on single-task learning instead of generalization or multi-task RL, but we plan on expanding them.
