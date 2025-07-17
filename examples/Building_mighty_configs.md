# Building Mighty Hydra Configs

Since Mighty pipelines can contain many different design decisions, we recommend using config files instead of long override sequences. 

## Configuration Basics

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

## Considerations for configuration stacking
So subconfigs are useful, but are there limitations? If you follow our suggested config structure, you'll notice that we use a superconfig per runner (even though this is not strictly necessary) and that for each sub-category, there is at least some overlap in config keys. Some of this can be split up more efficiently if you need it, for example you could define a 'network' category and explicitly reference to it in your algorithm_kwargs. 
Here the interaction and resolution between configs can become very complicated, however! 
A good example are the meta-components. They are part of the algorithm_kwargs, so they could be defined within the algorithm config (inefficient if we want to reuse this component with other algorithms) or defining them in a separate category. 
What do we do if we want to combine multiple meta-components, however? In this case, re-using two pre-defined configs and simply merging them will not work, since hydra will not merge the configs but they override each other instead. 
This should not be a problem in most cases, but for more complex configs, you should take a look at the hydra documentation to be extra sure config resolution works like you think it does!