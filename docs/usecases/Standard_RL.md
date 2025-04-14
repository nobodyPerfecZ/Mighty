If you want to use Mighty on standard RL environments, you can choose between the [Gymnasium](https://gymnasium.farama.org/) interface and [Pufferlib](https://puffer.ai/) as a fast alternative. 
Generally we recommend you use Pufferlib where possible, but the choice is yours!

### Mighty on Gymnasium Environments

Mighty can be used as a standard RL library for all environments that follow the Gymnasium interface.
In order to run a Mighty Agent, use the run_mighty.py script and provide any training options as keywords. If you want to know more about the configuration options, call:

```bash
python mighty/run_mighty.py --help
```

An example for running the PPO agent on the Pendulum gym environment for 1000 steps looks like this:

```bash
python mighty/run_mighty.py 'num_steps=1000' 'algorithm=ppo' 'env=Pendulum-v1'
```

We assume that if you don't specify anything beyond the name, you want to use Gymnasium. This will also work for environments that are registered with Gymnasium upon installation, e.g. [Gymnasium Robotics](https://robotics.farama.org/) and others.
You can assume specifying the environment name like this to work just like "gym.make()".

### Mighty on Pufferlib Environments
Pufferlib offers an efficient way to parallelize environment evaluations for a wide selection of tasks. Many well-known Gymnasium environments or benchmarks like [ProcGen]() are included in Pufferlib and we recommend it as a default for these.
Running Pufferlib environments is very similar to running Gymnasium environments, you only need to add the pufferlib domain:

```bash
python mighty/run_mighty.py 'num_steps=1000' 'algorithm=ppo' 'env=pufferlib.environments.procgen.bigfish'
```

We have some example configs where the env domain is pre-configured and you can override the name only. An example for minigrid would be:

```yaml
env_name: MiniGrid-DoorKey-8x8-v0    # Overide with names z.B MiniGrid-LavaGapS5-v0, MiniGrid-DoorKey-8x8-v0, MiniGrid-ObstructedMaze-1Dl-v0, MiniGrid-KeyCorridorS3R2-v0, MiniGrid-UnlockPickup-v0
env: pufferlib.environments.minigrid.${env_name}
env_kwargs: {}
```

Meaning you can use this configuration (let's call it pufferlib_minigrid) with just the env name, similar to above:
```bash
python mighty/run_mighty.py 'num_steps=1000' 'algorithm=ppo' 'env=pufferlib_minigrid' 'env_name=MiniGrid-LavaGapS5-v0'
```