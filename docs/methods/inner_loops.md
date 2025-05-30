# Mighty Inner Loops
A key motivation for Mighty is to make it easy to create systems that interact with the RL loop. If these systems work during an algorithm's runtime, they are inner loop components. In Mighty, we call them Meta Components. This page documents their structure and why they're so useful.

## What Are Meta Components?
Meta components are elements interacting with the main loop at various points. 
Within this interaction, they have access to virtually all current internal information and can adapt it.
This means everything from hyperparameter scheduling to learning a separate dynamics models and more is possible within this structure. 
Meta components can be stacked on top of one another to combine different approaches in a single run.
This enables complex inner loop setups without entangling methods in code.

## The Metrics Dictionary
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

## Interactions With The Main Loop
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

## Combining Components
When combining different modules, they are stacked on top of one another. 
This means they are executed in order for each method. 
For meta components interacting with each other or the same parts of the base loop, this order can be important!
If you, for example, use a curriculum based on training reward and intrinsic reward, you should likely configure the curriculum to be called first to avoid basing the difficulty on the reward bonus.