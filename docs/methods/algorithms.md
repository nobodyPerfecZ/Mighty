# Mighty Algorithms
Mighty implements three basic online RL algorithm classes: DQN, SAC and PPO. These can be extended and altered in a few different ways without implementing a completely new algorithm from scratch. This page will give you an overview how they're structured, what components you can work with and when it's time to code up a new algorithm.

## How Algorithms Work

Mighty has a central 'base_agent.py' class which handles the environment interaction, logging and calls to updates and meta components.
Algorithms are built as subclasses of this agent and define the structure of its value and policy functions as well as any additional preprocessing that needs to happen for initialization or before updates.
All other algorithm components have their own classes which can easily be replaced. These are:

- models
- policies
- buffers
- updates

Combined, these four implement the majority of algorithm functionality. You can think of the DQN, SAC and PPO algorithm classes as defining an algorithm structure and the assosciated components implementing the actual interaction patterns. 
This also means that in most cases, you likely want to modify the algorithm components instead of the base class.

## Exploration Policies
Exploration policies are the interface between actions predicted and actions taken. 
The base Mighty exploration policy class takes greedy actions in evaluation mode and calls an 'explore' function in training. Defining a new 'explore' function thus allows you to implement a new exploration strategy.
Exploration policies are called at every step. 
Note that this means they are not the correct category for every exploration behavior! If your exploration strategy requires additional information or interaction beyond the action prediction, you should instead implement it in a meta component. 
An example is RND, which needs to be updated regularly in addition to action selection.

## Buffers
Buffers in Mighty work similarly than in other RL libraries. There are rollout and replay buffers, each of which implements and 'add' and a 'sample' function. 
Buffers have access to the full trajectories and metrics, which means you can implement a variety of priorization or re-sampling ideas.

## Updates
Just like exploration policies or buffers, updates are classes with fairly contained utility.
Their main function is to implement a function computing an update and also to apply it. 
Examples for update classes are DQN vs DDQN for Q-learning: the algorithm structure can stay the same and only the update class is varied. 
The base DQN update class implements 'apply_update', 'td_error' and 'get_targets' functions. 
Thus, to get to DDQN we define a subclass of this update that overrides 'get_targets':
```python
class DoubleQLearning(QLearning):
    """Double Q-learning update."""

    def __init__(
        self, model, gamma, optimizer=torch.optim.Adam, **optimizer_kwargs
    ) -> None:
        """Initialize the Double Q-learning update."""
        super().__init__(model, gamma, optimizer, **optimizer_kwargs)

    def get_targets(self, batch, q_net, target_net=None):
        if target_net is None:
            target_net = q_net
        argmax_a = (
            q_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .argmax(dim=1)
            .unsqueeze(-1)
        )
        max_next = target_net(
            torch.as_tensor(batch.next_obs, dtype=torch.float32)
        ).gather(1, argmax_a)
        targets = (
            batch.rewards.unsqueeze(-1)
            + (~batch.dones.unsqueeze(-1)) * self.gamma * max_next
        )
        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)
```
Each algorithm class requires slightly different update functions. 

## When Should I Implement A New Algorithm?
In general, you should try to avoid this simply because it will involve more code duplication an implementation of fundamental methods like saving and loading. 
The main factor for when you need a new algorithm class is likely a fundamental change in network structures and combinations.
The agent classes define how to save and load networks as well as intialize them. 
They also implement the unified interfact for accessing policies and value functions.
Compare DQN:
```python
@property
    def value_function(self) -> DQN:
        """Q-function."""
        return self.q  # type: ignore
```
To PPO:
```python
@property
    def value_function(self) -> torch.nn.Module:
        """Return the value function model."""
        return self.model.value_head  # type: ignore
```
So if you combine elements like value/Q-functions and policies in ways that don't fit the interfaces of SAC, PPO and DQN, you will have to write a new algorithm class.

A different reason for a new algorithm class is preprocessing that needs to happen before the update and can't be done within the update class. 
This is likely a rare case, but there is an example in PPO where we first need to do the return computation:
```python
def update(self, metrics: Dict, update_kwargs: Dict) -> Dict:
        if len(self.buffer) < self._learning_starts:  # type: ignore
            return {}

        # Compute returns and advantages for PPO
        last_values = self.value_function(
            torch.as_tensor(update_kwargs["next_s"], dtype=torch.float32)
        ).detach()

        self.buffer.compute_returns_and_advantage(last_values, update_kwargs["dones"])  # type: ignore
        return super().update(metrics, update_kwargs)  # type: ignore
```
If this is the only issue, we recommend only overwriting this method and implementing other functionality in more suitable classes.

## The Base Agent - Spicy Bits
The base agent class is by far the most complex element in Mighty. 
It's the hub for all activity and environment interaction as well as logging and plotting. 
This is why we recommend you be **very careful** when changing things on this level. 
Think carefully why you can't accomplish your task in another way.
If you believe you definitely need to touch the base agent, please also create an issue in the Mighty repo so we can improve the workflow!

Here are possible changes in increasing order of possible problems. 
Take this list as a guide to how much testing you should do for your additions:

1. Changes to logging and command line output: this should be fairly unproblematic, if maybe slightly annoying. We would still be interested what you're missing!
2. Changes to the evaluation: depending on what you change, this could be almost entirely independent of the rest of the loop. You should pay attention to maintaining the evaluation.
3. Changes to meta component call timings: this can break existing components if you change the timing of existing functions. Adding new hooks should be fine in most cases, however. This is likely the most interesting case for us, so we'd be happy to discuss about our hook timings.
4. Changes to the interaction: you should be careful here to maintain datatypes, shapes and counting. Also pay attention to the buffer interactions. This could easily break many parts of the loop, so be careful! 
5. Changes to the base agent update functionality: changes here will almost surely generate several issues with the specific agent, update and model classes. Usually updates should really only be changed in the downstream classes. If you need large updates, talk to us on GitHub, but Mighty might not be the library for you.