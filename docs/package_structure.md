Mighty is designed to be highly modular, enabling access to the RL loop on different levels. This means it's not designed to be the absolute fastest way to run RL, but the most convenient one to apply different sorts of RL, MetaRL and AutoRL methods. As such, there are a few things you should know about the structure of Mighty. 

### Quickstart
If you only want to know which class category to choose for implementing your method, follow this handy diagram:

``` mermaid
stateDiagram
  direction TB
  classDef Peach stroke-width:1px,stroke-dasharray:none,stroke:#FBB35A,fill:#FFEFDB,color:#8F632D;
  classDef Aqua stroke-width:1px,stroke-dasharray:none,stroke:#46EDC8,fill:#DEFFF8,color:#378E7A;
  classDef Sky stroke-width:1px,stroke-dasharray:none,stroke:#374D7C,fill:#E2EBFF,color:#374D7C;
  classDef Pine stroke-width:1px,stroke-dasharray:none,stroke:#254336,fill:#8faea5,color:#FFFFFF;
  classDef Rose stroke-width:1px,stroke-dasharray:none,stroke:#FF5978,fill:#FFDFE5,color:#8E2236;
  classDef Ash stroke-width:1px,stroke-dasharray:none,stroke:#999999,fill:#EEEEEE,color:#000000;
  classDef Seven fill:#E1BEE7,color:#D50000,stroke:#AA00FF;
  Still --> root_end:Yes
  Still --> Moving:No
  Moving --> Crash:Yes
  Moving --> s2:No, only current transitions, env and network
  s2 --> s6:Action Sampling
  s2 --> s10:Policy Update
  s2 --> s8:Training Batch Sampling
  s2 --> Crash:More than one/not listed
  s2 --> s12:Direct algorithm change
  s12 --> s13:Yes
  s12 --> s14:No
  Still:Modify training settings and then repeated runs?
  root_end:Runner
  Moving:Access to update infos (gradients, batches, etc.)?
  Crash:Meta Component
  s2:Which interaction point with the algorithm?
  s6:Exploration Policy
  s10:Update
  s8:Buffer
  s12:Change only the model architecture?
  s13:Network and/or Model
  s14:Agent
  class root_end Peach
  class Crash Aqua
  class s6 Sky
  class s8 Pine
  class s10 Rose
  class s13 Ash
  class s14 Seven
  style root_end color:none
  style s8 color:#FFFFFF
```

### For Multiple Inner Runs: Mighty Runners
Mighty uses runner classes to control the outer training loop. In the simplest case, a runner will just directly call the agent's train and evaluation functions without any changes:

```python
def run(self) -> Tuple[Dict, Dict]:
        train_results = self.train(self.num_steps)
        eval_results = self.evaluate()
        return train_results, eval_results
```
This will result in a standard RL agent training run. Of course, we can at this point also run agents multiple times, make changes to their setup (hyperparameters, weights, environments) and integrate learning on this meta-level.
A still fairly simple example is our ESRunner for outer loops with Evolutionary Strategies:

```python
def run(self) -> Tuple[Dict, Dict]:
        es_state = self.es.initialize(self.rng)
        for _ in range(self.iterations):
            rng_ask, _ = jax.random.split(self.rng, 2)
            x, es_state = self.es.ask(rng_ask, es_state)
            eval_rewards = []

            for individual in x:
                if self.search_params:
                    self.apply_parameters(individual[: self.total_n_params])
                    individual = individual[self.total_n_params :]

                for i, target in enumerate(self.search_targets):
                    if target == "parameters":
                        continue
                    new_value = np.asarray(individual[i]).item()
                    if target in ["_batch_size", "n_units"]:
                        new_value = max(0, int(new_value))
                    setattr(self.agent, target, new_value)

                if self.train_agent:
                    self.train(self.num_steps_per_iteration)

                eval_results = self.evaluate()
                eval_rewards.append(eval_results["mean_eval_reward"])

            fitness = self.fit_shaper.apply(x, jnp.array(eval_rewards))
            es_state = self.es.tell(x, fitness, es_state)

        eval_results = self.evaluate()
        return {"step": self.iterations}, eval_results
```
Here we can change all sorts of things about the agent, train in between or only evaluate and use the ES to get fresh inputs. Runner classes are defined with these multiple evaluations of RL tasks in mind, i.e. these classes will usually train multiple agents, reset their policies completely or otherwise start over at some point. 

### For In-The-Loop Methods: Mighty Meta Modules

Not all Meta- or AutoRL methods operate in an outer loop, however. For the ones that configure training while it is still ongoing, we use the Mighty Meta Modules. 
These are classes that maintain lists of function calls to make at different points in training:

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
This gives meta modules a lot of flexibility of when to act upon training. Additionally, each of these function calls is given a "metrics" dictionary. This dictionary contains most, if not all, relevant information about training progress, e.g.:

- the last transitions
- the last losses, errors and predictions
- policy, Q- and value-networks
- hyperparameters

This means meta modules can use everything from the current timestep to agent predictions. 


### Algorithm Components: Mighty Exploration, Buffers and Updates

The Mighty algorithms themselves also have modules which can be easily switched. These are exploration policies, buffers and update classes. 
Exploration policies and buffers furthermore have access to the same metrics dictionary as meta modules, meaning you can get creative as to what they do with this information.
The way they are used in the RL loop is fixed, however, such that these are a bit more streamlined than the completely free meta-modules.


### Inside the Agent: Mighty Models

Agent loops outside of exploration, buffers and updates are harder to alter in Mighty, since Mighty is primarily focused on meta-methods.
You can control the network architecture of your agent fairly easily, however. 
There are two principal avenues for this: 

1. You can use one of the pre-defined Mighty Models and configure it to use a different network architecture in the config. We use torch internally, that means you can allocate torch.nn layers and activations in different parts of these networks to form a custom architecture.
2. If you also want to customize what exactly the network predicts or add things like frozen weights, you probably want to implement your own Mighty Model. These always contain a 'feature_extractor' as a base and can vary beyond that.