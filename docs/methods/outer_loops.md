# Mighty Outer Loops
Methods that interact with repeated runs of RL algorithms are our Mighty runners. These function a level above the standard RL training to modify the inner loop. On this page, you'll find information on their structure and what kind of usecases they cover.

## Runners
Runners are a wrapper class around the agent and can interact with the full task spectrum, i.e. adapt agent and environment and run this combination for an arbitrary amount of steps.
The very basic online runner simply executes a task and evaluates the resulting policy:
```python
class MightyOnlineRunner(MightyRunner):
    def run(self) -> Tuple[Dict, Dict]:
        train_results = self.train(self.num_steps)
        eval_results = self.evaluate()
        return train_results, eval_results
``` 
The ES runner, on the other hand, has a considerably longer 'run' function including multiple calls to versions of the agent:
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
Conceptually, you should think of runners creating new RL tasks, that is combinations of environment and agent, to achieve some goal. 
This can be meta-learning, hyperparameter optimization and more.

## Information Flow
Runners don't interact with the inner loop directly, but primarily via the agent class interface.
Running and evaluation the agent are the two most important function calls, but runners can also utilize the update and access buffers, environments, parameters and more. 
Thus, the information can be performance as well as much of the algorithm state after execution.
Notably, runners can also access meta components, enabling hybrid approaches inner loops that span multiple outer loops.