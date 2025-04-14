There are a few different ways you can use Mighty:

### Running Meta-Methods
This is the easiest part. We have several algorithms and meta-methods implemented in Mighty and you should be able to run them directly on any environment of your choosing. The most difficult part will likely be the configuration of each method since they might require specific keywords or are only compatible with a given base algorithm. So you will likely want to read up on whatever method you choose. Then you also need to know if your method is of the runner or meta module type. Each have their own configuration keyword. An example for using a specific runner is:

```bash
python mighty/run_mighty runner=es popsize=5 iterations=100 es=evosax.CMA_ES search_targets=["learning_rate", "_batch_size"] rl_train_agent=true
```
This will use the evosax CMA-ES implementation with population size 5 to optimize the learning rate and batch size in 100 iterations. Meta modules, on the other hand, use a different keyword:
```bash
python mighty/run_mighty.py +algorithm_kwargs.meta_methods=[mighty.mighty_meta.PrioritizedLevelReplay]
```
This meta methods list collects all meta modules in the order they should be used. So while you can't use multiple runners, you can use layers of meta modules. 

### Implementing New Components
Of course Mighty currently only supports a limited amount of methods. This is where you come in! It should be fairly easy for you to add your own. We recommend following these steps:
1. What are you adding? A runner, meta module, exploration policy, buffer, update variation or model? Make sure you choose the best level to implement your idea in.
2. Implement your method using the abstract class and existing methods as templates.
3. Plug your class into your Mighty config file. This works by replacing the default value with the import path of your custom class.
4. Run the algorithm.

Since you are passing the place from which to import your new class, you do not need to work within the Mighty codebase directly, but keep your changes separate. This way you can add several new methods to Mighty without copying the code. 

### Combining Different Ideas
You can combine different approaches with Mighty by varying the runner, exploration, buffer, update class and network architecture and combining them with an arbitrary number of meta modules.
At this point, configuration might become very difficult. We recommend that you take a close look at how to use different hydra configuration files to separately configure each of your methods so that you can keep track of everything.
Depending on what exactly you want to do, it can make sense to keep separate configuration files for each variation you make. This can be confusing, especially if you haven't worked with hydra before, so we recommed you take the time to focus on configurations when attempting combinations of several methods.