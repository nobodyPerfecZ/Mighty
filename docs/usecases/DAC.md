### What Is Dynamic Algorithm Configuration?
Dynamic Algorithm Configuration (DAC) [[Biedenkapp et al., ECAI 2020](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-ECAI-DAC.pdf), [Adriaensen et al., JAIR 2022](https://arxiv.org/pdf/2205.13881.pdf)]
is a hyperparameter optimization paradigm aiming to find the best possible hyperparameter configuration for a given *algorithm instance* at every *timestep* during runtime.
DAC can easily be modelled as a contextual MDP and is thus a real-world application of RL.


### Dynamic Algorithm Configuration with Mighty
In order to interface with configurable algorithms, we recommend [DACBench](https://github.com/automl/DACBench).
It provides algorithms from different fields as well as artificial benchmarks, all with the OpenAI gym interface.

Select the benchmark you want to run, for example the SigmoidBenchmark, and providing it as the "env" keyword:

```bash
python mighty/run_mighty.py 'algorithm=dqn' 'env=SigmoidBenchmark'
```
The naming here will make Mighty autodetect it as a DACBench environment.

The benchmarks in DACBench have many configuration options. You can use your hydra configs to include your changes, simply use the keyword TDO

```bash
python run_mighty.py TODO
```

Of course you can also load existing config files, e.g. to reproduce another experiment:

```bash
python run_mighty.py TODO
```

