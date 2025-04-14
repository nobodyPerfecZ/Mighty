### What Is Contextual RL?

Most RL environments are either not concerned with generalization at all or test generalization performance without providing much insight into what agents are tested on,
e.g. by using procedurally generated levels that are hard to understand as a structured training or test distribution.
Contextual RL (or cRL)[[Hallak et al., CoRR 2015](https://arxiv.org/pdf/1502.02259.pdf), [Benjamins et al., CoRR 2022](https://arxiv.org/pdf/2202.04500.pdf)] aims to make the task distributions agents are trained on a specific as possible in order to gain better insights where agents perform well and what is currently missing in RL generalization.

### Contextual RL With CARL

[CARL (context adaptive RL)](https://github.com/automl/CARL) (see [Benjamins et al., EcoRL 2021](<https://arxiv.org/pdf/2110.02102.pdf>) for more information) is a benchmark library specifically designed for contextual RL.
It provides highly configurable contextual extensions to several well-known RL environments and is what we recommend to get started in cRL.
Mighty is designed with contextual RL in mind and therefore fully compatible with CARL.

The training works similarly to a standard RL environment, but now you can specify the training and test distributions, for example 10 variations of gravity for CartPole from CARL's default distribution:

```bash
python mighty/run_mighty.py 'algorithm=dqn' 'env=CARLCartPoleEnv' '+env_kwargs.num_contexts=10' '+env_kwargs.context_feature_args=[gravity]'
```

Other CARL options are supported similarly as env_kwargs, though we recommend checking out the CARL examples to get a better idea of how to define these distributions.