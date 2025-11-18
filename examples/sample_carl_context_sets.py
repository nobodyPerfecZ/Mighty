import carl
from carl.context.sampler import ContextSampler
from carl.envs import CARLCartPole

import json
from pathlib import Path

def sample_context_sets():
    # Define the context set with different gravity and length values

    # Mode A: Full range of values
    gravity_dist = carl.context.context_space.UniformFloatContextFeature(
                    "gravity",
                    lower=7.0,
                    upper=10.0,
                )
    length_dist = carl.context.context_space.UniformFloatContextFeature(
                    "length",
                    lower=0.3,
                    upper=0.7,
                )

    context_sampler = ContextSampler(
            [gravity_dist, length_dist],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_a = context_sampler.sample_contexts(10)
    context_sampler.seed(123)
    eval_contexts_a = context_sampler.sample_contexts(100)

    # Mode B: Restrict intersection
    gravity_dist1 = carl.context.context_space.UniformFloatContextFeature(
                    "gravity",
                    lower=7.0,
                    upper=8.5,
                )
    length_dist1 = carl.context.context_space.UniformFloatContextFeature(
                    "length",
                    lower=0.3,
                    upper=0.5,
                )
    
    gravity_dist2 = carl.context.context_space.UniformFloatContextFeature(
                    "gravity",
                    lower=8.5,
                    upper=10.0,
                )
    length_dist2 = carl.context.context_space.UniformFloatContextFeature(
                    "length",
                    lower=0.5,
                    upper=0.7,
                )
    
    gravity_constant = carl.context.context_space.UniformFloatContextFeature(
                    "gravity",
                    lower=7.0,
                    upper=7.01,
                )
    length_constant = carl.context.context_space.UniformFloatContextFeature(
                    "length",
                    lower=0.3,
                    upper=0.31,
                )

    context_sampler = ContextSampler(
            [gravity_dist1, length_dist1],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_b1 = context_sampler.sample_contexts(4)
    context_sampler.seed(123)
    eval_contexts_b1 = context_sampler.sample_contexts(34)

    context_sampler = ContextSampler(
            [gravity_dist1, length_dist2],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_b2 = context_sampler.sample_contexts(3)
    contexts_b2 = dict((key+4, value) for (key, value) in contexts_b2.items())
    context_sampler.seed(123)
    eval_contexts_b2 = context_sampler.sample_contexts(33)
    eval_contexts_b2 = dict((key+34, value) for (key, value) in eval_contexts_b2.items())

    context_sampler = ContextSampler(
            [gravity_dist2, length_dist1],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_b3 = context_sampler.sample_contexts(3)
    contexts_b3 = dict((key+7, value) for (key, value) in contexts_b3.items())
    context_sampler.seed(123)
    eval_contexts_b3 = context_sampler.sample_contexts(33)
    eval_contexts_b3 = dict((key+67, value) for (key, value) in eval_contexts_b3.items())
    contexts_b1.update(contexts_b2)
    contexts_b1.update(contexts_b3)
    contexts_b = contexts_b1
    eval_contexts_b1.update(eval_contexts_b2)
    eval_contexts_b1.update(eval_contexts_b3)
    eval_contexts_b = eval_contexts_b1

    # Mode C: Full range for one value, constant for the other
    context_sampler = ContextSampler(
            [gravity_constant, length_dist],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_c1 = context_sampler.sample_contexts(5)
    context_sampler.seed(123)
    eval_contexts_c1 = context_sampler.sample_contexts(50)

    context_sampler = ContextSampler(
            [gravity_dist, length_constant],
            context_space=CARLCartPole.get_context_space(),
            seed=42
        )
    contexts_c2 = context_sampler.sample_contexts(5)
    contexts_c2 = dict((key+5, value) for (key, value) in contexts_c2.items())
    contexts_c2.update(contexts_c1)
    contexts_c = contexts_c2
    context_sampler.seed(123)
    eval_contexts_c2 = context_sampler.sample_contexts(50)
    eval_contexts_c2 = dict((key+50, value) for (key, value) in eval_contexts_c2.items())
    eval_contexts_c2.update(eval_contexts_c1)
    eval_contexts_c = eval_contexts_c2

    gravity_test_dist = carl.context.context_space.UniformFloatContextFeature(
                    "gravity",
                    lower=5.0,
                    upper=12.0,
                )
    length_test_dist = carl.context.context_space.UniformFloatContextFeature(
                    "length",
                    lower=0.1,
                    upper=0.9,
                )

    context_sampler = ContextSampler(
            [gravity_test_dist, length_test_dist],
            context_space=CARLCartPole.get_context_space(),
            seed=101
        )
    test_set = context_sampler.sample_contexts(500)

    output_dir = Path("examples") / "carl_generalization_example" / "context_sets"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_contexts_a.json", "w") as f:
        json.dump(contexts_a, f)
    with open(output_dir / "eval_contexts_a.json", "w") as f:
        json.dump(eval_contexts_a, f)
    with open(output_dir / "train_contexts_b.json", "w") as f:
        json.dump(contexts_b, f)
    with open(output_dir / "eval_contexts_b.json", "w") as f:
        json.dump(eval_contexts_b, f)
    with open(output_dir / "train_contexts_c.json", "w") as f:
        json.dump(contexts_c, f)
    with open(output_dir / "eval_contexts_c.json", "w") as f:
        json.dump(eval_contexts_c, f)
    with open(output_dir / "test_contexts.json", "w") as f:
        json.dump(test_set, f)

if __name__ == "__main__":
    if not Path("examples/carl_generalization_example/context_sets/test_contexts.json").exists():
        print("Sampling context sets for CARL CartPole generalization protocol...")
        sample_context_sets()
    else:
        print("Context sets already exist. Skipping sampling.")