from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mighty.mighty_runners.factory import get_runner_class


@hydra.main("./configs", "base", version_base=None)
def evaluate_mighty(cfg: DictConfig) -> None:
    # Make runner
    runner_cls = get_runner_class(cfg.runner)
    runner = runner_cls(cfg)

    # Run evaluation
    num_eval_instance = len(runner.agent.eval_env.instance_set.keys())
    num_eval_envs = runner.agent.eval_env.unwrapped.num_envs
    for _ in range(num_eval_instance // num_eval_envs):
        runner.evaluate()
    eval_df = pd.DataFrame(runner.agent.eval_buffer)
    eval_df.to_csv(Path(runner.agent.output_dir) / "eval_results.csv")

    # Print stats
    print("Evaluation finished!")
    print(
        f"Reached a reward of {np.round(eval_df['mean_eval_reward'].mean(), decimals=2)}."
    )
    return eval_df["mean_eval_reward"].mean()


if __name__ == "__main__":
    evaluate_mighty()
