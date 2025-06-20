"""Eureka: LLM-coded reward functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from omegaconf import OmegaConf

from mighty.mighty_runners.mighty_runner import MightyRunner
from mighty.mighty_utils.llm_prompting import get_llm

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Eureka(MightyRunner):
    """Eureka: LLM-coded reward functions."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.iterations = cfg.iterations
        local_llm = cfg.local_llm
        if local_llm:
            self.model, self.prompting_function = get_llm(cfg.model_name)
            self.generation_kwargs = OmegaConf.to_container(cfg.generation_kwargs)
        else:
            raise NotImplementedError("Currently only local models supported.")
        if "prompt_dir" in cfg:
            self.prompt_dir = cfg.prompt_dir
        else:
            self.prompt_dir = Path(__file__).parent.absolute() / "eureka_prompts"

        with open(self.prompt_dir / "user_prompt.txt", "r") as f:
            self.user_prompt = f.read()
        with open(self.prompt_dir / "system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        # TODO: adapt these to task
        self.prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def adapt_reward_function(self):
        response = self.prompting_function(
            self.model, self.prompt, **self.generation_kwargs
        )
        # TODO: get response to reward function

    def gather_feedback(self):
        # TODO: make prompt
        response = self.prompting_function(
            self.model, self.prompting_function, **self.generation_kwargs
        )
        # TODO: feedback response to next prompt

    def run(self) -> Tuple[Dict, Dict]:
        for _ in range(self.iterations):
            self.adapt_reward_function()
            train_results = self.train(self.num_steps)
            self.gather_feedback(train_results)
            eval_results = self.evaluate()
        return train_results, eval_results
