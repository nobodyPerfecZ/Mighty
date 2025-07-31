from __future__ import annotations

import logging
import warnings
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from hydra.utils import get_class

from mighty.mighty_agents.factory import get_agent_class
from mighty.mighty_utils.envs import make_mighty_env

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MightyRunner(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        env=None,
        base_eval_env: Callable = None,
        eval_default: int = None,
    ) -> None:
        """Parse config and run Mighty agent."""
        output_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}_{cfg.seed}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Make train and eval env
        if env is None or base_eval_env is None or eval_default is None:
            if any([env, base_eval_env, eval_default]):
                warnings.warn(
                    "When providing custom envs, all three parameters (env, base_eval_env, eval_default) must be provided. Defaulting to config values.",
                    UserWarning,
                )
            env, base_eval_env, eval_default = make_mighty_env(cfg)

        wrapper_classes = []
        for w in cfg.env_wrappers:
            wkwargs = cfg.wrapper_kwargs if "wrapper_kwargs" in cfg else {}
            cls = get_class(w)
            env = cls(env, **wkwargs)
            wrapper_classes.append((cls, wkwargs))

        def wrap_eval():  # type: ignore
            wrapped_env = base_eval_env()
            for cls, wkwargs in wrapper_classes:
                wrapped_env = cls(wrapped_env, **wkwargs)
            return wrapped_env

        eval_env = wrap_eval()

        # Setup agent
        agent_class = get_agent_class(cfg.algorithm)
        args_agent = dict(cfg.algorithm_kwargs)
        self.agent = agent_class(  # type: ignore
            env=env,
            eval_env=eval_env,
            output_dir=output_dir,
            seed=cfg.seed,
            **args_agent,
        )

        self.eval_every_n_steps = cfg.eval_every_n_steps
        self.num_steps = cfg.num_steps

        # Load checkpoint if one is given
        if cfg.checkpoint is not None:
            self.agent.load(cfg.checkpoint)
            logging.info("#" * 80)
            logging.info(f"Loading checkpoint at {cfg.checkpoint}")

        # Train
        logging.info("#" * 80)
        logging.info(f'Using agent type "{self.agent}" to learn')
        logging.info("#" * 80)

    def train(self, num_steps: int, env=None) -> Any:  # type: ignore
        return self.agent.run(
            n_steps=num_steps, env=env, eval_every_n_steps=self.eval_every_n_steps
        )

    def evaluate(self, eval_env=None) -> Any:  # type: ignore
        return self.agent.evaluate(eval_env)

    def run(self) -> Tuple[Dict, Dict]:
        raise NotImplementedError
