"""Base agent template."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from rich import print
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from mighty.mighty_exploration import MightyExplorationPolicy
from mighty.mighty_replay import MightyReplay, MightyRolloutBuffer
from mighty.mighty_utils.types import CARLENV, DACENV, MIGHTYENV, retrieve_class

if TYPE_CHECKING:
    from mighty.mighty_utils.types import TypeKwargs


def update_buffer(buffer, new_data):
    for k in buffer.keys():
        buffer[k].append(new_data[k])
    return buffer


def log_to_file(output_dir, result_buffer, hp_buffer, eval_buffer, loss_buffer):
    if loss_buffer is not None:
        loss_df = pd.DataFrame(loss_buffer)
        if (Path(output_dir) / "losses.csv").exists():
            (Path(output_dir) / "losses.csv").unlink()
        loss_df.to_csv(Path(output_dir) / "losses.csv")

    if (Path(output_dir) / "results.npz").exists():
        (Path(output_dir) / "results.npz").unlink()
    np.savez(Path(output_dir) / "results.npz", result_buffer)
    result_df = pd.DataFrame(result_buffer)
    if (Path(output_dir) / "results.csv").exists():
        (Path(output_dir) / "results.csv").unlink()
    result_df.to_csv(Path(output_dir) / "results.csv")

    hp_df = pd.DataFrame(hp_buffer)
    if (Path(output_dir) / "hyperparameters.csv").exists():
        (Path(output_dir) / "hyperparameters.csv").unlink()
    hp_df.to_csv(Path(output_dir) / "hyperparameters.csv")

    eval_df = pd.DataFrame(eval_buffer)
    if (Path(output_dir) / "eval_results.csv").exists():
        (Path(output_dir) / "eval_results.csv").unlink()
    eval_df.to_csv(Path(output_dir) / "eval_results.csv")


class MightyAgent(ABC):
    """Base agent for RL implementations."""

    def __init__(  # noqa: PLR0915, PLR0912
        self,
        output_dir,
        env: MIGHTYENV,  # type: ignore
        seed: int | None = None,
        eval_env: MIGHTYENV | None = None,  # type: ignore
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_wandb: bool = False,
        wandb_kwargs: dict | None = None,
        replay_buffer_class: str
        | DictConfig
        | type[MightyReplay]
        | type[MightyRolloutBuffer]
        | None = None,
        replay_buffer_kwargs: TypeKwargs | None = None,
        meta_methods: list[str | type] | None = None,
        meta_kwargs: list[TypeKwargs] | None = None,
        verbose: bool = True,
    ):
        """Base agent initialization.

        Creates all relevant class variables and calls agent-specific init function

        :param env: Train environment
        :param eval_env: Evaluation environment
        :param learning_rate: Learning rate for training
        :param epsilon: Exploration factor for training
        :param batch_size: Batch size for training
        :param render_progress: Render progress
        :param log_tensorboard: Log to tensorboard as well as to file
        :param log_wandb: Whether to log to wandb
        :param wandb_kwargs: Kwargs for wandb.init, e.g. including the project name
        :param replay_buffer_class: Replay buffer class from coax replay buffers
        :param replay_buffer_kwargs: Arguments for the replay buffer
        :param tracer_class: Reward tracing class from coax tracers
        :param tracer_kwargs: Arguments for the reward tracer
        :param meta_methods: Class names or types of mighty meta learning modules to use
        :param meta_kwargs: List of kwargs for the meta learning modules
        :return:
        """
        if meta_kwargs is None:
            meta_kwargs = []
        if meta_methods is None:
            meta_methods = []
        if wandb_kwargs is None:
            wandb_kwargs = {}
        self.learning_rate = learning_rate
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._learning_starts = learning_starts

        self.buffer: MightyReplay | None = None
        self.policy: MightyExplorationPolicy | None = None

        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.default_rng()

        # Replay Buffer
        replay_buffer_class = retrieve_class(
            cls=replay_buffer_class,
            default_cls=MightyReplay,  # type: ignore
        )
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {  # type: ignore
                "capacity": 1_000_000,
            }
        self.buffer_class = replay_buffer_class
        self.buffer_kwargs = replay_buffer_kwargs

        self.output_dir = output_dir
        self.verbose = verbose

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = Path(self.output_dir) / Path("models")

        # Create meta modules
        self.meta_modules = {}
        for i, m in enumerate(meta_methods):
            meta_class = retrieve_class(cls=m, default_cls=None)  # type: ignore
            assert meta_class is not None, (
                f"Class {m} not found, did you specify the correct loading path?"
            )
            kwargs: Dict = {}
            if len(meta_kwargs) > i:
                kwargs = meta_kwargs[i]
            self.meta_modules[meta_class.__name__] = meta_class(**kwargs)

        self.last_state = None
        self.total_steps = 0

        self.result_buffer = {
            "seed": [],
            "step": [],
            "reward": [],
            "action": [],
            "state": [],
            "next_state": [],
            "terminated": [],
            "truncated": [],
            "mean_episode_reward": [],
        }

        self.eval_buffer = {
            "step": [],
            "seed": [],
            "eval_episodes": [],
            "mean_eval_step_reward": [],
            "mean_eval_reward": [],
            "instance": [],
        }

        self.hp_buffer = {
            "step": [],
            "hp/lr": [],
            "hp/pi_epsilon": [],
            "hp/batch_size": [],
            "hp/learning_starts": [],
            "meta_modules": [],
        }
        self.loss_buffer = None
        starting_hps = {
            "step": 0,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }
        self.hp_buffer = update_buffer(self.hp_buffer, starting_hps)

        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(**wandb_kwargs)
            wandb.log(starting_hps)

        self.initialize_agent()
        self.steps = 0

    def _initialize_agent(self) -> None:
        """Agent/algorithm specific initializations."""
        raise NotImplementedError

    def process_transition(  # type: ignore
        self, curr_s, action, reward, next_s, dones, log_prob=None, metrics=None
    ) -> Dict:
        """Agent/algorithm specific transition operations."""
        raise NotImplementedError

    def initialize_agent(self) -> None:
        """General initialization of tracer and buffer for all agents.

        Algorithm specific initialization like policies etc.
        are done in _initialize_agent
        """
        self._initialize_agent()
        self.buffer = self.buffer_class(**self.buffer_kwargs)  # type: ignore

    def update_agent(self) -> Dict:
        """Policy/value function update."""
        raise NotImplementedError

    def adapt_hps(self, metrics: Dict) -> None:
        """Set hyperparameters."""
        old_hps = {
            "step": self.steps,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }
        self.learning_rate = metrics["hp/lr"]
        self._epsilon = metrics["hp/pi_epsilon"]
        self._batch_size = metrics["hp/batch_size"]
        self._learning_starts = metrics["hp/learning_starts"]

        updated_hps = {
            "step": self.steps,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }
        # TODO: this probably always fails
        if old_hps != updated_hps:
            self.hp_buffer = update_buffer(self.hp_buffer, updated_hps)

    def make_checkpoint_dir(self, t: int) -> None:
        """Checkpoint model.

        :param T: Current timestep
        :return:
        """
        self.upper_checkpoint_dir = Path(self.output_dir) / Path("checkpoints")
        if not self.upper_checkpoint_dir.exists():
            Path(self.upper_checkpoint_dir).mkdir()
        self.checkpoint_dir = self.upper_checkpoint_dir / f"{t}"
        if not self.checkpoint_dir.exists():
            Path(self.checkpoint_dir).mkdir()

    def __del__(self) -> None:
        """Close wandb upon deletion."""
        self.env.close()  # type: ignore
        if self.log_wandb:
            wandb.finish()

    def step(self, observation: torch.Tensor, metrics: Dict) -> torch.Tensor:
        for k in self.meta_modules.keys():
            self.meta_modules[k].pre_step(metrics)

        self.adapt_hps(metrics)
        return self.policy(observation, metrics=metrics, return_logp=True)  # type: ignore

    def update(self, metrics: Dict, update_kwargs: Dict) -> Dict:
        """Update agent."""
        for k in self.meta_modules:
            self.meta_modules[k].pre_update(metrics)

        agent_update_metrics = self.update_agent(**update_kwargs)
        metrics.update(agent_update_metrics)
        metrics = {k: np.array(v) for k, v in metrics.items()}
        metrics["step"] = self.steps

        if self.log_wandb:
            wandb.log(metrics)

        metrics["env"] = self.env
        metrics["vf"] = self.value_function  # type: ignore
        metrics["policy"] = self.policy
        for k in self.meta_modules:
            self.meta_modules[k].post_update(metrics)
        return metrics

    def run(  # noqa: PLR0915
        self,
        n_steps: int,
        eval_every_n_steps: int = 1_000,
        human_log_every_n_steps: int = 5000,
        save_model_every_n_steps: int | None = 5000,
        env: MIGHTYENV = None,  # type: ignore
    ) -> Dict:
        """Run agent."""
        episodes = 0
        if env is not None:
            self.env = env
        # FIXME: can we add the eval result here? Else the evals spam the command line in a pretty ugly way
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "Remaining:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            disable=not self.render_progress,
        ) as progress:
            steps_task = progress.add_task(
                "Train Steps",
                total=n_steps - self.steps,
                start=False,
                visible=False,
            )
            steps_since_eval = 0
            progress.start_task(steps_task)
            # FIXME: this is more of a question: are there cases where we don't want to reset this completely?
            # I can't think of any, can you? If yes, we should maybe add this as an optional argument
            metrics = {
                "env": self.env,
                "vf": self.value_function,  # type: ignore
                "policy": self.policy,
                "step": self.steps,
                "hp/lr": self.learning_rate,
                "hp/pi_epsilon": self._epsilon,
                "hp/batch_size": self._batch_size,
                "hp/learning_starts": self._learning_starts,
            }

            # Reset env and initialize reward sum
            curr_s, _ = self.env.reset()  # type: ignore
            if len(curr_s.squeeze().shape) == 0:
                episode_reward = [0]
            else:
                episode_reward = np.zeros(curr_s.squeeze().shape[0])  # type: ignore

            last_episode_reward = episode_reward
            if not torch.is_tensor(last_episode_reward):
                last_episode_reward = torch.tensor(last_episode_reward).float()
            progress.update(steps_task, visible=True)

            # Main loop: rollouts, training and evaluation
            while self.steps < n_steps:
                metrics["episode_reward"] = episode_reward

                # TODO Remove
                progress.stop()

                action, log_prob = self.step(curr_s, metrics)

                next_s, reward, terminated, truncated, _ = self.env.step(action)  # type: ignore
                dones = np.logical_or(terminated, truncated)

                transition_metrics = self.process_transition(
                    curr_s, action, reward, next_s, dones, log_prob, metrics
                )

                metrics.update(transition_metrics)

                episode_reward += reward

                # Log everything
                t = {
                    "seed": self.seed,
                    "step": self.steps,
                    "reward": reward,
                    "action": action,
                    "state": curr_s,
                    "next_state": next_s,
                    "terminated": terminated.astype(int),
                    "truncated": truncated.astype(int),
                    "mean_episode_reward": last_episode_reward.mean(),
                }
                metrics["episode_reward"] = episode_reward
                self.result_buffer = update_buffer(self.result_buffer, t)

                if self.log_wandb:
                    wandb.log(t)

                for k in self.meta_modules:
                    self.meta_modules[k].post_step(metrics)

                self.steps += len(action)
                metrics["step"] = self.steps
                steps_since_eval += len(action)
                for _ in range(len(action)):
                    progress.advance(steps_task)

                # Update agent
                if (
                    len(self.buffer) >= self._batch_size  # type: ignore
                    and self.steps >= self._learning_starts
                ):
                    update_kwargs = {"next_s": next_s, "dones": dones}

                    metrics = self.update(metrics, update_kwargs)

                # End step
                self.last_state = curr_s
                curr_s = next_s

                # Evaluate
                if eval_every_n_steps and steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.evaluate()

                # Log to command line
                if self.steps % human_log_every_n_steps == 0 and self.verbose:
                    mean_last_ep_reward = np.round(
                        np.mean(last_episode_reward), decimals=2
                    )
                    mean_last_step_reward = np.round(
                        np.mean(mean_last_ep_reward / len(last_episode_reward)),
                        decimals=2,
                    )
                    print(
                        f"""Steps: {self.steps}, Latest Episode Reward: {mean_last_ep_reward}, Latest Step Reward: {mean_last_step_reward}"""  # noqa: E501
                    )

                # Save
                if (
                    save_model_every_n_steps
                    and self.steps % save_model_every_n_steps == 0
                ):
                    self.save(self.steps)
                    log_to_file(
                        self.output_dir,
                        self.result_buffer,
                        self.hp_buffer,
                        self.eval_buffer,
                        self.loss_buffer,
                    )

                if np.any(dones):
                    last_episode_reward = np.where(  # type: ignore
                        dones, episode_reward, last_episode_reward
                    )
                    episode_reward = np.where(dones, 0, episode_reward)  # type: ignore
                    # End episode
                    if isinstance(self.env, DACENV) or isinstance(self.env, CARLENV):
                        instance = self.env.instance  # type: ignore
                    else:
                        instance = None
                    metrics["instance"] = instance
                    episodes += 1
                    for k in self.meta_modules:
                        self.meta_modules[k].post_episode(metrics)

                    # Remove rollout data from last episode
                    # TODO: only do this for finished envs
                    # FIXME: open todo, I think we need to use dones as a mask here
                    # Proposed fix: metrics[k][:, dones] = 0
                    # I don't think this is correct masking and I think we have to check the size of zeros
                    for k in list(metrics.keys()):
                        if "rollout" in k:
                            del metrics[k]

                    # Meta Module hooks
                    for k in self.meta_modules:
                        self.meta_modules[k].pre_episode(metrics)
        log_to_file(
            self.output_dir,
            self.result_buffer,
            self.hp_buffer,
            self.eval_buffer,
            self.loss_buffer,
        )
        return metrics

    def apply_config(self, config: Dict) -> None:
        """Apply config to agent."""
        for n in config:
            algo_name = n.split(".")[-1]
            if hasattr(self, algo_name):
                setattr(self, algo_name, config[n])
            elif hasattr(self, "_" + algo_name):
                setattr(self, "_" + algo_name, config[n])
            elif n in ["architecture", "n_units", "n_layers", "size"]:
                pass
            else:
                print(f"Trying to set hyperparameter {algo_name} which does not exist.")

    # FIXME: as above, logging down here is ugly and we should add it to the progress bar instead
    def evaluate(self, eval_env: MIGHTYENV | None = None) -> Dict:  # type: ignore
        """Eval agent on an environment. (Full rollouts).

        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """

        terminated, truncated = False, False
        options: Dict = {}
        if eval_env is None:
            eval_env = self.eval_env

        state, _ = eval_env.reset(options=options)  # type: ignore
        rewards = np.zeros(eval_env.num_envs)  # type: ignore
        steps = np.zeros(eval_env.num_envs)  # type: ignore
        mask = np.zeros(eval_env.num_envs)  # type: ignore
        while not np.all(mask):
            action = self.policy(state, evaluate=True)  # type: ignore
            state, reward, terminated, truncated, _ = eval_env.step(action)  # type: ignore
            rewards += reward * (1 - mask)
            steps += 1 * (1 - mask)
            dones = np.logical_or(terminated, truncated)
            mask = np.where(dones, 1, mask)

        eval_env.close()  # type: ignore

        if isinstance(self.eval_env, DACENV) or isinstance(self.env, CARLENV):
            instance = eval_env.instance  # type: ignore
        else:
            instance = "None"

        eval_metrics = {
            "step": self.steps,
            "seed": self.seed,
            "eval_episodes": np.array(rewards) / steps,
            "mean_eval_step_reward": np.mean(rewards) / steps,
            "mean_eval_reward": np.mean(rewards),
            "instance": instance,
        }
        self.eval_buffer = update_buffer(self.eval_buffer, eval_metrics)

        # FIXME: this is the ugly I'm talking about
        if self.verbose:
            print("")
            print(
                "------------------------------------------------------------------------------"
            )
            print(
                f"""Evaluation performance after {self.steps} steps:
                {np.round(np.mean(rewards), decimals=2)}"""
            )
            print(
                f"""Evaluation performance per step after {self.steps} steps:
                {np.round(np.mean(rewards / steps), decimals=2)}"""
            )
            print(
                "------------------------------------------------------------------------------"
            )
            print("")

        if self.log_wandb:
            wandb.log(eval_metrics)

        return eval_metrics

    def save(self, t: int) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
