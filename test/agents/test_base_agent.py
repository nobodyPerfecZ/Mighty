from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import pytest

from mighty.mighty_agents.dqn import MightyAgent, MightyDQNAgent
from mighty.mighty_utils.test_helpers import DummyEnv, clean
from mighty.mighty_utils.wrappers import ContextualVecEnv


@dataclasses.dataclass
class _NonSerializableInstance:
    """Mimics DACbench instance dataclasses that hold non-JSON objects."""

    fn: object
    coeffs: list


class _DACInstanceEnv(DummyEnv):
    """DummyEnv whose instance_set holds a non-serializable dataclass,
    mirroring DACbench's FunctionApproximationInstance (see issue #123)."""

    def __init__(self):
        super().__init__()
        self.instance_set = {
            0: _NonSerializableInstance(fn=lambda x: x, coeffs=[1.0, 2.0])
        }


class TestMightyAgent:
    def test_init(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_base_agent")
        output_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(NotImplementedError):
            MightyAgent(
                output_dir,
                env,
                meta_kwargs=None,
                wandb_kwargs=None,
                meta_methods=None,
            )
        clean(output_dir)

    def test_make_checkpoint_dir(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_base_agent")
        output_dir.mkdir(parents=True, exist_ok=True)
        agent = MightyDQNAgent(output_dir, env)

        agent.make_checkpoint_dir(1)
        assert Path(agent.checkpoint_dir).exists()
        clean(output_dir)

    def test_apply_config(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_base_agent")
        output_dir.mkdir(parents=True, exist_ok=True)
        agent = MightyDQNAgent(output_dir, env)
        config = {
            "learning_rate": -1,
        }
        agent.apply_config(config)
        assert agent.learning_rate == -1
        clean(output_dir)

    def test_init_dumps_non_serializable_instance_set(self):
        """Regression for #123: DACbench instance sets contain dataclass
        objects that are not JSON-serializable. Agent init must still write
        an instance_set.json artifact rather than crashing."""
        env = ContextualVecEnv([_DACInstanceEnv for _ in range(2)])
        eval_env = ContextualVecEnv([_DACInstanceEnv for _ in range(2)])
        output_dir = Path("test_base_agent")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightyDQNAgent(output_dir, env, eval_env=eval_env)

        instance_file = Path(agent.output_dir) / "instance_set.json"
        assert instance_file.exists()
        with open(instance_file) as f:
            json.load(f)  # must be valid JSON
        clean(output_dir)
