from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest

from mighty.mighty_agents.dqn import MightyAgent, MightyDQNAgent
from mighty.mighty_utils.test_helpers import DummyEnv, clean


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
