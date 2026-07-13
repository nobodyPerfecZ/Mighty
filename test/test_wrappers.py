from __future__ import annotations

import gymnasium as gym
import numpy as np

from mighty.mighty_utils.wrappers import CARLVectorEnvSimulator, ContextualVecEnv


class _StubCarlEnv:
    """Minimal single env (no num_envs) for CARLVectorEnvSimulator, returning
    a dict info like CARL does (e.g. {'context_id': 0}); see issue #122."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.contexts = {0: {}}

    def reset(self, **kwargs):
        return np.zeros(2), {"context_id": 0}

    def step(self, action):
        return np.zeros(2), 1.0, False, False, {"context_id": 0}

    def close(self):
        pass


class _DacLikeCloseEnv(gym.Env):
    """Env whose close() is not idempotent, like DACbench envs which do
    ``del self.instance_set`` on close (see issue #123)."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Discrete(2)
        self.inst_id = 0
        self.instance_set = {0: 1}

    @property
    def instance_id_list(self):
        return [0]

    def reset(self, *args, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        del self.instance_set  # raises AttributeError if called twice


class TestContextualVecEnv:
    def test_close_is_idempotent(self):
        """Regression for #123: closing the vec env more than once (e.g. via
        both the agent's __del__ and gym's VectorEnv.__del__) must not crash
        on envs whose close() is not idempotent."""
        env = ContextualVecEnv([_DacLikeCloseEnv for _ in range(2)])
        env.close()
        env.close()  # second close must be a no-op
        assert env.closed


class TestCARLVectorEnvSimulator:
    def test_single_env_step_info_is_dict(self):
        """Regression for #122: the single-env path must return infos as a
        dict so the training loop's t.update(infos) works, not a numpy
        object array of dicts."""
        env = CARLVectorEnvSimulator(_StubCarlEnv())
        _, _, _, _, infos = env.step(np.array([0]))
        assert isinstance(infos, dict)
        merged = {"reward": 1.0}
        merged.update(infos)  # must not raise

    def test_single_env_reset_info_is_dict(self):
        """Regression for #122: reset info must be a dict so log_infos'
        info.keys() works."""
        env = CARLVectorEnvSimulator(_StubCarlEnv())
        _, info = env.reset()
        assert isinstance(info, dict)
        assert "context_id" in info
