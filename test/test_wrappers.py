from __future__ import annotations

import gymnasium as gym

from mighty.mighty_utils.wrappers import ContextualVecEnv


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
