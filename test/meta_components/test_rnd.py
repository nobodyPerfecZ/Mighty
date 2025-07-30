from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mighty.mighty_agents.ppo import MightyPPOAgent
from mighty.mighty_meta.rnd import RND
from mighty.mighty_replay import RolloutBatch
from mighty.mighty_utils.test_helpers import DummyEnv, clean


def dummy_forward(x):
    return np.ones_like(x)


class TestRND:
    def init_rnd(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_rnd")
        output_dir.mkdir(parents=True, exist_ok=True)
        ppo = MightyPPOAgent(
            output_dir,
            env,
            meta_methods=["mighty.mighty_meta.RND"],
        )
        return ppo

    def test_init(self) -> None:
        ppo = self.init_rnd()
        assert len(ppo.meta_modules) == 1, "There should be one meta module."
        assert "RND" in ppo.meta_modules, "RND should be in meta modules."
        assert isinstance(
            ppo.meta_modules["RND"], RND
        ), "RND should be meta module when created from string."
        assert (
            ppo.meta_modules["RND"].rnd_output_dim == 512
        ), "Default output dim should be 512."
        assert (
            len(ppo.meta_modules["RND"].rnd_network_config) == 0
        ), f"Default network config should be empty, got {ppo.meta_modules['RND'].rnd_network_config}."
        assert (
            ppo.meta_modules["RND"].internal_reward_weight == 0.1
        ), "Default internal reward weight should be 0.1."
        assert (
            ppo.meta_modules["RND"].rnd_lr == 0.001
        ), "Default RND learning rate should be 0.001."
        assert (
            ppo.meta_modules["RND"].rnd_eps == 1e-5
        ), "Default RND epsilon should be 1e-5."
        assert (
            ppo.meta_modules["RND"].rnd_weight_decay == 0.01
        ), "Default RND weight decay should be 0.01."
        assert (
            ppo.meta_modules["RND"].update_proportion == 0.5
        ), "Default update proportion should be 0.5."
        assert ppo.meta_modules["RND"].rnd_net is None, "RND network should be None."

        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_rnd")
        output_dir.mkdir(parents=True, exist_ok=True)
        ppo = MightyPPOAgent(
            output_dir,
            env,
            meta_methods=[RND],
            meta_kwargs=[
                {
                    "rnd_output_dim": 12,
                    "rnd_lr": 0.2,
                    "rnd_weight_decay": 0.1,
                    "update_proportion": 0.3,
                    "internal_reward_weight": 0.5,
                    "rnd_network_config": {"test": True},
                    "rnd_eps": 1e-4,
                }
            ],
        )
        assert len(ppo.meta_modules) == 1, "There should be one meta module."
        assert "RND" in ppo.meta_modules, "RND should be in meta modules."
        assert isinstance(
            ppo.meta_modules["RND"], RND
        ), "RND should be meta module when created from class."
        assert ppo.meta_modules["RND"].rnd_output_dim == 12, "Output dim should be 12."
        assert ppo.meta_modules["RND"].rnd_network_config == {
            "test": True
        }, "Network config should be {'test': True}."
        assert (
            ppo.meta_modules["RND"].internal_reward_weight == 0.5
        ), "Internal reward weight should be 0.5."
        assert ppo.meta_modules["RND"].rnd_lr == 0.2, "RND learning rate should be 0.2."
        assert ppo.meta_modules["RND"].rnd_eps == 1e-4, "RND epsilon should be 1e-4."
        assert (
            ppo.meta_modules["RND"].rnd_weight_decay == 0.1
        ), "RND weight decay should be 0.1."
        assert (
            ppo.meta_modules["RND"].update_proportion == 0.3
        ), "Update proportion should be 0.3."
        assert ppo.meta_modules["RND"].rnd_net is None, "RND network should be None."
        clean("test_rnd")

    def test_reward_computation(self) -> None:
        ppo = self.init_rnd()
        dummy_metrics = {
            "transition": {
                "state": np.array([[1.0]]),
                "next_state": np.array([[2.0]]),
                "reward": np.array([0.0]),
                "done": np.array([False]),
            }
        }
        updated_metrics = ppo.meta_modules["RND"].get_reward(dummy_metrics)
        assert (
            ppo.meta_modules["RND"].rnd_net is not None
        ), "RND network should be initialized."
        assert (
            "intrinsic_reward" in updated_metrics["transition"]
        ), "Intrinsic reward should be in updated metrics."
        assert sum(updated_metrics["transition"]["intrinsic_reward"]) == sum(
            updated_metrics["transition"]["reward"]
        ), "Intrinsic reward should be added to base reward."
        clean("test_rnd")

    def test_update(self) -> None:
        ppo = self.init_rnd()

        dummy_metrics = {
            "transition": {
                "state": np.array([[1.0]]),
                "next_state": np.array([[2.0]]),
                "reward": np.array([0.0]),
                "done": np.array([False]),
            },
            "update_batches": [
                RolloutBatch(
                    **{
                        "observations": np.array([[1.0], [5.0], [0.2], [0.3], [0.7]]),
                        "actions": np.array([2.0]),
                        "rewards": np.array([0.0]),
                        "advantages": np.array([0.0]),
                        "returns": np.array([0.0]),
                        "episode_starts": np.array([False]),
                        "log_probs": np.array([0.0]),
                        "values": np.array([0.0]),
                    }
                )
            ],
        }
        ppo.meta_modules["RND"].get_reward(dummy_metrics)
        ppo.meta_modules["RND"].update_proportion = 1.0
        predictor_params = deepcopy(
            list(ppo.meta_modules["RND"].rnd_net.predictor.parameters())
        )
        target_params = deepcopy(
            list(ppo.meta_modules["RND"].rnd_net.target.parameters())
        )

        ppo.meta_modules["RND"].update_predictor(dummy_metrics)
        new_predictor_params = ppo.meta_modules["RND"].rnd_net.predictor.parameters()
        new_target_params = ppo.meta_modules["RND"].rnd_net.target.parameters()

        for new_param, old_param in zip(new_predictor_params, predictor_params):
            assert not torch.allclose(
                new_param, old_param
            ), "Predictor parameters should be updated."
        for new_param, old_param in zip(new_target_params, target_params):
            assert torch.allclose(
                new_param, old_param
            ), "Target parameters should stay fixed."
        clean("test_rnd")
