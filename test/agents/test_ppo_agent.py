from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Import the classes to test
from mighty.mighty_agents.ppo import MightyPPOAgent

# Import shared test helpers
from mighty.mighty_utils.test_helpers import DummyContinuousEnv, DummyEnv, clean


class TestPPOAgent:
    def test_init_continuous(self):
        """Test PPO agent initialization with continuous actions."""
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_continuous")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs={
                "buffer_size": 256,
            },
        )

        # Test initialization parameters
        assert agent.gamma == 0.99, "Gamma should be 0.99"
        assert agent.ppo_clip == 0.2, "PPO clip should be 0.2"
        assert agent.n_epochs == 4, "N epochs should be 4"
        assert agent.learning_rate == 3e-4, "Learning rate should be 3e-4"

        # Test that model is initialized
        assert agent.model is not None, "Model should be initialized"
        assert agent.model.continuous_action is True, "Should be continuous action"
        assert agent.update_fn is not None, "Update function should be initialized"

        # Test basic step functionality
        test_obs, _ = env.reset()
        metrics = {
            "env": agent.env,
            "step": 0,
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }

        prediction = agent.step(test_obs, metrics)[0]
        assert len(prediction) == 1, "Prediction should have shape (1, 2)"
        assert prediction.shape[1] == 2, "Action dimension should be 2"

        clean(output_dir)

    def test_init_discrete(self):
        """Test PPO agent initialization with discrete actions."""
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_discrete")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs={
                "buffer_size": 256,
            },
        )

        # Test initialization parameters
        assert agent.model.continuous_action is False, "Should be discrete action"
        assert agent.discrete_action is True, "Discrete action flag should be True"

        # Test basic step functionality
        test_obs, _ = env.reset()
        metrics = {
            "env": agent.env,
            "step": 0,
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }

        prediction = agent.step(test_obs, metrics)[0]
        assert len(prediction) == 1, "Prediction should have shape (1,)"
        assert (
            0 <= prediction[0] < 4
        ), "Action should be in valid range [0, 4)"  # Updated for 4 actions

        clean(output_dir)

    def test_update(self):
        """Test PPO agent update functionality with manual data collection."""
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_update_simple")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=2,
            batch_size=64,
            rollout_buffer_kwargs={
                "buffer_size": 128,
            },
        )

        metrics = {
            "env": agent.env,
            "step": 0,
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }

        # Store original parameters
        original_params = deepcopy(list(agent.model.policy_head.parameters()))
        original_value_params = deepcopy(list(agent.model.value_head.parameters()))

        # Manually collect data to fill the buffer
        curr_s, _ = env.reset(seed=42)

        # Collect enough transitions to fill the buffer
        for step in range(128):  # Fill buffer with 128 transitions
            # Get action from agent
            action, log_prob = agent.step(curr_s, metrics)

            # Take environment step
            next_s, reward, terminated, truncated, _ = env.step(action)
            dones = np.logical_or(terminated, truncated)

            # Process the transition (this adds to buffer)
            agent.process_transition(
                curr_s,
                action,
                reward,
                next_s,
                dones,
                log_prob.detach().cpu().numpy(),
                {"step": step},
            )

            # Update current state
            curr_s = next_s

            # Reset environment if done
            if np.any(dones):
                curr_s, _ = env.reset()

        print(f"Buffer size after manual collection: {len(agent.buffer)}")

        # Ensure we have enough data in buffer
        assert (
            len(agent.buffer) >= agent._batch_size
        ), f"Buffer size {len(agent.buffer)} should be >= batch size {agent._batch_size}"

        # Perform update
        update_kwargs = {"next_s": curr_s, "dones": np.array([False])}

        # Call the update method
        result_metrics = agent.update(metrics, update_kwargs)

        print(f"Update completed. Returned metrics keys: {list(result_metrics.keys())}")

        # Check that parameters have changed
        new_params = list(agent.model.policy_head.parameters())
        new_value_params = list(agent.model.value_head.parameters())

        # Check if policy parameters changed
        params_changed = False
        for old, new in zip(original_params, new_params):
            if not torch.allclose(old, new, atol=1e-6):
                params_changed = True
                print(
                    f"Policy parameter changed: max diff = {torch.max(torch.abs(old - new)).item()}"
                )
                break

        # Check if value parameters changed
        value_params_changed = False
        for old, new in zip(original_value_params, new_value_params):
            if not torch.allclose(old, new, atol=1e-6):
                value_params_changed = True
                print(
                    f"Value parameter changed: max diff = {torch.max(torch.abs(old - new)).item()}"
                )
                break

        # Assertions
        assert params_changed, "Policy parameters should change after update"
        assert value_params_changed, "Value parameters should change after update"
        assert isinstance(result_metrics, dict), "Update should return metrics dict"

        # Check for expected metrics in the result
        expected_metrics = [
            "Update/policy_loss",
            "Update/value_loss",
            "Update/entropy",
            "Update/approx_kl",
        ]
        for metric in expected_metrics:
            assert metric in result_metrics, f"Should have {metric} metric"
            print(f"{metric}: {result_metrics[metric]}")

        print("All parameter update checks passed!")

        clean(output_dir)

    def test_properties(self):
        """Test agent properties."""
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_properties")
        output_dir.mkdir(parents=True, exist_ok=True)

        ppo = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs={
                "buffer_size": 256,
            },
        )

        # Test parameters property
        params = ppo.parameters
        assert isinstance(params, list), "Parameters should be a list"
        assert len(params) > 0, "Should have parameters"
        assert all(
            isinstance(p, torch.nn.Parameter) for p in params
        ), "All should be Parameters"

        # Test value_function property - updated for new structure
        value_fn = ppo.value_function
        assert (
            value_fn is ppo.model.value_function_module
        ), "Value function should be model's value function module"

        # Test that the value function wrapper works - use correct obs shape
        obs_shape = env.single_observation_space.shape[
            0
        ]  # Get actual obs shape from env
        dummy_obs = torch.rand((1, obs_shape))
        value_output = value_fn(dummy_obs)
        assert value_output.shape == (1, 1), "Value function should output shape (1, 1)"

        clean(output_dir)

    def test_reproducibility(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent")
        output_dir.mkdir(parents=True, exist_ok=True)
        ppo = MightyPPOAgent(output_dir, env, batch_size=2, seed=42)
        init_params = deepcopy(list(ppo.model.parameters()))

        metrics = {
            "env": ppo.env,
            "step": 0,
            "hp/lr": ppo.learning_rate,
            "hp/pi_epsilon": ppo._epsilon,
            "hp/batch_size": ppo._batch_size,
            "hp/learning_starts": ppo._learning_starts,
        }
        curr_s, _ = env.reset(seed=42)
        # Collect enough transitions to fill the buffer
        for step in range(128):  # Fill buffer with 128 transitions
            # Get action from agent
            action, log_prob = ppo.step(curr_s, metrics)

            # Take environment step
            next_s, reward, terminated, truncated, _ = env.step(action)
            dones = np.logical_or(terminated, truncated)

            # Process the transition (this adds to buffer)
            ppo.process_transition(
                curr_s,
                action,
                reward,
                next_s,
                dones,
                log_prob.detach().cpu().numpy(),
                {"step": step},
            )

            # Update current state
            curr_s = next_s

            # Reset environment if done
            if np.any(dones):
                curr_s, _ = env.reset()

        update_kwargs = {"next_s": curr_s, "dones": np.array([False])}
        original_metrics = ppo.update(metrics, update_kwargs)
        original_params = deepcopy(list(ppo.model.parameters()))
        
        for _ in range(3):
            env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
            output_dir = Path("test_ppo_agent")
            output_dir.mkdir(parents=True, exist_ok=True)
            ppo = MightyPPOAgent(output_dir, env, batch_size=2, seed=42)
            for old, new in zip(init_params[:10], list(ppo.model.parameters())[:10], strict=False):
                assert not torch.allclose(old, new), "Parameter initialization should be the same with same seed"
            
            metrics = {
                "env": ppo.env,
                "step": 0,
                "hp/lr": ppo.learning_rate,
                "hp/pi_epsilon": ppo._epsilon,
                "hp/batch_size": ppo._batch_size,
                "hp/learning_starts": ppo._learning_starts,
            }
            curr_s, _ = env.reset(seed=42)
            # Collect enough transitions to fill the buffer
            for step in range(128):  # Fill buffer with 128 transitions
                # Get action from agent
                action, log_prob = ppo.step(curr_s, metrics)

                # Take environment step
                next_s, reward, terminated, truncated, _ = env.step(action)
                dones = np.logical_or(terminated, truncated)

                # Process the transition (this adds to buffer)
                ppo.process_transition(
                    curr_s,
                    action,
                    reward,
                    next_s,
                    dones,
                    log_prob.detach().cpu().numpy(),
                    {"step": step},
                )

                # Update current state
                curr_s = next_s

                # Reset environment if done
                if np.any(dones):
                    curr_s, _ = env.reset()

            update_kwargs = {"next_s": curr_s, "dones": np.array([False])}
            new_metrics = ppo.update(metrics, update_kwargs)

            for old, new in zip(original_params[:10], list(ppo.model.parameters())[:10], strict=False):
                assert not torch.allclose(old, new), "Model parameters should stay the same with same seed"

            for old, new in zip(original_metrics["Update/value_loss"], new_metrics["Update/value_loss"], strict=False):
                assert torch.allclose(old, new), "Value loss should be the same with same seed"
            for old, new in zip(original_metrics["Update/policy_loss"], new_metrics["Update/policy_loss"], strict=False):
                assert torch.allclose(old, new), "Policy loss should be the same with same seed"
            for old, new in zip(original_metrics["Update/entropy"], new_metrics["Update/entropy"], strict=False):
                assert torch.allclose(old, new), "Entropy should be the same with same seed"
        clean(output_dir)