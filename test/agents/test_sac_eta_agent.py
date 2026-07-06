from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mighty.mighty_agents.sac_eta import MightySACEtaAgent
from mighty.mighty_update import SACEtaUpdate
from mighty.mighty_utils.test_helpers import DummyContinuousEnv, clean


class TestSACEtaAgent:
    def test_init_continuous(self):
        """Test SACEta agent initialization with continuous actions."""
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_sac_eta_agent_continuous")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightySACEtaAgent(
            output_dir=output_dir,
            env=env,
            gamma=0.99,
            alpha_lr=3e-4,
            target_risk_ratio=0.2,
            eta_lr=1e-3,
        )

        assert agent.gamma == 0.99, "Gamma should be 0.99"
        assert agent.target_risk_ratio == 0.2, "Target risk ratio should be 0.2"
        assert agent.eta_lr == 1e-3, "Eta learning rate should be 1e-3"

        assert agent.model is not None, "Model should be initialized"
        assert isinstance(agent.update_fn, SACEtaUpdate), (
            "Update function should be a SACEtaUpdate"
        )
        assert agent.update_fn.target_risk_ratio == 0.2
        assert agent.update_fn.eta_optimizer.param_groups[0]["lr"] == 1e-3

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

    def test_update(self):
        """Test SACEta agent update including the eta parameter."""
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_sac_eta_agent_update")
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = MightySACEtaAgent(
            output_dir=output_dir,
            env=env,
            gamma=0.99,
            alpha_lr=3e-4,
            batch_size=32,
            learning_starts=16,
            update_every=1,
            n_gradient_steps=1,
        )

        metrics = {
            "env": agent.env,
            "step": 0,
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }

        initial_log_eta = agent.update_fn.log_eta.clone()

        # Manually collect data to fill the buffer
        curr_s, _ = env.reset(seed=42)
        for step in range(50):
            agent.steps = step
            action, log_prob = agent.step(curr_s, metrics)
            next_s, reward, terminated, truncated, _ = env.step(action)
            dones = np.logical_or(terminated, truncated)
            transition_metrics = {
                "step": step,
                "transition": {"terminated": terminated},
            }
            agent.process_transition(
                curr_s,
                action,
                reward,
                next_s,
                dones,
                log_prob.detach().cpu().numpy(),
                transition_metrics,
            )
            curr_s = next_s
            if np.any(dones):
                curr_s, _ = env.reset()

        assert len(agent.buffer) >= agent.batch_size

        # Satisfy learning_starts condition and update
        agent.steps = agent.learning_starts + agent.update_every
        result_metrics = agent.update_agent(next_s=curr_s, dones=np.array([False]))

        assert "Update/eta_loss" in result_metrics, "Eta loss should be logged"
        assert "Update/eta_value" in result_metrics, "Eta value should be logged"
        assert np.isfinite(result_metrics["Update/eta_loss"])
        assert result_metrics["Update/eta_value"] > 0.0

        eta_changed = not torch.allclose(
            initial_log_eta, agent.update_fn.log_eta, atol=1e-8
        )
        assert eta_changed, "Eta should change after the agent update"

        clean(output_dir)
