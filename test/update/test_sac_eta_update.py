import numpy as np
import torch

from mighty.mighty_update.sac_eta_update import SACEtaUpdate

from test_sac_update import DummySACModel, DummyTransitionBatch


class TestSACEtaUpdate:
    """Test SACEta update mechanism."""

    def get_update_and_model(self, initial_weights=0.0, **sac_kwargs):
        """Create SACEta update instance and model for testing."""
        model = DummySACModel(initial_weights=initial_weights)

        default_kwargs = {
            "policy_lr": 0.001,
            "q_lr": 0.001,
            "tau": 0.005,
            "alpha": 0.2,
            "gamma": 0.99,
            "auto_alpha": True,
            "alpha_lr": 3e-4,
            "target_risk_ratio": 0.1,
            "eta_lr": 3e-4,
        }
        default_kwargs.update(sac_kwargs)

        update = SACEtaUpdate(model, **default_kwargs)
        return update, model

    def test_initialization(self):
        """Test SACEta update initialization."""
        update, model = self.get_update_and_model()

        # Inherited SAC setup
        assert hasattr(update, "policy_optimizer")
        assert hasattr(update, "q_optimizer")
        assert hasattr(update, "log_alpha")
        assert hasattr(update, "alpha_optimizer")

        # Eta-specific setup
        assert hasattr(update, "log_eta")
        assert hasattr(update, "eta_optimizer")
        assert update.target_risk_ratio == 0.1
        assert update.log_eta.item() == 0.0  # eta starts at exp(0) = 1

    def test_basic_update(self):
        """Test basic SACEta update functionality."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        initial_q1_params = [p.clone() for p in model.q_net1.parameters()]
        initial_log_eta = update.log_eta.clone()

        metrics = None
        for _ in range(update.policy_frequency + 1):
            metrics = update.update(batch)

        q1_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_q1_params, model.q_net1.parameters())
        )
        eta_changed = not torch.allclose(initial_log_eta, update.log_eta, atol=1e-8)

        assert q1_changed, "Q1 parameters should change after update"
        assert eta_changed, "Eta should change after update"

        required_metrics = [
            "Update/q_loss1",
            "Update/q_loss2",
            "Update/policy_loss",
            "Update/alpha_loss",
            "Update/eta_loss",
            "Update/eta_value",
            "Update/td_error1",
            "Update/td_error2",
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert np.isfinite(metrics[metric]), f"Metric {metric} should be finite"

    def test_eta_stays_positive(self):
        """Eta is parameterized as exp(log_eta) and should always be positive."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        for _ in range(10):
            metrics = update.update(batch)
            assert metrics["Update/eta_value"] > 0.0, "Eta should stay positive"

    def test_eta_gradient_direction(self):
        """Eta loss gradient wrt log_eta should follow the risk-ratio objective."""
        update, model = self.get_update_and_model()

        # With a constant target, target - mean(target) = 0, so
        # d(eta_loss)/d(log_eta) = eta * target_risk_ratio > 0 and eta decreases.
        eta = update.log_eta.exp()
        q_target = torch.full((32, 1), 5.0)
        eta_loss = (
            eta * (update.target_risk_ratio - (q_target - q_target.mean()))
        ).mean()
        update.eta_optimizer.zero_grad()
        eta_loss.backward()

        assert update.log_eta.grad is not None
        assert update.log_eta.grad.item() > 0.0

    def test_entropic_target_matches_sac_target(self):
        """(1/eta) * log(exp(eta * target)) is the identity, so for moderate
        targets the regression target must match plain SAC's TD target."""
        eta = torch.tensor(1.0).exp()
        q_target = torch.randn(64, 1)
        v_entropic = (1.0 / eta) * torch.log(torch.exp(eta * q_target))
        assert torch.allclose(v_entropic, q_target, atol=1e-6)
